#!/usr/bin/env python3

'''
This script is used to show fitting the posnet model using cylceGAN data augmentation method.
'''

import os
import sys
import time
import copy
import yaml
import pickle
import random
import numpy as np
import webdataset as wds
from functools import partial
from pathlib import Path

import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.algorithms.join import Join

sys.path.append('ASD-AFM-dev') # Path to ASD-AFM-dev repo
import asdafm.common_utils          as cu
import asdafm.graph.graph_utils     as gu
import asdafm.preprocessing         as pp
import asdafm.data_loading          as dl
import asdafm.visualization         as vis
from asdafm.graph.models            import PosNetAdaptive
from asdafm.parsing_utils           import update_config
from asdafm.logging                 import LossLogPlot, SyncedLoss

import argparse
sys.path.append('./pytorch-CycleGAN-and-pix2pix') # Path to cycleGAN repo
from models import create_model
from torch.distributed import broadcast

def batch_to_device(batch, device):
    X, ref, *rest = batch
    X = X[0].to(device)
    ref = ref[0].to(device)
    return X, ref, *rest

def batch_to_host(batch):
    X, ref, pred, xyz = batch
    X = X.squeeze(1).cpu()
    ref = ref.cpu()
    pred = pred.cpu()
    return X, ref, pred, xyz

def apply_preprocessing(batch, cfg, gen_ab=None):
    box_res = cfg['box_res']
    z_lims = cfg['z_lims']
    zmin = cfg['zmin']
    peak_std = cfg['peak_std']

    X, atoms, scan_windows = [batch[k] for k in ['X', 'xyz', 'sw']]

    # Pick a random number of slices between 1 and 15
    nz = random.choice(range(1, 16))
    X = [x[:, :, :, -nz:] for x in X]

    atoms = [a[a[:, -1] != 79] for a in atoms] # Get rid of Au substrate
    atoms = pp.top_atom_to_zero(atoms)
    xyz = atoms.copy()
    mols = [gu.MoleculeGraph(a, []) for a in atoms]
    mols, sw = gu.shift_mols_window(mols, scan_windows[0])

    box_borders = (
        (0, 0, z_lims[0]),
        (box_res[0]*(X[0].shape[1] - 1), box_res[1]*(X[0].shape[2] - 1), z_lims[1])
    )
    pp.rand_shift_xy_trend(X, shift_step_max=0.02, max_shift_total=0.04)
    X, mols, box_borders = gu.add_rotation_reflection_graph(X, mols, box_borders, num_rotations=3,
        reflections=True, crop='max', per_batch_item=True)
    # Do style transfer if gen_ab is not None
    pp.style_translate(X, gen_ab, cfg['rank']) if gen_ab is not None else None
    pp.add_norm(X)
    pp.add_gradient(X, c=0.3)
    pp.add_noise(X, c=0.1, randomize_amplitude=True, normal_amplitude=True)
    pp.add_cutout(X, n_holes=5)
    
    mols = gu.threshold_atoms_bonds(mols, zmin)
    ref = gu.make_position_distribution(mols, box_borders, box_res=box_res, std=peak_std)

    return X, [ref], xyz, box_borders


def make_webDataloader(cfg, mode='train', gen_ab=None):
    assert mode in ['train', 'val', 'test'], mode
    shard_list = os.path.join(cfg['data_dir'], cfg['urls'][mode])
    apply_preprocessing_ = partial(apply_preprocessing, cfg=cfg, gen_ab=gen_ab)

    dataset = wds.WebDataset(dl.ShardList(shard_list, world_size=cfg['world_size'], rank=cfg['global_rank'],
        substitute_param=(mode == 'train'), log=Path(cfg['run_dir']) / 'shards.log'))
    dataset.pipeline.pop() # ??
    if mode == 'train': dataset.append(wds.shuffle(10)) # Shuffle order of shards ?
    dataset.append(wds.tariterators.tarfile_to_samples()) # Gather files inside tar files into samples ?
    dataset.append(wds.split_by_worker) # Use a different subset of samples in shards in different workers
    if mode == 'train': dataset.append(wds.shuffle(100)) # Shuffle samples within a worker process
    dataset.append(wds.decode('pill', dl.decode_xyz)) # Decode image and xyz files
    dataset.append(dl.rotate_and_stack(reverse=False)) # Combine separate images into a stack, reverse=True only for QUAM dataset
    dataset.append(dl.batched(cfg['batch_size'])) # Gather samples into batches
    dataset = dataset.map(apply_preprocessing_) # Preprocess

    dataloader = wds.WebLoader(dataset, num_workers=cfg['num_workers'], batch_size=None, pin_memory=True,
        collate_fn=dl.default_collate, persistent_workers=True)
    
    return dataset, dataloader

def make_model(device, cfg):
    model = PosNetAdaptive(
        # This is a simplified model for testing
        encode_block_channels   = [4, 8, 16, 32],
        encode_block_depth      = 2,
        decode_block_channels   = [32, 16, 8],
        decode_block_depth      = 2,
        decode_block_channels2  = [32, 16, 8],
        decode_block_depth2     = 2,
        attention_channels      = [32, 32, 32],
        res_connections         = True,
        activation              = 'relu',
        padding_mode            = 'zeros',
        pool_type               = 'avg',
        decoder_z_sizes         = [5, 15, 35],
        z_outs                  = [3, 3, 5, 10],
        afm_res                 = cfg['box_res'][0],
        grid_z_range            = cfg['z_lims'],
        peak_std                = cfg['peak_std']
    ).to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    lr_decay_rate = 1e-5
    lr_decay = optim.lr_scheduler.LambdaLR(optimizer, lambda b: 1.0/(1.0+lr_decay_rate*b))
    return model, criterion, optimizer, lr_decay


def cycleGAN_options(options_dict):
    default_values_dict = {
        'dataroot': 'image_input',
        'name': 'HyperTest-resnet_6blocks-2-16-10-0.5',
        'gpu_ids': '-1',
        'checkpoints_dir': './trained_models',
        'model': 'test',
        'input_nc': 1,
        'output_nc': 1,
        'ngf': 16,
        'ndf': 16,
        'netD': 'basic',
        'netG': 'resnet_6blocks',
        'n_layers_D': 3,
        'norm': 'instance',
        'init_type': 'normal',
        'init_gain': 0.02,
        'no_dropout': True,
        'dataset_mode': 'single',
        'direction': 'AtoB',
        'serial_batches': True,
        'num_threads': 0,
        'batch_size': 1,
        'load_size': 256, #
        'crop_size': 256, #
        'max_dataset_size': float("inf"),
        'preprocess': 'resize_and_crop', #
        'no_flip': True,
        'display_winsize': 256,
        'epoch': 'latest',
        'load_iter': 0,
        'verbose': False,
        'suffix': '',
        'use_wandb': False,
        'wandb_project_name': 'CycleGAN-and-pix2pix',
        'results_dir': './image_output/',
        'aspect_ratio': 1.0,
        'phase': 'test',
        'eval': False,
        'num_test': 50,
        'model_suffix': '',
        'isTrain': False, 
        'display_id': -1
    }

    # Ensure that options_dict only alters known arguments, if not you may raise an error
    for key in options_dict.keys():
        if key not in default_values_dict:
            raise ValueError(f"Unknown option {key}")

    # updating default values with options from `options_dict`
    default_values_dict.update(options_dict)

    # Convert dict to Namespace
    opt = argparse.Namespace(**default_values_dict)

    return opt


def run(rank, cfg, gen_ab):
    # Initialize the distributed environment. 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group(cfg['comm_backend'], rank=rank, world_size=cfg['world_size'])
    cfg['rank'] = rank
    cfg['local_rank'] = rank
    cfg['global_rank'] = rank

    start_time = time.perf_counter()  
    # Create a directory called Checkpoints in the run_dir ('.')
    checkpoint_dir = os.path.join(cfg['run_dir'], cfg['checkpoint_dir'])
    if rank == 0:
        if not os.path.exists(cfg['run_dir']):
            os.makedirs(cfg['run_dir'])
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    # Define model, optimizer, and loss
    model, criterion, optimizer, lr_decay = make_model(rank, cfg)
    if rank == 0:
        print(f'CUDA is available = {torch.cuda.is_available()}')
        print(f'Model total parameters: {cu.count_parameters(model)}')

    # Setup checkpointing and load a checkpoint if available
    dist.barrier() # This collective blocks processes until the whole group enters this function
    checkpointer = cu.Checkpointer(model, optimizer, additional_data={'lr_params': lr_decay},
        checkpoint_dir=checkpoint_dir, keep_last_epoch=True)
    init_epoch = checkpointer.epoch 

    # Setup logging
    loss_logger = LossLogPlot(
        log_path=os.path.join(cfg['run_dir'], 'loss_log.csv'),
        plot_path=os.path.join(cfg['run_dir'], 'loss_history.png'),
        loss_labels=cfg['loss_labels'],
        loss_weights=cfg['loss_weights'],
        print_interval=cfg['print_interval'],
        init_epoch=init_epoch,
        stream=open(cfg['batch_log_path'], 'a')
    )


    # Wrap model in DistributedDataParallel.
    model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)   

    if cfg['train']:
        # Create datasets and dataloaders
        train_set, train_loader = make_webDataloader(cfg, 'train', gen_ab)
        val_set, val_loader = make_webDataloader(cfg, 'val') # Validation set is not augmented

        if rank == 0:
            if init_epoch <= cfg['epochs']:
                print(f'\n ========= Starting training from epoch {init_epoch}')
            else:
                print('Model already trained')
        
        for epoch in range(init_epoch, cfg['epochs']+1):
            if rank == 0: print(f'\n === Epoch {epoch}')
            
            # Train
            if cfg['timings'] and rank == 0: t0 = time.perf_counter() # Start timer
            model.train() # model.train() tells your model that you 
                          # are training the model. This helps inform 
                          # layers such as Dropout and BatchNorm, which 
                          # are designed to behave differently during training and evaluation. 
            with Join([model, loss_logger.get_joinable('train')]):
                for ib, batch in enumerate(train_loader):
                    # print('Processing {}-th batch on rank {}...'.format(ib, rank))
                    # Transfer batch to device
                    X, ref, _, _ = batch_to_device(batch, rank)
                    #print(f"Batch size on GPU {rank}: {X.size(0)}")
                    if cfg['timings'] and rank == 0:
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                    
                    # Forward
                    pred = model(X)
                    loss = criterion(pred, ref)
                    
                    if cfg['timings'] and rank == 0: 
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()
                    
                    # Backward
                    optimizer.zero_grad()
                    loss.backward() # Get the grediant
                    optimizer.step() # Update the parameters
                    lr_decay.step() # Adaptive learning rate

                    # Log losses
                    loss_logger.add_train_loss(loss)
                    
                    if cfg['timings'] and rank == 0:
                        torch.cuda.synchronize()
                        t3 = time.perf_counter()
                        print(f'(Train) Load Batch/Forward/Backward: {t1-t0:6f}/{t2-t1:6f}/{t3-t2:6f}')
                        t0 = t3

            # Validate
            if rank == 0:
                val_start = time.perf_counter()
                if cfg['timings']: t0 = val_start
            model.eval()
            with Join([loss_logger.get_joinable('val')]):
                with torch.no_grad():
                    
                    for ib, batch in enumerate(val_loader):
                        
                        # Transfer batch to device
                        X, ref, _, _ = batch_to_device(batch, rank)
                        if cfg['timings']: 
                            torch.cuda.synchronize()
                            t1 = time.perf_counter()
                        
                        # Forward
                        pred = model.module(X)
                        loss = criterion(pred, ref)

                        loss_logger.add_val_loss(loss)
                        
                        if cfg['timings'] and rank == 0:
                            torch.cuda.synchronize()
                            t2 = time.perf_counter()
                            print(f'(Val) Load Batch/Forward: {t1-t0:6f}/{t2-t1:6f}')
                            t0 = t2

            # Write average losses to log and report to terminal
            loss_logger.next_epoch()
            # Save checkpoint
            checkpointer.next_epoch(loss_logger.val_losses[-1][0])
            #if rank == 0: checkpointer.next_epoch(loss_logger.val_losses[-1][0])

    # Return to best epoch, and save model weights
    dist.barrier()
    checkpointer.revert_to_best_epoch()
    if rank == 0:
        torch.save(model.module.state_dict(), save_path := os.path.join(cfg['run_dir'], 'best_model.pth'))
        print(f'\nModel saved to {save_path}')

    print(f'Best validation loss on epoch {checkpointer.best_epoch}: {checkpointer.best_loss}')
    print(f'Average of best {cfg["avg_best_epochs"]} validation losses: '
        f'{np.sort(loss_logger.val_losses[:, 0])[:cfg["avg_best_epochs"]].mean()}')
    

    if cfg['test'] or cfg['predict']:
        test_set, test_loader = make_webDataloader(cfg, 'test') # Test set is not augmented

    if cfg['test']:
        print(f'\n ========= Testing with model from epoch {checkpointer.best_epoch}')

        eval_losses = SyncedLoss(len(loss_logger.loss_labels))
        eval_start = time.perf_counter()
        if cfg['timings']: t0 = eval_start
        
        model.eval()
        with Join([model, eval_losses]):
            with torch.no_grad():
                for ib, batch in enumerate(test_loader):
                    # Transfer batch to device
                    X, ref, _, _ = batch_to_device(batch, rank)
                    
                    if cfg['timings']:
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                    
                    # Forward
                    pred = model(X)
                    loss = criterion(pred, ref)
                    eval_losses.append(loss)

                    if (ib+1) % cfg['print_interval'] == 0: print(f'Test Batch {ib+1}')
                    
                    if cfg['timings']:
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()
                        print(f'(Test) t0/Load Batch/Forward: {t1-t0:6f}/{t2-t1:6f}')
                        t0 = t2

        # Average losses and print
        eval_loss = eval_losses.mean()
        print(f'Test set loss: {loss_logger.loss_str(eval_loss)}')

        # Save test set loss to file
        with open(os.path.join(cfg['run_dir'], 'test_loss.txt'),'w') as f:
            f.write(';'.join([str(l) for l in eval_loss]))

    if cfg['predict'] and rank == 0:
        # Make predictions
        print(f'\n ========= Predict on {cfg["pred_batches"]} batches from the test set')
        counter = 0
        pred_dir = os.path.join(cfg['run_dir'], cfg['prediction_dir'])
        
        with torch.no_grad():
            for ib, batch in enumerate(test_loader):
                if ib >= cfg['pred_batches']: break
                
                # Transfer batch to device
                X, ref, xyz, box_borders = batch_to_device(batch, rank)
                
                # Forward
                pred = model(X)
                loss = criterion(pred, ref)

                # Back to host
                X, ref, pred, xyz = batch_to_host((X, ref, pred, xyz))

                # Save xyzs
                cu.batch_write_xyzs(xyz, outdir=pred_dir, start_ind=counter)
            
                # Visualize predictions
                vis.plot_distribution_grid(pred, ref, box_borders=box_borders, outdir=pred_dir,
                    start_ind=counter)
                vis.make_input_plots([X], outdir=pred_dir, start_ind=counter)

                counter += len(xyz)

    print(f'Done at rank {rank}. Total time: {time.perf_counter() - start_time:.0f}s')

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    # Read config used for posnet
    with open('./config_styleTrans.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = update_config(cfg)
    if not os.path.exists(cfg['run_dir']):
        os.makedirs(cfg['run_dir'])
    # Save config
    with open(os.path.join(cfg['run_dir'], 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)

    # Set random seeds from config file
    torch.manual_seed(cfg['random_seed'])
    random.seed(cfg['random_seed'])
    np.random.seed(cfg['random_seed']) 

    # Start run
    mp.set_start_method('spawn')
    world_size = torch.cuda.device_count()
    cfg['world_size'] = world_size 

    # Load cycle GAN model if style_trans is True
    gen_ab = None
    if cfg['style_trans'] == True:
        print('Style transfer is enabled. Loading cycleGAN model...')
        options_dict = {
            'dataroot': 'image_input',
            'name': 'HyperTest-resnet_6blocks-2-16-10-0.5',
            'gpu_ids': '-1',
            'checkpoints_dir': './trained_models',
            'model': 'test',
            'input_nc': 1,
            'output_nc': 1,
            'ngf': 16,
            'ndf': 16,
            'netD': 'basic',
            'netG': 'resnet_6blocks',
            'n_layers_D': 3,
            'norm': 'instance',
            'init_type': 'normal',
            'init_gain': 0.02,
            'no_dropout': True,
            'dataset_mode': 'single',
            'direction': 'AtoB',
            'serial_batches': True,
            'num_threads': 0,
            'batch_size': 1,
            'load_size': 256,
            'crop_size': 256,
            'max_dataset_size': float("inf"),
            'preprocess': 'resize_and_crop',
            'no_flip': True,
            'display_winsize': 256,
            'epoch': 'latest',
            'load_iter': 0,
            'verbose': False,
            'suffix': '',
            'use_wandb': False,
            'wandb_project_name': 'CycleGAN-and-pix2pix',
            'results_dir': './image_output/',
            'aspect_ratio': 1.0,
            'phase': 'test',
            'eval': False,
            'num_test': 50,
            'model_suffix': '',
            'isTrain': False, 
            'display_id': -1 }
        opt = cycleGAN_options(options_dict)
        opt.gpu_ids = [] 
        gen_ab = create_model(opt) # create a model given opt.model and other options
        gen_ab.setup(opt)  # regular setup: load and print networks; create schedulers
        gen_ab = gen_ab.netG 

    # Distributed Data Parallel
    mp.spawn(run, args=(cfg, gen_ab), nprocs=world_size, join=True)


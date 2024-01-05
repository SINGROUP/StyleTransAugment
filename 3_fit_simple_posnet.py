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
from data import create_dataset
from models import create_model

def batch_to_device(batch, device):
    X, ref, *rest = batch
    X = X[0].to(device)
    ref = ref[0].to(device)
    return X, ref, *rest

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
    if cfg['style_trans'] == True:
        pp.style_translate(X, gen_ab, debug=False)
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


def obtain_cycleGAN_options():
    # Load cycle GAN model options
    parser = argparse.ArgumentParser(description='Test the trained model.')
    parser.add_argument('--dataroot', default='image_input', help='path to images (should \
                        have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--name', type=str, default='HyperTest-resnet_6blocks-2-16-10-0.5', \
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./trained_models', help='models are saved here')
    parser.add_argument('--model', type=str, default='test', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
    parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=16, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=16, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='resnet_6blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', action='store_false', help='no dropout for the generator')

    parser.add_argument('--dataset_mode', type=str, default='single', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--serial_batches', action='store_false', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data') # test code only supports num_threads = 0
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size') # test code only supports batch_size = 1
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_false', help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')

    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--load_iter', type=int, default=0, help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

    # wandb parameters
    parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
    parser.add_argument('--wandb_project_name', type=str, default='CycleGAN-and-pix2pix', help='specify wandb project name')
    parser.add_argument('--results_dir', type=str, default='./image_output/', help='saves results here.')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')

    # Dropout and Batchnorm has different behavioir during training and test.
    parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
    parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
    parser.add_argument('--model_suffix', type=str, default='', help='')
    parser.add_argument('--isTrain', action='store_true', help='') 
    parser.add_argument('--display_id', type=int, default=-1, help='input batch size') # no visdom display

    opt = parser.parse_args()
    return opt


def run(rank, cfg):
    # Initialize the distributed environment. (Fixed pattern)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(cfg['comm_backend'], rank=rank, world_size=cfg['world_size'])
    cfg['rank'] = rank
    cfg['local_rank'] = rank
    cfg['global_rank'] = rank

    # Load cycle GAN model if style_trans is True
    if cfg['style_trans'] == True:
        opt = obtain_cycleGAN_options()
        opt.gpu_ids = [rank] # use local rank as the device ID
        gen_ab = create_model(opt) # create a model given opt.model and other options
        gen_ab.setup(opt)  # regular setup: load and print networks; create schedulers
    
    start_time = time.perf_counter()  
    # Create a directory called Checkpoints in the run_dir ('.')
    checkpoint_dir = os.path.join(cfg['run_dir'], 'Checkpoints/')
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
        val_set, val_loader = make_webDataloader(cfg, 'val', gen_ab)

        if rank == 0:
            if init_epoch <= cfg['epochs']:
                print(f'\n ========= Starting training from epoch {init_epoch}')
            else:
                print('Model already trained')
        
        for epoch in range(init_epoch, cfg['epochs']+1):
            if rank == 0: print(f'\n === Epoch {epoch}')
            
            # Train
            if cfg['timings'] and rank == 0: t0 = time.perf_counter() # Start timer

            # Ready
            model.train() # model.train() tells your model that you 
                          # are training the model. This helps inform 
                          # layers such as Dropout and BatchNorm, which 
                          # are designed to behave differently during training and evaluation. 
            # Go
            with Join([model, loss_logger.get_joinable('train')]):
                for ib, batch in enumerate(train_loader):

                    # Transfer batch to device
                    X, ref, _, _ = batch_to_device(batch, rank)

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
            if rank == 0: checkpointer.next_epoch(loss_logger.val_losses[-1][0])

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
        test_set, test_loader = make_webDataloader(cfg, 'test')

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
        pred_dir = os.path.join(cfg['run_dir'], 'predictions/')
        
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
    # Time interval
    print(f'Done at rank {rank}. Total time: {time.perf_counter() - start_time:.0f}s')

    dist.barrier()
    # DDP trianing step
    dist.destroy_process_group()

if __name__ == '__main__':
    # Read config used for posnet
    with open('./config_styleTrans.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

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

    # Distributed Data Parallel
    mp.spawn(run, args=(cfg, ), nprocs=world_size, join=True)


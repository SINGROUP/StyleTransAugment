#!/usr/bin/env python3
'''
This script is used to load data, and augment the data using style translation.
Usage:
    python 2_augment.py 
'''
import os, sys
sys.path.append( './ASD-AFM-dev')

import asdafm.data_loading          as dl
import asdafm.preprocessing         as pp
import asdafm.graph.graph_utils     as gu

from pathlib import Path
import webdataset as wds

from functools import partial
import random
import matplotlib.pyplot as plt
import yaml

import numpy as np

# CycleGAN model
sys.path.append('./pytorch-CycleGAN-and-pix2pix')
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images

import torch
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist

import time

# Make changes in this function to style translation
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
    pp.style_translate(X, gen_ab, debug=True) if cfg['style_trans'] else None
    pp.add_norm(X)
    pp.add_gradient(X, c=0.3)
    pp.add_noise(X, c=0.1, randomize_amplitude=True, normal_amplitude=True)
    pp.add_cutout(X, n_holes=5)
    
    mols = gu.threshold_atoms_bonds(mols, zmin)
    ref = gu.make_position_distribution(mols, box_borders, box_res=box_res, std=peak_std)

    return X, [ref], xyz, box_borders

def make_webDataloader(cfg, gen_ab, mode='train'):
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


def run(rank, cfg):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(cfg['comm_backend'], rank=rank, world_size=cfg['world_size'])
    torch.cuda.set_device(rank)

    cfg['rank'] = rank
    cfg['local_rank'] = rank
    cfg['global_rank'] = rank

    # Load cycle GAN model if style_trans is True
    if cfg['style_trans'] == True:
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
                'display_id': -1
            }
            opt = cycleGAN_options(options_dict)
            opt.gpu_ids = [rank] # GPU
            gen_ab = create_model(opt) # create a model given opt.model and other options
            gen_ab.setup(opt)  # regular setup: load and print networks; create schedulers
    
    start_time = time.perf_counter()
    train_set, train_loader = make_webDataloader(cfg, gen_ab, 'train')

    # Load a batch of data
    for ib, batch in enumerate(train_loader):
        # Do nothing, but load a batch with style translation
        if ib == 0:
            break

    print(f"Rank {rank} finished in {time.perf_counter() - start_time} seconds")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    data_dir = '/scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped/Water-Au111/'
    with open('./config_styleTrans.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # Rewrite the data_dir based on the system
    cfg['data_dir'] = data_dir

    mp.set_start_method('spawn')
    world_size = torch.cuda.device_count()
    cfg['world_size'] = world_size 

    # Distributed Data Parallel
    mp.spawn(run, args=(cfg, ), nprocs=world_size, join=True)

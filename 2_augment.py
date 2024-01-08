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
def apply_preprocessing(batch, cfg, gen_ab):
    box_res = cfg['box_res']
    z_lims = cfg['z_lims']
    zmin = cfg['zmin']
    peak_std = cfg['peak_std']

    X, atoms, scan_windows = [batch[k] for k in ['X', 'xyz', 'sw']]

    # Pick a random number of slices between 1 and 15
    # nz = random.choice(range(1, 16))
    nz = 5 # Choose 5 slices for now
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
    X, mols, box_borders = gu.add_rotation_reflection_graph(X, mols, box_borders, num_rotations=1,
        reflections=True, crop='max', per_batch_item=True)
    pp.style_translate(X, gen_ab, debug=True)
    pp.add_norm(X)
    
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
    torch.cuda.set_device(rank)

    cfg['rank'] = rank
    cfg['local_rank'] = rank
    cfg['global_rank'] = rank

    opt = obtain_cycleGAN_options()
    opt.gpu_ids = [rank] # GPU
    #opt.gpu_ids = [] # CPU
    gen_ab = create_model(opt) # create a model given opt.model and other options
    gen_ab.setup(opt)  # regular setup: load and print networks; create schedulers

    if cfg['train']:
        train_set, train_loader = make_webDataloader(cfg, gen_ab, 'train')

    for ib, batch in enumerate(train_loader):
        # Do nothing, but load a batch with style translation
        if ib == 0:
            break
    dist.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":

    data_dir = '/scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped/Water-Au111/'
    with open('./config_1.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # Rewrite the data_dir based on the system
    cfg['data_dir'] = data_dir

    mp.set_start_method('spawn')
    world_size = torch.cuda.device_count()
    cfg['world_size'] = world_size 
    # Distributed Data Parallel
    mp.spawn(run, args=(cfg, ), nprocs=world_size, join=True)

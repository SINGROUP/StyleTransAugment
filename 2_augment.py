'''
This script is used to load data, and augment the data using style translation.
Usage:
    python 2_augment.py --dataroot image_input --name HyperTest-resnet_6blocks-2-16-10-0.5 \
       --model test --netG resnet_6blocks  --ngf 16   --no_dropout --input_nc 1 \
       --output_nc 1  --checkpoints_dir trained_models \
       --results_dir image_output

    or

    ./run.sh 
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

# CycleGAN model
sys.path.append('./pytorch-CycleGAN-and-pix2pix')
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
import numpy as np
import torch

# Make changes in this function to style translation
def apply_preprocessing(batch, cfg, gen_ab):
    box_res = cfg['box_res']
    z_lims = cfg['z_lims']
    zmin = cfg['zmin']
    peak_std = cfg['peak_std']

    X, atoms, scan_windows = [batch[k] for k in ['X', 'xyz', 'sw']]

    # Pick a random number of slices between 1 and 15
    # nz = random.choice(range(1, 16))
    nz = 5 
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
    # X is one-element list
    # X[0] is a three-element list
    # X[0][0] is in the shape of (width, hight, layers)
    # print('Info:', X[0][0].shape)
    #pp.add_gradient(X, c=0.3)
    #pp.add_noise(X, c=0.1, randomize_amplitude=True, normal_amplitude=True)
    #pp.add_cutout(X, n_holes=5)
    
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

    # Dataloader
    dataloader = wds.WebLoader(dataset, num_workers=cfg['num_workers'], batch_size=None, pin_memory=True,
        collate_fn=dl.default_collate, persistent_workers=True)
    
    return dataset, dataloader

def main():

    # Load cycle GAN model
    opt = TestOptions().parse()  # get options from command line
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling.
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    gen_ab = create_model(opt) # create a model given opt.model and other options
    gen_ab.setup(opt)  # regular setup: load and print networks; create schedulers
    torch.multiprocessing.set_start_method('spawn') # avoid errors in dataloader ?

    data_dir = '/scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped/Water-Au111/'
    with open('./config_1.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # Rewrite the data_dir based on the system
    cfg['data_dir'] = data_dir
    cfg['world_size'] = 1
    cfg['global_rank'] = 0

    if cfg['train']:
        train_set, train_loader = make_webDataloader(cfg, gen_ab, 'train')
    nz = 5
    for ib, batch in enumerate(train_loader):
        # print('---------------------------------')
        # print(batch[0][0])
        # print(batch[0][0].shape)
        # print(batch[0][0][0].shape)
        # print(batch[0][0][0][0].shape)
        # print(batch[0][0][0][0][:, :, 1].shape)
        # print(batch[1][0].shape)
        # print(batch[1][0].shape[-1]/nz) # nz = 5 in this case
        step = batch[1][0].shape[-1]/nz

        # print(batch[3])
        z_top = batch[3][1][-1]

        # Plot the input and label
        fig, axs = plt.subplots(2, nz, sharey=True, figsize=(10, 4))
        for s in range(nz):
            # Inputs
            img = batch[0][0][0][0][:, :, s]
            axs[0, s].imshow(img)

            # Label
            ref = batch[1][0][0][:, :, int(s*step)]
            axs[1, s].imshow(ref)
        plt.savefig('temp/style_transfored_input-label-ib_{}.png'.format(ib))
        if ib == 4:
            break

if __name__ == "__main__":
    main()

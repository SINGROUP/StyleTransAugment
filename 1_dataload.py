'''
This script is used to test the data loading and preprocessing pipeline.
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

def apply_preprocessing(batch, cfg):
    box_res = cfg['box_res']
    z_lims = cfg['z_lims']
    zmin = cfg['zmin']
    peak_std = cfg['peak_std']

    X, atoms, scan_windows = [batch[k] for k in ['X', 'xyz', 'sw']]

    # Pick a random number of slices between 1 and 15
    # nz = random.choice(range(1, 16))
    nz = 10
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
    pp.add_norm(X)
    pp.add_gradient(X, c=0.3)
    pp.add_noise(X, c=0.1, randomize_amplitude=True, normal_amplitude=True)
    pp.add_cutout(X, n_holes=5)
    
    
    mols = gu.threshold_atoms_bonds(mols, zmin)
    ref = gu.make_position_distribution(mols, box_borders, box_res=box_res, std=peak_std)

    return X, [ref], xyz, box_borders

def make_webDataloader(cfg, mode='train'):
    # Three modes
    assert mode in ['train', 'val', 'test'], mode

    shard_list = os.path.join(cfg['data_dir'], cfg['urls'][mode])
    apply_preprocessing_ = partial(apply_preprocessing, cfg=cfg)

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
    print('Dataloading demostration start...')
    data_dir = '/scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped/Water-Au111/'
    with open('./config_styleTrans.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # Rewrite the data_dir based on the system
    cfg['data_dir'] = data_dir
    cfg['world_size'] = 1
    cfg['global_rank'] = 0

    if cfg['train']:
        train_set, train_loader = make_webDataloader(cfg, 'train')

    # Visualize the input and label
    nz = 10
    fig, axs = plt.subplots(2, nz, sharey=True, figsize=(20, 4))
    plt.tight_layout()
    for ib, batch in enumerate(train_loader):
        step = batch[1][0].shape[-1]/nz
        z_top = batch[3][1][-1]
        for s in range(nz):
            # Inputs
            img = batch[0][0][0][0][:, :, s]
            axs[0, s].imshow(img, cmap='gray')
            axs[0, s].set_xticks([])
            axs[0, s].set_yticks([])

            # Label
            ref = batch[1][0][0][:, :, int(s*step)]
            axs[1, s].imshow(ref, cmap='gray')
            axs[1, s].set_xticks([])
            axs[1, s].set_yticks([])
        plt.savefig('temp/input_label_ib_{}.png'.format(ib))
        # Save several images
        if ib == 4:
            print('Image samples saved to temp.')
            break 

if __name__ == "__main__":
    main()

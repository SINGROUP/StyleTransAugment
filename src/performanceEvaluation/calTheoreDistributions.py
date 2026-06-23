#!/usr/bin/env python
# %%
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from water import read_samples_from_folder 
from water import mean_rdf, mean_adf, mean_distance_distribution, mean_adf_OH, compute_sk_sg_all
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution
from water import cal_all_hydrogen_bonds

from utils import plot_kde_fill, plot_joint_distribution, plot_kde_fill_
from utils import plot_joint_distributions
from utils import plot_joint_distributions_in_row

import seaborn as sns
from scipy.stats import gaussian_kde
# %%
if __name__ == '__main__':
    inputFolder = '../../data/structures/simulations/'
    processedFolder = '../../processed_data/theory_distributions/'
    os.makedirs(processedFolder, exist_ok=True)
    baseOut = '../../results/theoretical_distributions/'
    structures = ['Label']
    show = True
    hist = False

    simcolor = '#ed9d2c'
    expcolor = '#de461c'
    dftcolor = '#2ca3cf'
    bg07color = '#479FB1'
    bv17color = '#6E7CBC'

    all_color = expcolor
    bottom_color = simcolor
    top_color = dftcolor

    plt.rcParams['font.size']=14
    #plt.rcParams['font.family']='Arial'
    plt.rcParams['pdf.fonttype']=42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['text.usetex'] = True # Render text with LaTeX
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # %% 
    for structure in structures:
        # Create the output folders
        figureOut = os.path.join(baseOut, structure)
        os.makedirs(figureOut, exist_ok=True)
        # Processed data output
        npzOut = os.path.join(processedFolder, structure)
        os.makedirs(npzOut, exist_ok=True)

        # Read the samples
        sampleFolder = os.path.join(inputFolder, structure) 
        samples = read_samples_from_folder(sampleFolder)
        print('Calculating for structure: {}'.format(structure))
        print('  Number of input samples: {}'.format(len(samples)))

        fig, axs = plt.subplots(2, 2, figsize=(6, 6))
        axs = axs.flatten()
        # Common parameters for the plots, and output folder
        r_max = 3.5
        #mic = True if structure == 'Label' else False
        mic = False
        bins = 120
        color = '#299035'
        outputFolder = os.path.join(baseOut, structure)
        alpha = 0.3
        # %%
        # O-O distances 
        #z_thresholds = {'All': None,  'Bottom': False, 'Top': True}
        #z_thresholds = {'All': None,  'Bottom': False, 'Top': True}
        z_thresholds = {'All': None, 'Top': True}
        #colors = {'All': all_color,  'Bottom': bottom_color, 'Top': top_color}
        colors = {'All': all_color,  'Top': top_color}
        #linestypes = {'All': '-', 'Top': '-', 'Bottom': '-'}
        linestypes = {'All': '-', 'Bottom': '-', 'Top': '-'}
        #fills = {'All': False, 'Bottom': False, 'Top': False}
        fills = {'All': True, 'Top': True}
        #markers = {'All': 'o',  'Bottom': 's', 'Top': 'p'}
        markers = {'All': 'o',  'Top': 's'}
        num_points = 120
        oo_counts = {}  # Track sample counts for All and Top
        for key, value in z_thresholds.items():
            label = 'OO_distances_{}_{}'.format(key, structure)
            legend = '{}'.format(key) 
            npzFile = '{}/OO_{}.npz'.format(npzOut, key) 
            if os.path.exists(npzFile):
                print('Loading OO_distance from file: {}'.format(npzFile))
                OO_distances = np.load(npzFile)['OO']
            else:
                print('Calculating OO_distance ...')
                OO_distances = mean_rdf(samples, 'O', 'O', r_max=r_max, mic=mic, aboveZthres=value, onlyDistances=True)
                np.savez(npzFile, OO=OO_distances, r_max=r_max)
            oo_counts[key] = len(OO_distances)
            print('  OO_distance_{}: {} values'.format(key, len(OO_distances)))
            if key != 'All' and hist==True:
                axs[0].hist(OO_distances, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=1)
            #sns.kdeplot(OO_distances, ax=axs[0], linewidth=1, label=legend, bw_adjust=1.5, color=colors[key], linestyle=linestypes[key], fill=fills[key], alpha=0.3)
            plot_kde_fill_(ax=axs[0], xmin=0, xmax=r_max, data=OO_distances, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=alpha, marker=markers[key], num_points=num_points)
        axs[0].set_xlabel(rf'$d_\text{{OO}}$ (Å)')
        axs[0].set_ylabel(r'Probability density')
        axs[0].set_ylim(0, 6.5) 
        axs[0].set_yticks([])
        axs[0].set_xlim(2.2, r_max-0.01)
        axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        axs[0].tick_params(which='minor', length=3, width=1, direction='in')
        axs[0].tick_params(which='major', length=5, width=1.2, direction='in')
        #axs[0].legend(frameon=False, ncol=1)
        axs[0].legend(
        frameon=False,
        ncol=1,
        loc='upper left')
        axs[0].tick_params(axis='both', direction='in')
        # Add sample counts with colors
        y_pos = 0.78
        axs[0].text(0.6, y_pos+0.14, fr"$n_\text{{All}} = {oo_counts['All']}$",
                   transform=axs[0].transAxes, ha='left', va='top',
                   fontsize=11, color=colors['All'],
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))
        axs[0].text(0.6, y_pos+0.01, fr"$n_\text{{Top}} = {oo_counts['Top']}$",
                   transform=axs[0].transAxes, ha='left', va='top',
                   fontsize=11, color=colors['Top'],
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))

        # O-H distances
        r_max = 1.25
        num_points = 500
        oh_counts = {}  # Track sample counts for All and Top
        for key, value in z_thresholds.items():
            label = 'OH_distances_{}_{}'.format(key, structure)
            legend = 'OH {} ({})'.format(key, structure) if structure != 'P' else 'OH {} (Reference)'.format(key)
            npzFile = '{}/OH_{}.npz'.format(npzOut, key)
            if os.path.exists(npzFile):
                print('Loading OH_distance from file: {}'.format(npzFile))
                OH_distances = np.load(npzFile)['OH']
            else:
                print('Calculating OH_distance ...')
                OH_distances = mean_rdf(samples, 'O', 'H', r_max=r_max, mic=mic, aboveZthres=value, onlyDistances=True)
                np.savez(npzFile, OH=OH_distances, r_max=r_max)
            oh_counts[key] = len(OH_distances)
            print('  OH_distance_{}: {} values'.format(key, len(OH_distances)))
            if key != 'All' and hist==True:
                axs[1].hist(OH_distances, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=1)
            #sns.kdeplot(OH_distances, ax=axs[1], linewidth=1, label=legend, bw_adjust=1.5, color=colors[key], linestyle=linestypes[key], fill=fills[key], alpha=0.3)
            plot_kde_fill_(ax=axs[1], data=OH_distances, xmin=0, xmax=r_max, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=0.3, marker=markers[key], num_points=num_points)
        axs[1].set_xlabel(rf'$d_\text{{OH}}$ (Å)')
        #axs[1].set_ylabel(r'$\rho(r)$')
        #axs[1].set_ylim(0, 120)
        axs[1].set_yticks([])
        axs[1].set_ylabel(r'Probability density')
        axs[1].set_xlim(0.94, 1.05)
        axs[1].tick_params(axis='both', direction='in')
        axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        axs[1].tick_params(which='minor', length=3, width=1, direction='in')
        axs[1].tick_params(which='major', length=5, width=1.2, direction='in')
        # Add sample counts with colors
        y_pos = 0.78
        axs[1].text(0.6, y_pos + 0.14, fr"$n_\text{{All}} = {oh_counts['All']}$",
                   transform=axs[1].transAxes, ha='left', va='top',
                   fontsize=11, color=colors['All'],
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))
        axs[1].text(0.6, y_pos + 0.01, fr"$n_\text{{Top}} = {oh_counts['Top']}$",
                   transform=axs[1].transAxes, ha='left', va='top',
                   fontsize=11, color=colors['Top'],
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))
        # H-O-H angles
        print('Plotting H-O-H ...')
        firstTwo = False
        onlyAngle = True
        r_max = 1.25
        y_lim = 0.4
        num_points = 500
        hoh_counts = {}  # Track sample counts for All and Top
        for key, value in z_thresholds.items():
            label = "HOH_dist_{}_{}" .format(key, structure)
            legend = 'HOH {} ({})'.format(key, structure) if structure != 'P' else 'HOH {} (Reference)'.format(key)
            npzFile = '{}/HOH_{}.npz'.format(npzOut, key)
            if os.path.exists(npzFile):
                print('Loading HOH_distance from file: {}'.format(npzFile))
                angles = np.load(npzFile)['HOH']
            else:
                print('Calculating HOH_distance ...')
                angles = mean_adf(samples, 'H', 'O', 'H', r_max=r_max, firstTwo=firstTwo, mic=mic, onlyAngle=onlyAngle, aboveZthres=value)
                np.savez(npzFile, HOH=angles)
            hoh_counts[key] = len(angles)
            print('  HOH_angle_{}: {} values'.format(key, len(angles)))
            if key != 'All' and hist==True:
                axs[2].hist(angles, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
            #sns.kdeplot(angles, ax=axs[2], linewidth=1, label=legend, bw_adjust=1.5, color=colors[key], linestyle=linestypes[key], fill=fills[key], alpha=0.3)
            plot_kde_fill_(ax=axs[2], data=angles, xmin=0, xmax=180, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=0.3, marker=markers[key], num_points=num_points)
        axs[2].set_xlabel(rf'$\theta_\text{{HOH}}$ (°)')
        axs[2].set_xlim(98, 112)
        axs[2].set_ylabel('Probability density')
        axs[2].set_yticks([])
        axs[2].tick_params(axis='both', direction='in')
        axs[2].xaxis.set_minor_locator(AutoMinorLocator())
        axs[2].tick_params(which='minor', length=3, width=1, direction='in')
        axs[2].tick_params(which='major', length=5, width=1.2, direction='in')
        # Add sample counts with colors
        y_pos = 0.78
        axs[2].text(0.6, y_pos + 0.14, fr"$n_\text{{All}} = {hoh_counts['All']}$",
                   transform=axs[2].transAxes, ha='left', va='top',
                   fontsize=11, color=colors['All'],
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))
        axs[2].text(0.6, y_pos + 0.01, fr"$n_\text{{Top}} = {hoh_counts['Top']}$",
                   transform=axs[2].transAxes, ha='left', va='top',
                   fontsize=11, color=colors['Top'],
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))

        # %%
        # ZOH angles 
        r_max = 1.25
        y_lim = 0.04
        bins = 120 
        num_points = 45
        zoh_counts = {}  # Track sample counts for All and Top
        for key, value in z_thresholds.items():
            label = "Theta_OH_dist_{}_{}" .format(key, structure)
            legend = r"$\theta_{{\text{{OH}}}}$ {} ({})" .format(key, structure) if structure != 'P' else r"$\theta_{{\text{{OH}}}}$ {} (Reference)" .format(key)
            npzFile = '{}/ZOH_{}.npz'.format(npzOut, key)
            if os.path.exists(npzFile):
                print('Loading Theta_OH_distance from file: {}'.format(npzFile))
                angles = np.load(npzFile)['ZOH']
            else:
                print('Calculating Theta_OH_distance ...')
                angles = mean_adf_OH(samples, r_max=r_max, firstTwo=False, mic=False, onlyAngle=True, aboveZthres=value)
                np.savez(npzFile, ZOH=angles)
            zoh_counts[key] = len(angles)
            print('  ZOH_angle_{}: {} values'.format(key, len(angles)))
            if key != 'All' and hist==True: 
                axs[3].hist(angles, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=1)
            #sns.kdeplot(angles, ax=axs[3], linewidth=1, label=legend, bw_adjust=0.5, color=colors[key], linestyle=linestypes[key], fill=fills[key], alpha=0.3)
            plot_kde_fill_(ax=axs[3], data=angles, xmin=0, xmax=180, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=0.3, marker=markers[key], num_points=num_points)
        axs[3].set_xlabel(rf'$\theta_\text{{ZOH}}$ (°)')
        axs[3].set_xlim(0, 180)
        #axs[3].set_ylabel(r'$\rho(\theta)$')
        axs[3].set_yticks([])
        axs[3].set_ylabel(r'Probability density')
        axs[3].tick_params(axis='both', direction='in')
        axs[3].xaxis.set_minor_locator(AutoMinorLocator())
        axs[3].tick_params(which='minor', length=3, width=1, direction='in')
        axs[3].tick_params(which='major', length=5, width=1.2, direction='in')
        # Add sample counts with colors
        y_pos = 0.78
        axs[3].text(0.6, y_pos + 0.14, fr"$n_\text{{All}} = {zoh_counts['All']}$",
                   transform=axs[3].transAxes, ha='left', va='top',
                   fontsize=11, color=colors['All'],
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))
        axs[3].text(0.6, y_pos + 0.01, fr"$n_\text{{Top}} = {zoh_counts['Top']}$",
                   transform=axs[3].transAxes, ha='left', va='top',
                   fontsize=11, color=colors['Top'],
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))

        fig.subplots_adjust(hspace=0.25, wspace=0.25, left=0.1, bottom=0.1, right=0.95, top=0.98)
        plt.savefig('{}/RDF_ADF_{}.pdf'.format(figureOut, structure))
        plt.savefig('{}/RDF_ADF_{}.png'.format(figureOut, structure), dpi=300)
        plt.savefig('{}/RDF_ADF_{}.svg'.format(figureOut, structure))
        if show: plt.show()
        plt.close() 

        # %%


        ##################
        # H-bonds
        ##################
        hbonds_ = cal_all_hydrogen_bonds(samples)
        distances_ = np.array([hb[3] for hb in hbonds_])
        angles_ = np.array([hb[4] for hb in hbonds_])
        # %%
        x_min, y_min = distances_.min(), angles_.min()
        x_max, y_max = distances_.max(), angles_.max()
        #x_max, y_max = 3.5, 180
        x_label, y_label = rf'$d_{{\text{{O}}_\text{{d}}\text{{O}}_\text{{a}}}}$ (Å)', rf'$\theta_{{\text{{O}}_\text{{d}}\text{{H}}\text{{O}}_\text{{a}}}}$ (°)'
        # Calculate All, Top, and Bottom separately
        hbond_counts = {}  # Track sample counts for All and Top
        for k, (key, value) in enumerate(z_thresholds.items()):
            npzFile = '{}/Hbond_{}.npz'.format(npzOut, key)
            if os.path.exists(npzFile):
                print('Loading Hbond from file: {}'.format(npzFile))
                hbonds = np.load(npzFile)['hbond']
                OO_OHO = np.load(npzFile)['OO_OHO']
            else:
                print('Calculating Hbond ...')
                hbonds = cal_all_hydrogen_bonds(samples, aboveZthres=value, zThresholdO=4.85)
                distances_da = [hb[3] for hb in hbonds]
                angles_dha = [hb[4] for hb in hbonds]
                OO_OHO = np.array([distances_da, angles_dha]).T
                np.savez(npzFile, hbond=hbonds, distance=distances_da, 
                                  angle=angles_dha, OO_OHO=OO_OHO)
            hbond_counts[key] = len(OO_OHO)
            print('  Hbond_{}: {} values'.format(key, len(OO_OHO)))
        # Plot All, Top, and Bottom in one figure
        npz_prefix = f"{npzOut}/Hbond"
        npz_x, npz_y = 'distance', 'angle'
        image_prefix = f"{figureOut}/Hbond_{structure}_overlay"
        plot_joint_distributions(z_thresholds, npz_prefix, npz_x, npz_y, colors, markers,  x_min, x_max, y_min, y_max, x_label, y_label, image_prefix, linestypes, show, sample_counts=hbond_counts)

        # %%
        ##################
        # Order parameter
        ##################
        r_max = 3.5
        sks_, sgs_ = compute_sk_sg_all(samples, r_max=r_max)
        # %%
        x_min, y_min = sks_.min(), sgs_.min()
        x_max, y_max = sks_.max(), sgs_.max()
        # Calculate All, Top, and Bottom separately 
        orderp_counts = {}  # Track sample counts for All and Top
        for k, (key, value) in enumerate(z_thresholds.items()):
            npzFile = '{}/OrderP_{}.npz'.format(npzOut, key)
            if os.path.exists(npzFile): 
                print('Loading OrderParameter from file: {}'.format(npzFile))
                sk_sg = np.load(npzFile)['sk_sg']
                sks, sgs = np.load(npzFile)['sks'], np.load(npzFile)['sgs']
            else:
                print('Calculating OrderParameter ...')
                sks, sgs = compute_sk_sg_all(samples, r_max=r_max, aboveZthres=value)
                sk_sg = np.array([sks, sgs]).T
                np.savez(npzFile, sk_sg=sk_sg, sks=sks, sgs=sgs)
            orderp_counts[key] = len(sk_sg)
            print('  OrderP_{}: {} values'.format(key, len(sk_sg)))
            
        #x_min, y_min = -0.3, 0.993 
        #x_min = 0.994
        #x_max, y_max = 0.006, 1
        #xs, ys = sgs, sks
        xs, ys = sks, sgs
        x_label, y_label = r'$S_k$', r'$S_g$'
        # Plot All, Top, and Bottom in one figure
        npz_prefix = f"{npzOut}/OrderP"
        npz_x, npz_y = 'sks', 'sgs'
        image_prefix = f"{figureOut}/OrderP_{structure}_overlay"
        plot_joint_distributions(z_thresholds, npz_prefix, npz_x, npz_y, colors, markers, x_min, x_max, y_min, y_max, x_label, y_label, image_prefix, linestypes, show, sample_counts=orderp_counts) 
# %%
 

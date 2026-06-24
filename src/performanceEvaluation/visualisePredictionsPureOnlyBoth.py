#!/usr/bin/env python
# %%
import matplotlib.pyplot as plt
import numpy as np
from water import read_xyz_with_atomic_numbers
from ase.data import covalent_radii as radii
from ase.data.colors import jmol_colors
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter
import os, re

show = True

def x_tick_formatter(value, _pos):
    rounded = int(round(value))
    if np.isclose(value, 0.0):
        return ''
    if np.isclose(value, 4.0):
        return ''
    if np.isclose(value, rounded):
        return str(rounded)
    return ''

def structure_x_tick_formatter(value, _pos):
    nm_value = value / 10.0
    rounded = int(round(nm_value))
    if np.isclose(nm_value, 0.0):
        return ''
    if np.isclose(nm_value, 4.0):
        return ''
    if np.isclose(nm_value, rounded):
        return str(rounded)
    return ''

def extract_lambda_values(s):
    """Extract numeric values following 'L' in the input string."""
    return [float(x) if '.' in x else int(x) for x in re.findall(r'_L([0-9.]+)', s)]

# %%
#models = ['Ref_Best', 'PPAFM2Exp_CoAll_L20_L1_Elatest_C1', 'PPAFM2Exp_CoAll_L10_L10_Elatest_C6', 'PPAFM2Exp_CoAll_L50_L1_Elatest']
#models = ['Ref_Pure_C9', 'Ref_Best', 'PPAFM2Exp_CoAll_L10_L10_Elatest_Only_C3', 'PPAFM2Exp_CoAll_L20_L1_Elatest_Only_C4']
models = ['Ref_Pure_C9', 'Ref_best', 'PPAFM2Exp_CoAll_L20_L1_Elatest_Only_C7', 'PPAFM2Exp_CoAll_L20_L1_Elatest_C1']
angles = [0, 90, 180, 270]
samples = ['Ying_Jiang_1', 'Ying_Jiang_2_1', 'Ying_Jiang_2_2', 'Ying_Jiang_3', 'Ying_Jiang_5', 'Ying_Jiang_6'] # 'Ying_Jiang_4'
sizes = [(32, 32), (34, 34), (34, 34), (40, 40), (40,40), (39, 39)] # 'Ying_Jiang_4'
indexes = [[0, 8], [0, 8], [0, 8], [0, 8], [0, 8], [0, 8], [0, 6]] 

# %%
plt.rcParams['font.size']=14
#plt.rcParams['font.family']='Arial'
plt.rcParams['pdf.fonttype']=42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = True # Render text with LaTeX

# %%
expImage = '../../data/expPNG'
expNpz = '../../data/expData'
predictions = '../../data/structures/predictions'
output = '../../results/predictionsPureOnlyBoth'
if not os.path.exists(output):
    os.makedirs(output)

# %%
#for perspective in ['xy', 'xz']:
for perspective in ['xy']:
    # Look different rotations individually
    for angle in angles:
        numRows = len(samples)
        numCols = len(models) + 2 # Add two for the input images
        fig, axs = plt.subplots(numRows, numCols, figsize=(numCols*2, numRows*2+3))

        # Load experimental AFM slices from NPZ and compute a shared color scale.
        exp_slices = {}
        global_min = np.inf
        global_max = -np.inf
        for i, sample in enumerate(samples):
            npz_file = '{}/{}.npz'.format(expNpz, sample)
            npz_data = np.load(npz_file)
            exp_data = npz_data['data']
            close = exp_data[:, :, indexes[i][0]]
            far = exp_data[:, :, indexes[i][1]]
            close = np.rot90(close, k=(angle+90)//90)
            far = np.rot90(far, k=(angle+90)//90)
            x_size_nm = float(npz_data['lengthX']) / 10.0
            y_size_nm = float(npz_data['lengthY']) / 10.0
            exp_slices[sample] = (close, far, x_size_nm, y_size_nm)
            local_min = min(np.nanmin(close), np.nanmin(far))
            local_max = max(np.nanmax(close), np.nanmax(far))
            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)

        shared_norm = Normalize(vmin=global_min, vmax=global_max)
        image_mappable = None

        for i, sample in enumerate(samples):
            x_size, y_size = sizes[i]
            close, far, x_size_nm, y_size_nm = exp_slices[sample]
            image_extent = [0, x_size_nm, 0, y_size_nm]
            # Show the image with corresponding rotation 
            #image_mappable = axs[i, 0].imshow(close, cmap='inferno', norm=shared_norm, extent=image_extent, origin='lower')
            image_mappable = axs[i, 0].imshow(close, cmap='inferno', norm=shared_norm, extent=image_extent)
            axs[i, 1].imshow(far, cmap='inferno', norm=shared_norm, extent=image_extent)
            for image_ax in (axs[i, 0], axs[i, 1]):
                image_ax.xaxis.set_major_locator(MultipleLocator(1.0))
                image_ax.yaxis.set_major_locator(MultipleLocator(1.0))
                image_ax.xaxis.set_minor_locator(MultipleLocator(0.5))
                image_ax.yaxis.set_minor_locator(MultipleLocator(0.5))
                image_ax.xaxis.set_major_formatter(FuncFormatter(x_tick_formatter))
                image_ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
                # Major ticks: long with labels. Minor ticks: short unlabeled half-step marks.
                # Bottom x labels only, but ticks on all sides.
                image_ax.tick_params(which='major', axis='x', direction='in', bottom=True, top=True, labelbottom=True, labeltop=False, labelsize=8, length=5, pad=-13)
                image_ax.tick_params(which='minor', axis='x', direction='in', bottom=True, top=True, length=3)
                image_ax.tick_params(which='major', axis='y', direction='in', left=True, right=True, labelleft=False, labelright=False, length=5)
                image_ax.tick_params(which='minor', axis='y', direction='in', left=True, right=True, length=3)
                image_ax.set_xlim(0, x_size_nm)
                image_ax.set_ylim(0, y_size_nm)
                for tick_label in image_ax.get_xticklabels():
                    if tick_label.get_text() == '0':
                        tick_label.set_horizontalalignment('left')
            axs[i, 0].text(0.15, 0.045, 'nm', transform=axs[i, 0].transAxes,
                               ha='right', va='bottom', fontsize=10, color='white')

            for j, model in enumerate(models):
                structure = '{}/{}/Prediction_c/{}_d{}_mol.xyz'.format(predictions, model, sample, angle)
                atoms = read_xyz_with_atomic_numbers(structure)

                axs[i, j+2].set_aspect('equal')
                axs[i, j+2].xaxis.set_major_locator(MultipleLocator(10.0))
                axs[i, j+2].xaxis.set_minor_locator(MultipleLocator(5.0))
                axs[i, j+2].yaxis.set_major_locator(MultipleLocator(10.0))
                axs[i, j+2].yaxis.set_minor_locator(MultipleLocator(5.0))
                axs[i, j+2].xaxis.set_major_formatter(FuncFormatter(structure_x_tick_formatter))
                axs[i, j+2].tick_params(which='major', axis='x', direction='in', bottom=True, top=True, labelbottom=True, labeltop=False, labelsize=8, length=5, pad=-13)
                axs[i, j+2].tick_params(which='minor', axis='x', direction='in', bottom=True, top=True, length=3)
                axs[i, j+2].tick_params(which='major', axis='y', direction='in', left=True, right=True, labelleft=False, labelright=False, length=5)
                axs[i, j+2].tick_params(which='minor', axis='y', direction='in', left=True, right=True, length=3)
                # Sort atoms by z-position to draw farther atoms first
                atoms = sorted(atoms, key=lambda atom: atom.position[2])

                z_values = np.array([atom.position[2] for atom in atoms])
                z_min, z_max = np.min(z_values), np.max(z_values)
                z_range = z_max - z_min if z_max > z_min else 1.0  # avoid divide-by-zero


                for atom in atoms:
                    color = jmol_colors[atom.number]
                    radius = radii[atom.number]*1.3
                    if perspective == 'xy':
                        x, y, z = atom.position[0], atom.position[1], atom.position[2]
                        # Scale size by z depth (closer atoms are larger)
                        scale = 0.5 + 0.5 * (z - z_min) / z_range  # scale between 0.5x and 1.0x
                        scaled_radius = radius * scale
                    elif perspective == 'xz':
                        x, y = atom.position[0], atom.position[2]  # height view
                        scaled_radius = radius
                    else:
                        raise ValueError("Perspective must be 'xy' or 'xz'")
                    circle = Circle((x, y), scaled_radius, facecolor=color, edgecolor='k', linewidth=0.5)
                    axs[i, j+2].add_patch(circle)

                if j == 0:
                    x_positions = [atom.position[0] for atom in atoms]
                    y_positions = [atom.position[1] if perspective == 'xy' else atom.position[2] for atom in atoms]

                axs[i, j+2].set_xlim([0, x_size])
                axs[i, j+2].set_ylim([0, y_size])
        # Add the sublabels
        #lambdas = [extract_lambda_values(model) for model in models[1:]]
        r'''
        subLabels = [
            '$v$: Exp. AFM (far)',
            'Exp. AFM (close)',
            rf'$F_{{\mathcal{{U}}}}(v)$',
            rf'$F_{{\tilde{{\mathcal{{V}}}}}}^{{\lambda_\mathrm{{c}}, \lambda_\mathrm{{i}} = {lambdas[0][0]}, {lambdas[0][1]}}}(v)$',
            rf'$F_{{\tilde{{\mathcal{{V}}}}}}^{{\lambda_\mathrm{{c}}, \lambda_\mathrm{{i}} = {lambdas[1][0]}, {lambdas[1][1]}}}(v)$',
            rf'$F_{{\tilde{{\mathcal{{V}}}}}}^{{\lambda_\mathrm{{c}}, \lambda_\mathrm{{i}} = {lambdas[2][0]}, {lambdas[2][1]}}}(v)$'
        ]
        '''
        subLabels = [
            '$v$: Exp. AFM (far)',
            'Exp. AFM (close)',
            r'$F_{\mathcal{U}}(v)$: Pure Simulation',
            rf'$F_{{\overline{{\mathcal{{V}}}}}}$: Handcrafted',
            rf'$F_{{\widetilde{{\mathcal{{V}}}}}}$: Style Translated',
            rf'$F_{{\mathcal{{V}}^{{\dagger}}}}$: Hybrid'
        ]
        xoffsets = [-0.14, -0.11, -0.103, -0.055, -0.043, 0.005]
        for j in range(numCols):
            ax = axs[0, j]      # first row's axes
            pos = ax.get_position()
            x_center = (pos.x0 + pos.x1) / 2  # middle between left and right
            y_top = pos.y1 + 0.01             # a little bit above the top

            fig.text(x_center+xoffsets[j], y_top, subLabels[j],
                    ha='left', va='bottom')
        if image_mappable is not None:
            pos_left = axs[0, 0].get_position()
            pos_right = axs[0, 1].get_position()
            cbar_left = pos_left.x0
            cbar_right = pos_right.x1
            cbar_bottom = pos_left.y1 + 0.04
            cbar_height = 0.015
            cax = fig.add_axes([cbar_left, cbar_bottom, cbar_right - cbar_left, cbar_height])
            cbar = fig.colorbar(image_mappable, cax=cax, orientation='horizontal')
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.tick_params(direction='in', pad=2)
            cbar.set_label(r'$\Delta f$ (Hz)')
        plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.025, right=0.975, top=0.86)
        plt.savefig('{}/predictions_{}_{}.png'.format(output, angle, perspective), dpi=600)
        plt.savefig('{}/predictions_{}_{}.pdf'.format(output, angle, perspective))
        plt.savefig('{}/predictions_{}_{}.svg'.format(output, angle, perspective))
        if show: plt.show()

# %%

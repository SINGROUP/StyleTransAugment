#!/usr/bin/env python
# %%
import matplotlib.pyplot as plt
import numpy as np
from water import read_xyz_with_atomic_numbers
from ase.data import covalent_radii as radii
from ase.data.colors import jmol_colors
from matplotlib.patches import Circle
import os, re

show = True

def extract_lambda_values(s):
    """Extract numeric values following 'L' in the input string."""
    return [float(x) if '.' in x else int(x) for x in re.findall(r'_L([0-9.]+)', s)]

# %%
#models = ['Ref_Best', 'PPAFM2Exp_CoAll_L20_L1_Elatest_C1', 'PPAFM2Exp_CoAll_L10_L10_Elatest_C6', 'PPAFM2Exp_CoAll_L50_L1_Elatest']
#models = ['Ref_Pure_C9', 'Ref_Best', 'PPAFM2Exp_CoAll_L10_L10_Elatest_Only_C3', 'PPAFM2Exp_CoAll_L20_L1_Elatest_Only_C4']
models = ['Ref_Pure_C9', 'Ref_best', 'PPAFM2Exp_CoAll_L20_L1_Elatest_Only_C7', 'PPAFM2Exp_CoAll_L20_L1_Elatest_C1']
angles = [0, 90, 180, 270]
samples = ['Ying_Jiang_1', 'Ying_Jiang_2_1', 'Ying_Jiang_2_2', 'Ying_Jiang_3', 'Ying_Jiang_5', 'Ying_Jiang_6'] # 'Ying_Jiang_4'
indexes = [[0, 8], [0, 8], [0, 8], [0, 8], [0, 8], [0, 8], [0, 6]] 

# %%
plt.rcParams['font.size']=14
#plt.rcParams['font.family']='Arial'
plt.rcParams['pdf.fonttype']=42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = True # Render text with LaTeX

# %%
expImage = '../../data/expPNG'
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

        for i, sample in enumerate(samples):
            # Load the input image: close and far 
            closeImg = '{}/{}_{}.png'.format(expImage, sample, indexes[i][0])
            farImg = '{}/{}_{}.png'.format(expImage, sample, indexes[i][1])
            close = plt.imread(closeImg, )
            far = plt.imread(farImg)
            # Rotate the images
            close = np.rot90(close, k=(angle+90)//90)
            far = np.rot90(far, k=(angle+90)//90)
            # Show the image with corresponding rotation 
            axs[i, 0].imshow(close, cmap='inferno')
            axs[i, 1].imshow(far, cmap='inferno')
            # Show no axis
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')

            for j, model in enumerate(models):
                structure = '{}/{}/Prediction_c/{}_d{}_mol.xyz'.format(predictions, model, sample, angle)
                atoms = read_xyz_with_atomic_numbers(structure)

                axs[i, j+2].set_aspect('equal')
                axs[i, j+2].tick_params(axis='both', direction='in', labelright=False)

                axs[i, j+2].set_xticks([])
                axs[i, j+2].set_yticks([])
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

                x_positions = [atom.position[0] for atom in atoms]
                y_positions = [atom.position[1] if perspective == 'xy' else atom.position[2] for atom in atoms]

                if j == 0:
                    xmin, xmax = min(x_positions), max(x_positions)
                    ymin, ymax = min(y_positions), max(y_positions)
                offset = 1

                # Calculate the center and the maximum span to ensure square axes
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                span = max(xmax - xmin, ymax - ymin) / 2 + 6 * offset  # add padding

                axs[i, j+2].set_xlim([x_center - span, x_center + span])
                axs[i, j+2].set_ylim([y_center - span, y_center + span])
                # Draw 1 nm scale bar at top left for the first model
                if j == 0:
                    # 1 nm in Angstroms (10 Å)
                    scale_length = 10  # 1 nm = 10 Å
                    # Place bar 5% from left and 10% from top
                    x0 = x_center - span + 0.05 * (2 * span)
                    y0 = y_center + span - 0.98 * (2 * span)
                    x1 = x0 + scale_length
                    y1 = y0
                    axs[i, j+2].plot([x0, x1], [y0, y1], color='k')
                    axs[i, j+2].text((x0 + x1) / 2, y0 + 0.03 * (2 * span), '1 nm', color='k',
                                ha='center', va='bottom', fontsize=10)
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
            rf'$F_{{\bar{{\mathcal{{V}}}}}}$: Handcrafted',
            rf'$F_{{\tilde{{\mathcal{{V}}}}}}$: Style Translated',
            rf'$F_{{\bar{{\tilde{{\mathcal{{V}}}}}}}}$: Hybrid'
        ]
        xoffsets = [-0.14, -0.11, -0.103, -0.055, -0.043, 0.005]
        for j in range(numCols):
            ax = axs[0, j]      # first row's axes
            pos = ax.get_position()
            x_center = (pos.x0 + pos.x1) / 2  # middle between left and right
            y_top = pos.y1 + 0.01             # a little bit above the top

            fig.text(x_center+xoffsets[j], y_top, subLabels[j],
                    ha='left', va='bottom')
        plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.025, right=0.975)
        plt.savefig('{}/predictions_{}_{}.png'.format(output, angle, perspective), dpi=600)
        plt.savefig('{}/predictions_{}_{}.pdf'.format(output, angle, perspective))
        plt.savefig('{}/predictions_{}_{}.svg'.format(output, angle, perspective))
        if show: plt.show()

# %%

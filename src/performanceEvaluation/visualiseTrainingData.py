#!/usr/bin/env python
from ase.data import covalent_radii as radii
from ase.data.colors import jmol_colors
from matplotlib.patches import Circle
from water import read_xyz_with_atomic_numbers
from water import read_samples_from_folder
from water import calculate_lattice_vectors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from ase.visualize import view
from ase.io import read, write
from ase.visualize.plot import plot_atoms
from ase import Atoms
import os
import numpy as np
import imageio.v3 as iio  

from utils import get_scan_window_from_xyz, draw_unit_cells, draw_3d_axis_indicator
# Input structure for visualization
demoIndex = 0
demoStructure = f'../../data/structures/simulations/Label/{demoIndex}.xyz'
exampleInput = '../../data/overview'
refZ0 = 14.0 # Needs to be confirmed latter
dz = 0.4 # Units: Å

# Output folder for the output structures
figure_folder = '../../results/train_data'
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

# Parameters
sw=((0, 0, 0), (31.875, 31.875, 2.4))
ss = (25.6, 25.6, 2.4)
sw_x, sw_y, sw_z = sw[1][0], sw[1][0], sw[1][2] 
show=False
showScanRegion = False
showImageRegion = True
showLattice = True
showIndicator = True

simcolor = '#ed9d2c'
expcolor = '#de461c'
bg07color = '#479FB1'
bv17color = '#6E7CBC'

plt.rcParams['font.size']=14
#plt.rcParams['font.family']='Arial'
plt.rcParams['pdf.fonttype']=42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = True # Render text with LaTeX


xyz_min, xyz_max = get_scan_window_from_xyz(demoStructure)
xyz_center = xyz_min + (xyz_max - xyz_min)/2

####################################################################
# 1. Plot the atoms of demonstration configuration in the xy plane
####################################################################
atoms = read_xyz_with_atomic_numbers(demoStructure)
view(atoms)
substrate = atoms[atoms.numbers == 79] # Au substrate
subPositions = substrate.get_positions()
lattice_vectors = calculate_lattice_vectors(substrate)
atoms.set_cell(lattice_vectors)
atoms.set_pbc([True, True, False])
print('lattice vector:', lattice_vectors)

repNum = (3, 3, 1)
supercell = atoms.repeat(repNum)
#view(supercell)
fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_aspect('equal')
ax1.tick_params(axis='both', direction='in', labelright=False)

# Move the center
xyz_center = xyz_center + lattice_vectors[0] + lattice_vectors[1] #+ (repNum[2]-1)*lattice_vectors[2]
xy_center = xyz_center[:2]

supercellList = sorted(supercell, key=lambda atom: atom.position[2])
h_flag, o_flag, au_flag = 'H', 'O', 'Au'
for atom in supercellList:
    color = jmol_colors[atom.number]
    radius = radii[atom.number]
    if atom.number == 1:
        circle = Circle((atom.x, atom.y), radius, facecolor=color,
                        edgecolor='k', linewidth=0.5)
    elif atom.number == 8:
        circle = Circle((atom.x, atom.y), radius, facecolor=color, edgecolor='k', linewidth=0.5)
    else:
        circle = Circle((atom.x, atom.y), radius, facecolor=color, alpha=0.5)
    ax1.add_patch(circle)
#print(type(supercell))   
#plot_atoms(supercell, ax1, radii=0.8, show_unit_cell=True)
xy_origin = subPositions[:, :2].min(axis=0)
draw_unit_cells(ax1, origin=np.array([xy_origin[0], xy_origin[1], 0]), cell_vectors=lattice_vectors, nx=3, ny=3)

if showIndicator:
    draw_3d_axis_indicator(ax1, anchor=(0.86, 0.052), length=40)

if showScanRegion:
    sw = ((xy_center[0] - 31.875 / 2, xy_center[1] - 31.875 / 2, sw[0][2]),
          (xy_center[0] + 31.875 / 2, xy_center[1] + 31.875 / 2, sw[1][2]))
    # Extract rectangle coordinates
    (x0, y0) = sw[0][0], sw[0][1]
    (x1, y1) = sw[1][0], sw[1][1]
    # Width and height
    width, height = x1 - x0, y1 - y0
    # Create and add rectangle
    rect = Rectangle((x0, y0), width, height, linewidth=1, edgecolor='r',
                     facecolor='none', label='scan region')
    ax1.add_patch(rect)



if showImageRegion:
    sw = ((xy_center[0] - ss[0] / 2, xy_center[1] - ss[1] / 2, sw[0][2]),
          (xy_center[0] + ss[0] / 2, xy_center[1] + ss[1] / 2, sw[1][2]))
    # Extract rectangle coordinates
    (x0, y0) = sw[0][0], sw[0][1]
    (x1, y1) = sw[1][0], sw[1][1]
    # Width and height
    width = x1 - x0
    height = y1 - y0
    # Create and add rectangle
    rect = Rectangle((x0, y0), width, height, linewidth=1, edgecolor='k',
                     facecolor='none', label='image region')
    ax1.add_patch(rect)
    offset_x = 0.04 * width
    offset_y = 0.04 * height
    ax1.text(x0 - 1.3*offset_x, y1 + 1.3*offset_y, r"$m$", va='top', ha='left')


yticks = np.arange(20, 70.1, 5)
ax1.set_yticks(yticks)
ax1.set_yticklabels([f'{y-30:.0f}' for y in yticks]) 
xticks = np.arange(50, 100, 5)
ax1.set_xticks(xticks)
ax1.set_xticklabels([f'{x-50:.0f}' for x in xticks])
#ax1.set_xticklabels([f'{x/10:.0f}' for x in xticks])

offset = 2.5
ax1.set_xlim([xy_center[0]-sw_x/2-offset, xy_center[0]+sw_x/2+offset])
ax1.set_ylim([xy_center[1]-sw_y/2-offset, xy_center[1]+sw_y/2+offset])
#ax1.set_xlabel(r'$x$ (nm)')
#ax1.set_ylabel(r'$y$ (nm)')
ax1.set_xlabel(r'$x$ (Å)')
ax1.set_ylabel(r'$y$ (Å)')
# ax1.legend()
# # Add the label
# offset_text = 0.05 
# ax1.text(offset_text, 1 - offset_text, "a", transform=ax1.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

#fig.subplots_adjust(hspace=0, wspace=0, left=0.08, bottom=0.15, right=0.99, top=0.95)
plt.tight_layout()
if show: plt.show() 
fig.savefig("{}/xy_view.png".format(figure_folder), dpi=600, bbox_inches='tight')  # Set DPI to 300
fig.savefig("{}/xy_view.pdf".format(figure_folder))  # Set DPI to 300
fig.savefig("{}/xy_view.svg".format(figure_folder))
plt.close(fig)




########################################################
# 2. Plot simulation images and style translated images
########################################################
showScanRegion = False
showImageRegion = True
showLattice = False

# Atom positions and types
positions = supercell.get_positions()
numbers = supercell.get_atomic_numbers()

zlim = [-2.9, 0.5]
zref = positions[:, 2].max()
print(zref)

# Obtain the range
xImgMin, xImgMax = xy_center[0] - ss[0]/2, xy_center[0] + ss[0]/2 
yImgMin, yImgMax = xy_center[1] - ss[1]/2, xy_center[1] + ss[1]/2 
zMin, zMax = zref + zlim[0], zref + zlim[1]

print(xImgMin, xImgMax)
print(yImgMin, yImgMax)



# Select O (8) and H (1) atoms
is_water_atom = (numbers == 1) | (numbers == 8)

# Apply x-y range filter
in_xy_range = ((positions[:, 0] >= xImgMin) & (positions[:, 0] <= xImgMax) &
               (positions[:, 1] >= yImgMin) & (positions[:, 1] <= yImgMax))

in_z_range = (positions[:, 2] >= zMin) & (positions[:, 2] <= zMax)

# Final selection: O or H atoms AND inside range
selection_mask = is_water_atom & in_xy_range & in_z_range
Y = supercell[selection_mask]


rows, cols = 3, 2
fig = plt.figure(figsize=(2*cols, 2*rows))
gs = fig.add_gridspec(rows, cols)
# Step 1: Collect all images first to compute global vmin and vmax, then I can use the same vmin and vmax for all images
all_images = []
for j in range(cols):
    for i in range(rows):
        if j == 0:
            imagePath = '{}/PPAFM/{:.2f}.png'.format(exampleInput, i * 0.1 * dz) # PPAFM
        else:
            imagePath = '{}/FakeAFM/{:.2f}.png'.format(exampleInput, i * 0.1 * dz) # FakeAFM
        image = iio.imread(imagePath).astype(np.float32)  
        rotated_image = np.rot90(image, k=3)
        all_images.append(rotated_image)
all_images_np = np.array(all_images)
vmin = all_images_np.min()
vmax = all_images_np.max()

# Calculate the error maps
error_maps = []
for i in range(rows):
    imagePPAFM = '{}/PPAFM/{:.2f}.png'.format(exampleInput, i * 0.1 * dz) # PPAFM
    imageFakeAFM = '{}/FakeAFM/{:.2f}.png'.format(exampleInput, i * 0.1 * dz) # FakeAFM
    imagePPAFM = iio.imread(imagePPAFM).astype(np.float32)  
    imageFakeAFM = iio.imread(imageFakeAFM).astype(np.float32)
    print('imagePPAFM:', imagePPAFM)
    print('imageFakeAFM:', imageFakeAFM)
    error_map = imageFakeAFM - imagePPAFM
    print('error_map:', error_map)
    error_maps.append(error_map)
all_error_maps = np.array(error_maps)
emin = all_error_maps.min()
emax = all_error_maps.max()
print('emin:', emin)
print('emax:', emax)

subLabels = [fr"$u=H_{{\mathcal{{M}}}}(m)$", fr"$\tilde{{v}}=G_{{\mathcal{{U}}}}(u)$", fr"$x-x^\prime$"]

# Store axes for each column
axes_col = [[], [], []]
ims_col = [None, None, None]

offset_text = 0.05
offset_text_y = 0.05

# Draw a gray vertical line at horizontal = width/2
height, width = rotated_image.shape
x_center = width / 2
# Markers at 1/6 and 5/6 of the height on the line
y1 = height / 5
y2 = 4 * height / 5
for j in range(cols): # Columns
    for i in range(rows): # Rows
        ax = fig.add_subplot(gs[i, j])
        ax.set_aspect('equal')
        ax.tick_params(axis='both', direction='in', labelright=False)
        im = None
        if j == 0:
            imagePath = '{}/PPAFM/{:.2f}.png'.format(exampleInput, i * 0.1 * dz) # PPAFM
            image = iio.imread(imagePath).astype(np.float32)  # or use imageio.imread for older versions
            rotated_image = np.rot90(image, k=3)  # 90 degrees counter-clockwise
            im = ax.imshow(rotated_image, cmap='inferno', vmin=vmin, vmax=vmax)
        elif j == 1:
            imagePath = '{}/FakeAFM/{:.2f}.png'.format(exampleInput, i * 0.1 * dz) # FakeAFM
            image = iio.imread(imagePath).astype(np.float32)  # or use imageio.imread for older versions
            rotated_image = np.rot90(image, k=3)  # 90 degrees counter-clockwise
            im = ax.imshow(rotated_image, cmap='inferno', vmin=vmin, vmax=vmax)
        elif j == 2:
            image = all_error_maps[i]  # Use the error map
            rotated_image = np.rot90(image, k=3)  # 90 degrees counter-clockwise
            im = ax.imshow(rotated_image, cmap='BrBG', vmin=emin, vmax=emax)
        
        if i == 1:
            ax.axvline(x=x_center, color=simcolor if j==0 else expcolor , linestyle='dashed' if j==0 else 'solid', zorder=10)
            ax.plot(x_center, y1, marker='.', color='grey', markersize=10, zorder=11)
            ax.plot(x_center, y2, marker='.', color='grey', markersize=10, zorder=11)
            ax.text(x_center + 0.04 * width, y1, r"P$_0$", va='center', ha='left', fontsize=14, color='grey')
            ax.text(x_center + 0.04 * width, y2, r"P$_1$", va='center', ha='left', fontsize=14, color='grey')
        ims_col[j] = im 
        ax.axis('off')  # optionally hide axis ticks and labels
        if i == 0:
            pos = ax.get_position()
            # Create a new axes above the subplot, same width, small height
            label_ax = fig.add_axes([pos.x0, pos.y1 + 0.01, pos.width, 0.04])
            label_ax.axis('off')
            x  = 0.0 if j==0 else -0.1
            label_ax.text(x+0.55, 0.0, subLabels[j], ha='center', va='bottom', transform=label_ax.transAxes)
            #ax.text(offset_text, 1 - offset_text_y, subLabels[j], transform=ax.transAxes, va='top', ha='left')
        if j == 0:
            ax.text(offset_text, offset_text_y-0.05, fr"$z_{i} = {refZ0 - (i + 1) * dz :.2f}$ Å",  transform=ax.transAxes, va='bottom', ha='left', color='lightgrey' if i == 2 else 'k')

# Add colorbars above each column
pos = ax.get_position()
cax = fig.add_axes([pos.x0 - 0.22, pos.y0 - 0.02, pos.width, 0.015])  # 0.06 is the vertical gap, adjust as needed
cbar = fig.colorbar(ims_col[0], cax=cax, orientation='horizontal')
cbar.set_label('Pixel intensity')

fig.subplots_adjust(hspace=0.01, wspace=0.01)
#plt.tight_layout()
if show: plt.show() 
fig.savefig("{}/xy_view_data.png".format(figure_folder), dpi=600)  # Set DPI to 300
fig.savefig("{}/xy_view_data.pdf".format(figure_folder))  # Set DPI to 300
fig.savefig("{}/xy_view_data.svg".format(figure_folder))
plt.close(fig)


fig = plt.figure(figsize=(4, 2))
gs = fig.add_gridspec(1, 3)

shape = all_images[0].shape
print('shape:', shape)
xps = [all_images[i][:, int(shape[1]/2)] for i in range(3)]
xs  = [all_images[i+rows][:, int(shape[1]/2)] for i in range(3)]

for i in range(3):
    if i == 0:
        ax1 = fig.add_subplot(gs[0, i])
    else:
        ax1 = fig.add_subplot(gs[0, i], sharey=ax1)
    yvals = list(range(shape[1]))
    diff = xps[i] - xs[i]
    ax1.invert_yaxis() # >>
    ax1.set_ylim([shape[1], 0]) # << Need to be adjusted
    ax1.text(0.45, 0.9, rf'$z_{{{i}}}$', transform=ax1.transAxes)
    ax1.plot(xps[i], yvals, color=simcolor, ls='dashed', label=rf'$u$')
    ax1.plot(xs[i], yvals, color=expcolor, label=rf'$\tilde{{v}}$')
    ax1.fill_betweenx(yvals, xps[i], xs[i], where=diff > 0, color=bg07color, alpha=0.3, label=rf'$\Delta > 0$')
    ax1.fill_betweenx(yvals, xps[i], xs[i], where=diff < 0, color=bv17color, alpha=0.3, label=rf'$\Delta < 0$', hatch='..', edgecolor='k')
    # ax1.axhline(y=y1, color='gray', linestyle='dashed', lw=0.5, zorder=10, alpha=0.5)
    # ax1.axhline(y=y2, color='gray', linestyle='dashed', lw=0.5, zorder=10, alpha=0.5)
    ax1.axhline(y=y1, color='gray', linestyle='dashed', lw=0.5, zorder=10, alpha=0.5)
    if i == 0: ax1.text(vmin + 0.05 * (vmax - vmin), y1, r"P$_0$", va='bottom', ha='left', zorder=11, color='gray')
    ax1.axhline(y=y2, color='gray', linestyle='dashed', lw=0.5, zorder=10, alpha=0.5)
    if i == 0: ax1.text(vmin + 0.05 * (vmax - vmin), y2, r"P$_1$", va='bottom', ha='left', zorder=11, color='gray')
    ax1.set_xlim([vmin, vmax])
    if i != 0: 
        ax1.set_yticklabels([])
    else:
        ax1.tick_params(axis='y', labelleft=True)
    if i == 1: ax1.set_xlabel('Pixel intensity')
    if i == 2: ax1.legend(loc='right', bbox_to_anchor=(1.13, 0.46), handlelength=1, handletextpad=0.1, labelspacing=0.3)
    if i == 0:
        ax1.set_ylabel(r'$y$ (Å)')

#fig.subplots_adjust(hspace=0.0, wspace=0.05, left=0.05, bottom=0.15, right=0.99, top=0.95)
fig.subplots_adjust(hspace=0.0, wspace=0.05, left=0.05, bottom=0.15, right=0.99, top=0.95)
if show: plt.show() 
fig.savefig("{}/style_difference.png".format(figure_folder), dpi=600, bbox_inches='tight')  # Set DPI to 300
fig.savefig("{}/style_difference.pdf".format(figure_folder), bbox_inches='tight')  # Set DPI to 300
fig.savefig("{}/style_difference.svg".format(figure_folder), bbox_inches='tight')
plt.close(fig)


########################################################################
# 3. Get the z distribution for all the structures in the Label folder
########################################################################
showAll = True
if showAll:
    samples = read_samples_from_folder('../../data/structures/simulations/Label')
    z = []
    for structure in samples:
        atoms = read_xyz_with_atomic_numbers(structure)
        z_positions_Au = [atom.position[2] for atom in atoms if atom.symbol == 'Au']
        if len(z_positions_Au) > 0:
            mean_z_Au = sum(z_positions_Au) / len(z_positions_Au)
        else:
            mean_z_Au = 0  # or handle this case appropriately
            print('No Au atoms in {}'.format(structure))
        mean_z_Au = sum(z_positions_Au) / len(z_positions_Au)
        z_positions_O = [atom.position[2] - mean_z_Au for atom in atoms if atom.symbol == 'O']
        z.extend(z_positions_O)
# Plot the atoms of demonstration configuration  of cross section in the yz plane
# And the distribution of z positions of O atoms
# Use the mean z position of Au atoms as the reference plane z=0
atoms = read_xyz_with_atomic_numbers(demoStructure)
z_positions_Au = [atom.position[2] for atom in atoms if atom.symbol == 'Au']
mean_z_Au = sum(z_positions_Au) / len(z_positions_Au)
for atom in atoms:
    atom.position[2] -= mean_z_Au
z_positions_O = [atom.position[2] for atom in atoms if atom.symbol == 'O']

# Rotate view
atoms.rotate(270, 'x',  rotate_cell=True)
atoms.rotate(270, 'y', rotate_cell=True)

fig = plt.figure(figsize=(6, 2))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
ymin, ymax = -2, 12 

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_aspect('equal')

# Get the minimum and maximum x and y positions of the atoms
x_positions = [atom.position[0] for atom in atoms]
y_positions = [atom.position[1] for atom in atoms]
xmin, xmax = min(x_positions), max(x_positions)
ymin, ymax = min(y_positions), max(y_positions)

offset = 1
xmin, xmax = xmin - 3*offset, xmax + 3*offset
ymin, ymax = ymin - 2*offset, ymax + 5*offset

#xticks = np.arange(xmin, xmax, 10)
#yticks = np.arange(ymin, ymax, 5)
#ax1.set_xticks(xticks)
#ax1.set_xticklabels([f'{x/10:.0f}' for x in xticks])  # Convert Å to nm
#ax1.set_yticks(yticks)
#ax1.set_yticklabels([f'{y/10:.0f}' for y in yticks])  # Convert Å to nm
ax1.set_xlim([xmin, xmax])
ax1.set_ylim([ymin, ymax])
ax1.tick_params(axis='both', direction='in', labelright=False)
ax1.set_xlabel(r'$y$ (Å)')
ax1.set_ylabel(r'$z$ (Å)')
# ax1.text(offset_text, 1 - offset_text, "b", transform=ax1.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
draw_3d_axis_indicator(ax1, anchor=(0.89, 0.65), length=40, style='x-out')
# Add the atoms to the plot as circles.
# Reorder the atoms based on the z position so that the atoms at the back are plotted first
atoms = sorted(atoms, key=lambda atom: atom.position[2])
for atom in atoms:
    color = jmol_colors[atom.number]
    radius = radii[atom.number]
    circle = Circle((atom.x, atom.y), radius, facecolor=color,
                        edgecolor='k', linewidth=0.5)
    ax1.add_patch(circle)

# Plot the distribution of z positions
ax2 = fig.add_subplot(gs[0, 1])
#ax2.hist(z_positions_O, orientation='horizontal', bins=30, density=True, color=jmol_colors[8], alpha=1)
sns.kdeplot(y=z_positions_O, fill=False, bw_adjust=1, ax=ax2, color=jmol_colors[8], label='O')
if showAll:
    sns.kdeplot(y=z, fill=False, bw_adjust=1, ax=ax2, color=jmol_colors[8], linestyle='dotted', label='O (all)')
# ax2.axhline(0, color=jmol_colors[79], linestyle='-', label='Au') # Au surface
# ax2.axhline(5.89, color='k', linestyle='--', lw=0.5) # 
# ax2.axhline(4.85, color='k', linestyle='--', lw=0.5) # 
# ax2.axhline(3.32, color='k', linestyle='--', lw=0.5) # 

ax2.plot([0, 0.13], [5.89, 5.89], color='k', linestyle='--', lw=0.5)
ax2.plot([0, 0.05], [4.9, 4.9], color='k', linestyle='--', lw=0.5)
ax2.plot([0, 0.58], [3.32, 3.32], color='k', linestyle='--', lw=0.5)


ax2.set_xlabel('')
ax2.legend(loc='upper right', handlelength=0.7, labelspacing=0.2, bbox_to_anchor=(1, 1))
# Hide the y-axis labels on the second plot
ax2.tick_params(axis='y', labelleft=True)
ax2.set_xlabel(r'Density $\rho(z)$')
ax2.set_xlim([0, 1.2])
ax2.set_ylim([ymin, ymax])
ax2.tick_params(axis='both', direction='in', labelleft=False)
# ax2.text(offset_text, 1 - offset_text, "c", transform=ax2.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
fig.subplots_adjust(hspace=0, wspace=0, left=0.05, bottom=0.15, right=0.99, top=0.95)
if show: plt.show() 
fig.savefig("{}/z_distribution.png".format(figure_folder), dpi=600, bbox_inches='tight')  # Set DPI to 300
fig.savefig("{}/z_distribution.pdf".format(figure_folder), bbox_inches='tight')  # Set DPI to 300
fig.savefig("{}/z_distribution.svg".format(figure_folder), bbox_inches='tight')
plt.close(fig)

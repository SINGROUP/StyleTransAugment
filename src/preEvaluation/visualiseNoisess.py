#!/usr/bin/env python

# %% 
import matplotlib.pyplot as plt
import os

plt.rcParams['font.size']=14
plt.rcParams['font.family']='Arial'
plt.rcParams['pdf.fonttype']=42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = True # Render text with LaTeX


simulated_sample = '902.12.png'
noise_types = ['noise_previous', 'cutout_previous', 'gradient_previous', 'combined_previous', 'saltpepper']
tiltles = ['Gaussian', 'Cutout', 'GradientBG', 'Comb.', r'Salt \& Pepper']
input_folder = '../../data/preEvaluate/'
output_folder = '../../results/noiseTypes/'
os.makedirs(output_folder, exist_ok=True)

# %%
# Create a figure with subplots with 1 row and 6 columns
fig, axes = plt.subplots(1, len(noise_types) + 1, figsize=(12, 2.5))
simulated_image_path = os.path.join(input_folder, 'realA', simulated_sample)
# Load the simulated image and show it in the first subplot
simulated_image = plt.imread(simulated_image_path)
axes[0].imshow(simulated_image, cmap='inferno')
axes[0].axis('off')
# Set the title for the first subplot
axes[0].set_title('Simulated Image')

for i, noise_type in enumerate(noise_types):
    # Noised image path
    noised_image_path = os.path.join(input_folder, 'noisedA_{}_{}'.format(noise_type, 3 if noise_type=='gradient_previous' else 4), simulated_sample)
    # Load the noised image
    noised_image = plt.imread(noised_image_path)
    # Show the noised image in the corresponding subplot
    axes[i + 1].imshow(noised_image, cmap='inferno')
    # No axis
    axes[i + 1].axis('off')
    # Set the title for the subplot
    axes[i + 1].set_title(tiltles[i])

plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'noise_types_.png'), dpi=600)
plt.savefig(os.path.join(output_folder, 'noise_types_.pdf'), dpi=600)
plt.savefig(os.path.join(output_folder, 'noise_types_.svg'), dpi=600)

plt.show()

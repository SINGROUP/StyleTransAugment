#!/usr/bin/env python

# %% Libraries
import pandas as pd
import matplotlib.pyplot as plt
from drawarrow import ax_arrow
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
import os, json 
# %% Functions
def plot_fidelity(x, y1, y2, color1, color2, axes, left=True, right=True, top=True, bottom=True, title=None, xlabel=None, rotate=True):
    axes.fill_betweenx(x, -y1, 0, color=color1, alpha=0.5)
    axes.fill_betweenx(x, y2, 0, color=color2, alpha=0.5)
    axes.set_xlabel(xlabel) if xlabel else None
    axes.set_ylabel(rf"Authenticity $s$") if left else None
    axes.tick_params(which='minor', length=3, width=1, direction='in', top=True)
    axes.tick_params(which='major', length=5, width=1.2, direction='in', top=True)
    axes.set_xlim(-6, 6)
    axes.set_xticks([-5, 0, 5])
    axes.set_xticklabels([5, 0, 5])
    axes.set_xticks(np.arange(-5, 6, 1), minor=True)
    # Set no y-ticks if left is False
    if not left:
        axes.yaxis.set_ticks_position('none')
        axes.yaxis.set_ticklabels([])

    axes.spines['right'].set_visible(right)
    axes.spines['top'].set_visible(top)
    axes.spines['left'].set_visible(left)
    axes.spines['bottom'].set_visible(bottom)
    
    axes.text(0.70, 0.70, rf'$\mathcal{{V}}$', transform=axes.transAxes, 
                  va='bottom', ha='center', color=color2)

    # Show rotated text at the center if title is provided
    if title:
        axes.text(0.25, 0.20, title, transform=axes.transAxes, rotation=90 if rotate else 0,
                  va='bottom', ha='center', color=color1)

# %% Load data
if __name__ == "__main__":
    #noiseTypes = ['gaussian', 'saltpepper', 'blur', 'gamma', 'speckle', 'drift']
    noiseTypes = ['noise_previous', 'cutout_previous', 'gradient_previous', 'combined_previous', 'saltpepper']
    noiseULabels = [
                    rf"$\overline{{\mathcal{{V}}}}_\text{{1: Gaussian}}$",
                    rf"$\overline{{\mathcal{{V}}}}_\text{{2: Cutout}}$",
                    rf"$\overline{{\mathcal{{V}}}}_\text{{3: Gradient BG}}$",
                    rf"$\overline{{\mathcal{{V}}}}_\text{{4: Combination}}$",
                    rf"$\overline{{\mathcal{{V}}}}_\text{{5: Salt \& Pepper}}$",
                    ]
    df = pd.read_csv("../../processed_data/fidelityTest/fidelity_results.csv")
    df_u = df[df["Class"].str.contains("realA")]
    df_v = df[df["Class"].str.contains("realB")]

    # Use the order in noiseTypes for all noise types
    noise_types = ['noisedA_' + noise_type for noise_type in noiseTypes]
    print("Noise types:", noise_types)
    noise_dfs = [
        df[df["Class"].str.contains(noise_type)]["Output Value"].dropna()
        for noise_type in noise_types
    ]

    # Automatically find all v_t types and get their Output Value series
    vt_mask = df["Class"].str.contains(r"fakeB_Water-bilayerCGPPAFM2Exp_CoAllL\d+L[\d\.]+Elatest")
    vt_types = df.loc[vt_mask, "Class"].str.extract(r"(fakeB_Water-bilayerCGPPAFM2Exp_CoAllL\d+L[\d\.]+Elatest)")[0].unique()
    vt_dfs = [
        df[df["Class"].str.contains(vt_type)]["Output Value"].dropna()
        for vt_type in vt_types
    ]
    
    show = True

    # %
    simcolor = '#ed9d2c'
    dftcolor = '#2ca3cf'
    expcolor = '#de461c'
    bg07color = '#479FB1'
    bv17color = '#6E7CBC'

    #colors = ['#E54434', '#447F37', '#5986C0']
    #colors = sns.blend_palette([dftcolor, simcolor], n_colors=7)
    colors = sns.color_palette("Set1",    n_colors=7)
    outputFolder = '../../results/fidelity'
    os.makedirs(outputFolder, exist_ok=True)

    distanceFile = '../../results/distance_analysis/all_distances.json'
    with open(distanceFile) as f:
        distances = json.load(f)

    plt.rcParams['font.size']=14
    #plt.rcParams['font.family']='Arial'
    plt.rcParams['pdf.fonttype']=42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['text.usetex'] = True # Render text with LaTeX
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # %% Plot
    # fig, axes = plt.subplots(1, 9, figsize=(12, 4))
    n_noise = len(noiseTypes)                # = 6 in your script
    n_small = 2 + n_noise                    # the first two plots + one per noise type
    n_axes   = n_small + 1                   # +1 for your “big” last plot
    widths   = [1]*n_small + [5]             # make the last one twice as wide

    fig, axes = plt.subplots(
        1, n_axes,
        figsize=(10, 2.5),
        gridspec_kw={'width_ratios': widths, 'wspace': 0.3}
    )
    axes = axes.flatten()
    x_grid = np.linspace(0, 1, 200)
    axes = axes.flatten()  # Flatten the 2x2 array of axes for easier indexing

    # 1. u and v
    x = x_grid
    kde_u = gaussian_kde(df_u["Output Value"].dropna())(x_grid)
    kde_v = gaussian_kde(df_v["Output Value"].dropna())(x_grid)
    y1, y2 = kde_u, kde_v
    color1 = dftcolor 
    color2 = expcolor
    plot_fidelity(x, y1, y2, color1, color2, axes[0], right=True, title=rf"$\mathcal{{U}}$", rotate=False)


    # %%
    # 2. vt (mean and SE)
    kdes_vt = [gaussian_kde(vt)(x_grid) for vt in vt_dfs]
    kde_mean_vt = np.mean(kdes_vt, axis=0)
    error = np.std(kdes_vt, axis=0) 
    color1 = simcolor
    plot_fidelity(x, kde_mean_vt, y2,  color1, color2, axes[1], left=True, right=True, title=rf"$\widetilde{{\mathcal{{V}}}}$", rotate=False)

    # 3. Plot each noise type independently
    for i, (noise_type, noise_data) in enumerate(zip(noise_types, noise_dfs)):
        kde_noise = gaussian_kde(noise_data)(x_grid)
        xlabel = rf'Density $\rho(s)$' if i == 1 else None
        left = True if i == 0 else False
        right = True if i == n_noise - 1 else False
        plot_fidelity(
            x, kde_noise, y2, color1, color2, axes[i+2],
            left=left, right=right, title=noiseULabels[i], xlabel=xlabel
        )


    # 4. Placeholder for additional plot
    print("Distances:", distances)
    marker_area = 180
    marker_size = 9
    # V
    axes[-1].scatter(0, 0, marker='*', color='none', edgecolors=expcolor, s=marker_area, label=rf'$\mathcal{{V}}$', linewidths=1)  # V
    # U
    x_mean_U, x_std = distances['FID(U, V)']['mean'], distances['FID(U, V)']['stderr']
    y_mean_U, y_std = distances['W(U, V)']['mean'], distances['W(U, V)']['stderr']
    axes[-1].scatter(x_mean_U, y_mean_U, color='none', edgecolors=dftcolor, s=marker_area, label=rf'$\mathcal{{U}}$', linewidths=1)
    # Draw a arrow from U to (0, 0)
    scale = 0.5
    ax_arrow(
        head_position=(0, 0),
        tail_position=(x_mean_U, y_mean_U),
        width=1,
        head_width=2*scale,
        head_length=4*scale,
        radius=0,
        color="grey",
        fill_head=True,
        mutation_scale=2,
        ls='--',
        ax=axes[-1]
    )
    # Add text along the arrow: "Ideal translation"
    mid_x = (x_mean_U) / 4
    mid_y = (y_mean_U) / 4
    axes[-1].text(
        mid_x, mid_y, "Ideal translation",
        color="grey", fontsize=10, rotation=47,
        ha='left', va='bottom', rotation_mode='anchor'
    )


    # Tilde V
    x_mean_tV, x_std = distances['FID(V_tilde, V)']['mean'], distances['FID(V_tilde, V)']['stderr']
    y_mean_tV, y_std = distances['W(V_tilde, V)']['mean'], distances['W(V_tilde, V)']['stderr']
    axes[-1].errorbar(
        x_mean_tV, y_mean_tV, xerr=x_std, yerr=y_std, fmt='o', 
        markerfacecolor='none', markeredgecolor=simcolor, color=simcolor, 
        markersize=marker_size,
        label=rf'$\widetilde{{\mathcal{{V}}}}$'
    )
    # Draw a arrow from tilde V to (0, 0)
    ax_arrow(
        head_position=(x_mean_tV, y_mean_tV),
        tail_position=(x_mean_U, y_mean_U),
        width=2,
        head_width=2*scale,
        head_length=4*scale,
        radius=0,
        color="grey",
        fill_head=True,
        mutation_scale=2,
        ax=axes[-1])
    # Add text along the arrow: r"Translation $G_\mathcal{U}$"
    mid_x_tV = (x_mean_U + x_mean_tV) / 2.4
    mid_y_tV = (y_mean_U + y_mean_tV) / 6
    axes[-1].text(
        mid_x_tV, mid_y_tV, "Style translation",
        color="grey", fontsize=10, rotation=70,
        ha='left', va='bottom', rotation_mode='anchor'
    )


    # Noise types
    markers = ['s', 'D', '^', 'v', 'P', 'X', '*', '<', '>', '1', '2', '3', '4']  # Add more if needed
    x_, y_ = None, None
    for i, (noise_type, noise_data) in enumerate(zip(noise_types, noise_dfs)):
    #for i, (noise_type, noise_data) in enumerate(zip(noise_types[-3:], noise_dfs[-3:]), start=len(noise_types)-3):
        x_mean, x_std = distances[f'FID({noise_type}, V)']['mean'], distances[f'FID({noise_type}, V)']['stderr']
        y_mean, y_std = distances[f'W({noise_type}, V)']['mean'], distances[f'W({noise_type}, V)']['stderr']
        current_marker_size = marker_size - 1 if noise_type == 'noisedA_cutout_previous' else marker_size
        if noise_type == 'noisedA_saltpepper':
            x_, y_ = x_mean, y_mean
            marker = 'p'
        else:
            marker = markers[i % len(markers)]
        axes[-1].errorbar(
            x_mean, y_mean, xerr=x_std, yerr=y_std, fmt=marker, 
            markerfacecolor='none', markeredgecolor=simcolor, color=simcolor,
            markersize=current_marker_size,
            label=noiseULabels[i])
    # Draw a arrow from tilde U to (x_, y_)
    ax_arrow(
        head_position=(x_, y_),
        tail_position=(x_mean_U, y_mean_U),
        width=1,
        head_width=2*scale,
        head_length=4*scale,
        radius=0,
        color="grey",
        fill_head=True,
        mutation_scale=2,
        ax=axes[-1])
    
    # Fill the rectangle between U and V_tilde
    axes[-1].fill_between(
        [0, x_mean_U], 
        [0, 0], 
        [y_mean_U, y_mean_U], 
        color=dftcolor, 
        alpha=0.2, 
        linewidth=0,
        edgecolor='none'
    ) 
    axes[-1].fill_between(
        [0, x_mean_tV],  # FID range
        [0, 0],   # bottom edge (flat at U)
        [y_mean_tV, y_mean_tV], # top edge (flat at tildeV)
        color=simcolor,
        alpha=0.5,
        linewidth=0,
        edgecolor='none'
    )


    new_labels = [rf'$\mathcal{{U}}$', rf"$\widetilde{{\mathcal{{V}}}}$", 'Gaussian', 'Speckle', 'Drift', 'Blur', 'Saltpepper', 'Gamma'] 
    axes[-1].set_xlabel(r"$\mathrm{FID}(\cdot ,\, \mathcal{V})$")
    axes[-1].set_ylabel(r"$\mathrm{WD}(\cdot ,\, \mathcal{V})$")
    axes[-1].legend(
        loc='lower right',
        fontsize=10,
        frameon=False,
        handlelength=0.8,
        borderpad=0.3,
        labelspacing=0.2,
        handletextpad=0.4,
        borderaxespad=0.2,
        handler_map={
            ErrorbarContainer: HandlerErrorbar(xerr_size=0.2, yerr_size=0.2)
        }
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.1, left=0.1, bottom=0.2, right=0.95, top=0.99)
    plt.savefig(f"{outputFolder}/distance.png", dpi=600)
    plt.savefig(f"{outputFolder}/distance.pdf")
    plt.savefig(f"{outputFolder}/distance.svg")
    if show: plt.show()
# %%

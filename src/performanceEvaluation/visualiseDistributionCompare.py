#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, inferno
import os, json
from scipy.stats import gaussian_kde

plt.rcParams['font.size']=14
plt.rcParams['pdf.fonttype']=42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = True # Render text with LaTeX

simcolor = '#ed9d2c'
expcolor = '#de461c'
dftcolor = '#2ca3cf'
bg07color = '#479FB1'
bv17color = '#6E7CBC'



baseOut = '../../results/compare_distributions/'
os.makedirs(baseOut, exist_ok=True)
show = True

def plot_multi_2d_scatter(data_dict, distance_map, convert_label_func,
                       ncols=6, nrows=6, xlabel=r'$x$', ylabel=r'$y$',
                       xlim=None, ylim=None,
                       baseOut='../../results/compare_distributions/',
                       property_name="2D_property", show=False, sample_counts=None,
                       sample_count_style=None, wd_text_style=None):
    """
    Improved version that adds model labels, sample counts, and WD distances to each subplot.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols * 1.8, nrows * 1.7),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    keys = list(data_dict.keys())

    # After creating axes and keys
    n_axes = len(axes)
    if len(keys) > n_axes:
        # Keep only the first (n_axes - 1) keys and the last key (max distance)
        keys = keys[:n_axes-1] + [keys[-1]]
    elif len(keys) < n_axes:
        # Optionally, trim axes as well if you have fewer keys than axes
        axes = axes[:len(keys)]

    #num = len(keys)

    # Normalize WD distances for color mapping
    wds = [distance_map.get(k, 0) for k in keys]
    valid_wds = [wd for wd in wds if wd >= 0]
    norm = Normalize(vmin=min(valid_wds), vmax=max(valid_wds))
    cmap = inferno
    smap = ScalarMappable(norm=norm, cmap=cmap)

    for i, key in enumerate(keys):
        ax = axes[i]
        data = data_dict[key]
        if data.shape[1] != 2:
            ax.axis('off')
            continue
        x, y = data[:, 0], data[:, 1]
        wd = distance_map.get(key, None)
        color = smap.to_rgba(wd if wd is not None and wd >= 0 else 0)

        ax.scatter(x, y, s=10, alpha=0.5, color=color, edgecolor='k', linewidth=0.1)

        # Plot label at top left
        label = convert_label_func(key)
        if 'dagger' in label:
            ycolor = dftcolor
        elif 'Pure' in label:
            ycolor = simcolor
        else:
            if 'U' in label:
                ycolor = 'k'
            else:
                ycolor = expcolor
        ax.text(
            0.02, 0.25, label, ha='left', va='top', transform=ax.transAxes,
            fontsize=10,
            #color='black' if 'tilde' not in label else dftcolor,
            color= ycolor,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.05')
        )
        # Plot sample count below the label
        if sample_counts is not None and key in sample_counts:
            count_style = {
                'x': 0.02,
                'y': 0.10,
                'fontsize': 9,
                'ha': 'left',
                'va': 'top',
                'bbox': dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.03')
            }
            if sample_count_style is not None:
                count_style.update(sample_count_style)
            ax.text(
            count_style['x'], count_style['y'], fr"$n = {sample_counts[key]:.0f}$",
            ha=count_style['ha'], va=count_style['va'], transform=ax.transAxes,
            fontsize=count_style['fontsize'],
            color='black',
            bbox=count_style['bbox']
            )
        # Plot WD distance below the sample count (if available and > 0)
        if wd is not None and wd > 0:
            wd_style = {
                'x': 0.02,
                'y_with_count': 0.02,
                'y_without_count': 0.10,
                'fontsize': 8,
                'ha': 'left',
                'va': 'top',
                'bbox': dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.03')
            }
            if wd_text_style is not None:
                wd_style.update(wd_text_style)
            y_pos = wd_style['y_with_count'] if sample_counts is not None and key in sample_counts else wd_style['y_without_count']
            ax.text(
            wd_style['x'], y_pos, fr"$\mathrm{{WD}}(\cdot, \mathcal{{U}}_\mathrm{{Top}}) = {wd:.5f}$",
            ha=wd_style['ha'], va=wd_style['va'], transform=ax.transAxes,
            fontsize=wd_style['fontsize'],
            color='black',
            bbox=wd_style['bbox']
            )
        # Add "Min" or "Max" at top right corner if needed (bold font)
        if i == 2:  # Excluding 'All' and 'Top', this is the third plot
            ax.text(
            0.7, 0.2, rf"$\mathrm{{Min}}$", ha='left', va='top', transform=ax.transAxes,
            fontsize=14, color='green',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.02')
            )
        elif i == len(keys) - 1:
            ax.text(
            0.7, 0.2, rf"$\mathrm{{Max}}$", ha='left', va='top', transform=ax.transAxes,
            fontsize=14, color='red',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.02')
            )

        # Axes styling
        ax.tick_params(axis='both', which='both', direction='in', labelsize=10)
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    #fig.supxlabel(xlabel, fontsize=12)
    #fig.supylabel(ylabel, fontsize=12)
    fig.supxlabel(xlabel, x=0.5, ha='left')
    fig.supylabel(ylabel, x=0.01, y=0.5, va='top')

    plt.tight_layout()
    fig.subplots_adjust(left=0.06, right=0.98, top=0.99, bottom=0.05, wspace=0.01, hspace=0.01)

    outbase = f"{baseOut}/compare2D_{property_name}"
    plt.savefig(outbase + ".png", dpi=600)
    plt.savefig(outbase + ".pdf")
    plt.savefig(outbase + ".svg")
    if show: plt.show()


def plot_multi_kdes(
    datasets,
    width=6,
    height_per_subplot=6.0,
    fill=True,
    bw=0.025,
    overlap_space=0.1,
    colormap=plt.cm.viridis,
    xlabel='Value',
    xlim=None,
    ylabels=None,
    annotations=None,
    left_margin=0.02,
    right_margin=0.26,
    baseOut=baseOut, 
    property_name=None, 
    show=False,
    sample_counts=None,
    sample_count_style=None
):
    """
    Plot multiple KDEs in vertically overlapping subplots with colormap fill and black edges.

    Parameters:
    - datasets: list of 1D numpy arrays
    - height_per_subplot: height of each subplot
    - fill: whether to fill the KDE areas
    - overlap_space: fraction overlap between subplots (0.0 to 1.0)
    - colormap: matplotlib colormap (e.g., plt.cm.viridis, plt.cm.inferno)
    - left_margin: fraction of figure width for left margin (default 0.15)
    - right_margin: fraction of figure width for right margin (default 0.05)
    """
    num_plots = len(datasets)

    # Compute KDEs
    kdes = [gaussian_kde(data, bw_method=bw) for data in datasets]
    x_min = min(data.min() for data in datasets) - 2
    x_max = max(data.max() for data in datasets) + 2
    x_vals = np.linspace(x_min, x_max, 500)
    kde_vals_list = [kde(x_vals) for kde in kdes]
    y_max = max(kde.max() for kde in kde_vals_list)

    # Generate color list from colormap
    step = 0.03
    start = 0.0
    cmap_positions = [start + i * step for i in range(num_plots)]
    cmap_positions = [min(1.0, p) for p in cmap_positions]
    cmap_colors = [colormap(p) for p in cmap_positions]

    # Layout computation
    gap = height_per_subplot * (1 - overlap_space)
    bottom_margin = 4
    top_margin = 0.01
    total_height = (num_plots - 1) * gap + height_per_subplot + bottom_margin + top_margin
    fig = plt.figure(figsize=(width, total_height * 5 / num_plots))

    axes = []
    plot_width = 1.0 - left_margin - right_margin
    for i in range(num_plots):
        bottom = (bottom_margin + i * gap) / total_height
        height = height_per_subplot / total_height
        ax = fig.add_axes([left_margin, bottom, plot_width, height])
        axes.append(ax)

    for i, (ax, kde_vals, color, label) in enumerate(zip(reversed(axes), kde_vals_list, cmap_colors, ylabels or [])):
        if fill:
            ax.fill_between(x_vals, kde_vals, color=color, alpha=0.7)
        ax.plot(x_vals, kde_vals, color='black', linewidth=1.0)
        ax.set_ylim(0, y_max * 1.1)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_facecolor((1, 1, 1, 0))
        ax.set_yticks([])
        if ylabels is not None:
            # Determine label color based on whether it contains 'dagger'
            # I konw it's ugly. Forgive me. 
            if 'dagger' in label:
                ycolor = dftcolor
            elif 'Pure' in label:
                ycolor = simcolor
            else:
                if 'U' in label:
                    ycolor = 'k'
                else:
                    ycolor = expcolor
            ax.text(
                #xlim[0] - 0.02 * (xlim[1] - xlim[0]),  # just outside the left of plot
                xlim[0],
                ax.get_ylim()[1] * 0.1,               # vertical center
                label,
                #color=expcolor if 'tilde' not in label else dftcolor,
                color=ycolor,
                fontsize=9,
                ha='left',
                va='center',
                transform=ax.transData
            )
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        for spine in ['left', 'right', 'top', 'bottom']:
            ax.spines[spine].set_visible(False)

        # --- Add sample count annotation if provided ---
        if sample_counts and i < len(sample_counts) and sample_counts[i] is not None:
            count_style = {
                'x': 0.98,
                'y': 0.95,
                'fontsize': 9,
                'ha': 'right',
                'va': 'top',
                'bbox': dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.03')
            }
            if sample_count_style is not None:
                count_style.update(sample_count_style)
            ax.text(
            count_style['x'], count_style['y'],
            fr"$n = {sample_counts[i]:.0f}$",
            transform=ax.transAxes,
            ha=count_style['ha'],
            va=count_style['va'],
            fontsize=count_style['fontsize'],
            bbox=count_style['bbox']
            )

        # --- Add WD annotation if provided ---
        if annotations and annotations[i] is not None:
            # WD annotation
            ax.text(
            1.01, 0.05,
            fr"$\mathrm{{WD}}(\cdot, \mathcal{{U}}_\mathrm{{Top}}) = {annotations[i]:.5f}$",
            transform=ax.transAxes,
            ha='left',
            va='center',
            fontsize=8,
            )
            # Min/Max annotation (separate, colored)
            if i == 2:  # Excluding 'All' and 'Top', this is the third plot
                ax.text(
                    1.29, 0.05,
                    r"$\mathrm{Min}$",
                    transform=ax.transAxes,
                    ha='left',
                    va='center',
                    fontsize=10,
                    color='green'
                )
            elif i == len(annotations) - 1:
                ax.text(
                    1.29, 0.05,
                    r"$\mathrm{Max}$",
                    transform=ax.transAxes,
                    ha='left',
                    va='center',
                    fontsize=10,
                    color='red'
                )
    # Bottom axis
    axes[0].set_xlabel(xlabel)
    axes[0].tick_params(bottom=True, labelbottom=True)
    axes[0].spines['bottom'].set_visible(True)
    # Save the figure
    plt.savefig(os.path.join(baseOut, 'multi_kdes_{}.pdf'.format(property_name)))
    plt.savefig(os.path.join(baseOut, 'multi_kdes_{}.svg'.format(property_name)))
    plt.savefig(os.path.join(baseOut, 'multi_kdes_{}.png'.format(property_name)), dpi=600)
    if show: plt.show()


def convert_label_to_latex(name):
    if name == "All":
        return r"$\mathcal{U}$"
    elif name == "Top":
        return r"$\mathcal{U}_\mathrm{Top}$"
    elif name == "P":
        return r"$\mathcal{U}_\mathrm{Training}$"
    elif name.startswith("Ref"):
        suffix = name[3:]  # e.g., "_C3" or ""
        if suffix.startswith("_"):
            suffix = suffix[1:]
        return rf"$F_{{\bar{{\mathcal{{V}}}}}}(\mathcal{{V}})$" + (f" {suffix}" if suffix else "")
    elif name.startswith("PPAFM2Exp_CoAll_"):
        parts = name.split("_")
        try:
            l_values = [p[1:] for p in parts if p.startswith("L")]
            l1 = l_values[0] if len(l_values) > 0 else "?"
            l2 = l_values[1] if len(l_values) > 1 else "?"
        except IndexError:
            l1, l2 = "?", "?"
        suffix = parts[-1] if parts[-1].startswith("C") else ""
        latex = rf"$F_{{\mathcal{{V}}^{{\dagger}}}}^{{\lambda_\mathrm{{c}},\lambda_\mathrm{{i}}={l1},{l2}}} (\mathcal{{V}})$"
        return latex + f" {suffix}" if suffix else latex
    else:
        return name  # fallback
    
def extract_2d_group_data(df, ordered_groups):
    """
    Extract 2D data (Nx2 arrays) for each group from the dataframe.

    Parameters:
    - df: pandas DataFrame with columns 'group' and 'value', where each 'value' is a 2-element array.
    - ordered_groups: list of group names to extract in order

    Returns:
    - dict: group name -> (N, 2) numpy array
    """
    data_dict = {}
    for group in ordered_groups:
        group_df = df[df['group'] == group]
        values = np.array(group_df['value'].tolist())  # Convert list of arrays into 2D array
        if values.ndim == 2 and values.shape[1] == 2:
            data_dict[group] = values
        else:
            print(f"[Warning] Skipping group {group}: shape {values.shape} is not 2D with 2 columns")
    return data_dict


def plot_distribution_property(property_name):
    xlim_map = {
    "OO_dist": (0, 3.5),
    "OH_dist": (0, 1.25),
    "HOH_dist": (35, 180),
    "ThetaOH_dist": (0, 179.9),
    "Hbonds": (1.9, 3.49), 
    "OrderP": (0.00001, 0.0045)
    }

    ylim_map = {
    "Hbonds": (120, 179.9), 
    "OrderP": (0, 1.19) 
    }

    file_key_map = {
    "OO_dist": "OO",
    "OH_dist": "OH",
    "HOH_dist": "HOH",
    "ThetaOH_dist": "ZOH",  # special case
    "Hbonds": "OO_OHO",  # special case
    "OrderP": "sk_sg"
    }

    bw_method_map = {
    "OO_dist": 0.025,
    "OH_dist": 0.05,
    "HOH_dist": 0.025,
    "ThetaOH_dist": 0.08,  # special case
    }

    xlabel_map = {
        "OO_dist": r"$d_\mathrm{OO}$ ($\mathrm{\AA}$)",
        "OH_dist": r"$d_\mathrm{OH}$ ($\mathrm{\AA}$)",
        "HOH_dist": r"$\theta_\mathrm{HOH}$ ($^\circ$)",
        "ThetaOH_dist": r"$\theta_\mathrm{OH}$ ($^\circ$)",  # special case
        "Hbonds": rf'$d_{{\mathrm{{O}}_\mathrm{{d}}\mathrm{{O}}_\mathrm{{a}}}}$ (Å)',
        "OrderP": r'$S_k$'
    }

    ylabel_map = {
        "Hbonds": rf'$\theta_{{\mathrm{{O}}_\mathrm{{d}}\mathrm{{H}}\mathrm{{O}}_\mathrm{{a}}}}$ ($^\circ$)',
        "OrderP": r'$S_g$'
    }

    overlap_map = {
    "OO_dist": 0.7,
    "OH_dist": 0.7,
    "HOH_dist": 0.7,    
    "ThetaOH_dist": 0.7,  # special case
    }

    height_per_subplot_map = {
    "OO_dist": 5.0,
    "OH_dist": 5.0,
    "HOH_dist": 5.0,
    "ThetaOH_dist": 5.0,  # special case
    }

    # Tune sample-count label position/size per 1D property.
    # Coordinates are in axes fraction (0 to 1).
    sample_count_style_1d_map = {
        "OO_dist": {'x': 0.50, 'y': 0.2, 'fontsize': 10, 'ha': 'right', 'va': 'top'},
        "OH_dist": {'x': 0.50, 'y': 0.2, 'fontsize': 10, 'ha': 'right', 'va': 'top'},
        "HOH_dist": {'x': 0.90, 'y': 0.2, 'fontsize': 10, 'ha': 'right', 'va': 'top'},
        "ThetaOH_dist": {'x': 0.80, 'y': 0.2, 'fontsize': 10, 'ha': 'right', 'va': 'top'},
    }

    # Tune sample-count and WD text for 2D properties.
    sample_count_style_2d_map = {
        "Hbonds": {'x': 0.02, 'y': 0.40, 'fontsize': 10, 'ha': 'left', 'va': 'top'},
        "OrderP": {'x': 0.02, 'y': 0.90, 'fontsize': 10, 'ha': 'left', 'va': 'top'},
    }
    wd_text_style_2d_map = {
        "Hbonds": {'x': 0.02, 'y_with_count': 0.1, 'y_without_count': 0.10, 'fontsize': 10, 'ha': 'left', 'va': 'top'},
        "OrderP": {'x': 0.02, 'y_with_count': 0.1, 'y_without_count': 0.10, 'fontsize': 10, 'ha': 'left', 'va': 'top'},
    }

    if property_name not in file_key_map:
        raise ValueError(f"Unknown property_name: {property_name}")
    key = file_key_map[property_name]
    # Paths and config
    structure = 'Label'
    processedFolder = '../../processed_data/theory_distributions/'
    npzOut = os.path.join(processedFolder, structure)
    similarity_json = '../../processed_data/distribution_distances/similarities_Label_Top.json'
    if property_name not in ['Hbonds', 'OrderP']:
        reference_file = f'../../processed_data/structure_properties/P/{key}.npz'
    else:
        if property_name == 'Hbonds':
            reference_file = f'../../processed_data/structure_properties/P/Hbond.npz'
        elif property_name == 'OrderP':
            reference_file = f'../../processed_data/structure_properties/P/OrderP.npz'
        else:
            raise ValueError(f"Unknown property_name for reference file: {property_name}")
        
    # Load similarities
    with open(similarity_json, 'r') as f:
        similarities = json.load(f)

    # Get distances for the given property
    distance_items = {
        k: v[property_name]["wdistancec_nor"]
        for k, v in similarities.items()
        if property_name in v and "wdistancec_nor" in v[property_name]
    }
    filtered_distances = {k: v for k, v in distance_items.items() if k != 'P'}
    min_model = min(filtered_distances, key=filtered_distances.get)
    max_model = max(filtered_distances, key=filtered_distances.get)
    print(f"[{property_name}] Min: {min_model} ({filtered_distances[min_model]:.5f})")
    print(f"[{property_name}] Max: {max_model} ({filtered_distances[max_model]:.5f})")

    # Select models
    model_names = list(similarities.keys())
    selected_models = {
        # 'L20_L1': [m for m in model_names if 'L20_L1' in m],
        # 'L10_L10': [m for m in model_names if 'L10_L10' in m],
        # 'Ref': [m for m in model_names if 'Ref' in m],
        'L20_L1': [m for m in model_names if 'L20_L1' in m and 'Only' not in m],
        'L10_L10': [m for m in model_names if 'L10_L10' in m and 'Only' not in m],
        'Ref': [m for m in model_names if 'Ref' in m and 'Pure' not in m],
        'MinMax': [min_model, max_model]
    }
    all_selected = sorted(set().union(*selected_models.values()))
    # Make sure the order is All, Top, P, then the rest
    #model_tags = ['All', 'Top', 'P'] + [m for m in all_selected if m not in ['All', 'Top', 'P']] 
    model_tags = ['All', 'Top'] + [m for m in all_selected if m not in ['All', 'Top']] 

    # Collect data
    data = {'value': [], 'group': []}
    for model in all_selected:
        if property_name not in ['Hbonds', 'OrderP']:
            npz_path = f'../../processed_data/structure_properties/{model}/{key}.npz'
        else:
            if property_name == 'Hbonds':
                npz_path = f'../../processed_data/structure_properties/{model}/Hbond.npz'
            elif property_name == 'OrderP':
                npz_path = f'../../processed_data/structure_properties/{model}/OrderP.npz'
        if os.path.exists(npz_path):
            values = np.load(npz_path)[key]
            if values is not None and len(values) > 0:
                data['value'].extend(values)
                data['group'].extend([model] * len(values))
            else:
                print(f"Warning: empty data in {model}")
        else:
            print(f"Warning: {npz_path} not found")

    # # Add P (reference)
    # if os.path.exists(reference_file):
    #     values = np.load(reference_file)[key]
    #     data['value'].extend(values)
    #     data['group'].extend(['P'] * len(values))

    # Add All and Top
    for label in ['All', 'Top']:
        if property_name not in ['Hbonds', 'OrderP']:
            npz_path = os.path.join(npzOut, f'{key}_{label}.npz')
        else:
            if property_name == 'Hbonds':
                npz_path = os.path.join(npzOut, f'Hbond_{label}.npz')
            elif property_name == 'OrderP':
                npz_path = os.path.join(npzOut, f'OrderP_{label}.npz')
        if os.path.exists(npz_path):
            #values = np.load(npz_path)[property_name.split("_")[0]]
            values = np.load(npz_path)[key]
            data['value'].extend(values)
            data['group'].extend([label] * len(values))
        else:
            print(f"Warning: {npz_path} not found")

    # Build DataFrame
    df = pd.DataFrame(data)

    # Build distance map
    distance_map = {}
    for tag in model_tags:
        #if tag in ['All', 'Top', 'P']:
        if tag in ['All', 'Top']:
            distance_map[tag] = -1
        elif tag in similarities and property_name in similarities[tag]:
            distance_map[tag] = similarities[tag][property_name]['wdistancec_nor']
        else:
            distance_map[tag] = 999

    # Sort
    #fixed_order = ['All', 'Top', 'P']
    fixed_order = ['All', 'Top']
    others = sorted([g for g in distance_map if g not in fixed_order], key=lambda x: (distance_map[x], x))
    ordered_groups = fixed_order + others
    #ordered_groups = sorted(distance_map, key=lambda x: (distance_map[x], x))
    df['group'] = pd.Categorical(df['group'], categories=ordered_groups, ordered=True)
    df = df.sort_values('group')

    # Generate annotations (WD text), None for All/Top/P
    annotations = []
    for g in ordered_groups:
        #if g in ['All', 'Top', 'P']:
        if g in ['All', 'Top']:
            annotations.append(None)
        else:
            annotations.append(distance_map[g])
    # Calculate sample counts for each group
    sample_counts_list = [len(df[df['group'] == g]['value'].values) for g in ordered_groups]
    
    if property_name in ['OO_dist', 'OH_dist', 'HOH_dist', 'ThetaOH_dist']:
        plot_multi_kdes(
            datasets=[df[df['group'] == g]['value'].values for g in ordered_groups],
            ylabels=[convert_label_to_latex(g) for g in ordered_groups],
            fill=True,
            bw=bw_method_map[property_name],
            colormap=plt.cm.inferno,
            height_per_subplot=height_per_subplot_map[property_name],
            overlap_space=overlap_map[property_name],
            xlabel=xlabel_map.get(property_name, "Unknown Property"), 
            xlim=xlim_map.get(property_name, None),
            annotations=annotations, 
            property_name=property_name, 
            show=show,
            sample_counts=sample_counts_list,
            sample_count_style=sample_count_style_1d_map.get(property_name, None),
        )
    elif property_name in ['Hbonds', 'OrderP']:
        data_dict = extract_2d_group_data(df, ordered_groups)
        # Create sample count dict for 2D scatter
        sample_count_dict = {g: len(df[df['group'] == g]['value'].values) for g in ordered_groups}
        plot_multi_2d_scatter(
            data_dict=data_dict,
            distance_map=distance_map,
            convert_label_func=convert_label_to_latex,
            xlabel=xlabel_map[property_name],
            ylabel=ylabel_map[property_name],
            xlim=xlim_map[property_name],
            ylim=ylim_map[property_name],
            property_name=property_name, 
            show=show,
            sample_counts=sample_count_dict,
            sample_count_style=sample_count_style_2d_map.get(property_name, None),
            wd_text_style=wd_text_style_2d_map.get(property_name, None),
        )



plot_distribution_property("OO_dist")
plot_distribution_property("OH_dist")
plot_distribution_property("HOH_dist")
plot_distribution_property("ThetaOH_dist")
plot_distribution_property("Hbonds")
plot_distribution_property("OrderP")

#!/usr/bin/env python
import numpy as np
import pandas as pd
import json, os, re
import matplotlib.pyplot as plt
from utils import radar_plot

simcolor = '#ed9d2c'
expcolor = '#de461c'
dftcolor = '#2ca3cf'
bg07color = '#479FB1'
bv17color = '#6E7CBC'

plt.rcParams['font.size']=14
plt.rcParams['font.family']='Arial'
plt.rcParams['pdf.fonttype']=42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = True # Render text with LaTeX


if __name__ == '__main__':
    show = True
    inputFolder = '../../processed_data/distribution_distances'
    ground_truth, layer = 'Label', 'Top'
    #for layer in ['Top', 'All']:
    for layer in ['Top']:
        print('Comparing layer:', layer)
        outputFolder = '../../results/meanInRadar'
        os.makedirs(outputFolder, exist_ok=True)
        with open('{}/similarities_{}_{}.json'.format(inputFolder, ground_truth, layer), "r") as file:
            similarities = json.load(file)

        numeric_columns = ["OO", "OH", "HOH",  "ZOH", "Hbond", "OrderP"]  # Changed order of ZOH and Hbond
        # The cases that have multiple individuals trained on the same data 
        # Ref is add gaussian and radom crop to augment the data
        all_keys = ['Ref', 'PPAFM2Exp_CoAll_L20_L1_Elatest_Only',  'PPAFM2Exp_CoAll_L20_L1_Elatest', 'PPAFM2Exp_CoAll_L10_L10_Elatest_Only', 'PPAFM2Exp_CoAll_L10_L10_Elatest'] 
        for i, comp_key in enumerate(all_keys):
            print('Comparing: ', comp_key)
            fig, axs = plt.subplots(3, 1, figsize=(4, 12), subplot_kw=dict(polar=True))
            #fig.suptitle('Comparison of {} layer: {}'.format(layer, comp_key))
            index = 0
            for distance, y_label in zip(['wdistancec_nor', 'edistancec_nor', 'mdistancec_nor'], ['WD Score', 'ED Score', 'MMD Score']):
                print('Distance type:', distance)
                df = pd.DataFrame(columns = ['Structure', 'Truth', 'OO', 'OH', 'HOH', 'ZOH', 'Hbond', 'OrderP'])
                for i, (key, value) in enumerate(similarities.items()):
                    df.loc[i] = [key, ground_truth, value['OO_dist'][distance], value['OH_dist'][distance], value['HOH_dist'][distance], value['ThetaOH_dist'][distance], value['Hbonds'][distance], value['OrderP'][distance]]
                # Find the min and max values for each column for normalization
                min_values = df[numeric_columns].min()
                max_values = df[numeric_columns].max()
                min_values = np.array(min_values)
                max_values = np.array(max_values)
                print('Min values:', min_values)
                print('Max values:', max_values)

                # Find the reference performance
                ref_key = "Ref_Pure"
                ref_df = df[df["Structure"].str.contains(ref_key)]
                print(ref_df)
                # Get the numeric columns and compute the mean and std or ref_df 
                mean_values = ref_df[numeric_columns].mean()
                std_values = ref_df[numeric_columns].std() / np.sqrt(ref_df[numeric_columns].count())
                print('Mean', mean_values)
                print('Std', std_values)
                labels = [r"$d_{\mathrm{OO}}$", r"$d_{\mathrm{OH}}$", r"$\theta_{\mathrm{HOH}}$", r"$\theta_{\mathrm{ZOH}}$", r"$(d_{\mathrm{O_d}\mathrm{O_a}}, \theta_{\mathrm{O_d}\mathrm{H}\mathrm{O_a}})$", r"$(S_k, S_g)$"]
                title = y_label 
                radar_plot(ax=axs[index], mins=min_values, maxs=max_values, data=mean_values, color='k', errors=std_values, labels=labels, title=title, legend=rf'$F_{{\mathcal{{U}}}}(\mathcal{{V}})$: Pure Simulation' if index == 1 else None)

                # Individual plots for each model
                if comp_key == 'PPAFM2Exp_CoAll_L10_L10_Elatest':
                    pattern = r'^PPAFM2Exp_CoAll_L10_L10_Elatest(_C\d+)?$'
                    compare_data = df[df["Structure"].str.match(pattern)]
                elif comp_key == 'PPAFM2Exp_CoAll_L20_L1_Elatest':
                    pattern = r'^PPAFM2Exp_CoAll_L20_L1_Elatest(_C\d+)?$'
                    compare_data = df[df["Structure"].str.match(pattern)]
                else: 
                    compare_data = df[df["Structure"].str.contains(comp_key)] 
                mean_values = compare_data[numeric_columns].mean()
                std_values = compare_data[numeric_columns].std() / np.sqrt(compare_data[numeric_columns].count())
                if 'Only' in comp_key:
                    # Style translation only
                    legend = rf'$F_{{\widetilde{{\mathcal{{V}}}}}}(\mathcal{{V}})$: Style Translated'
                else:
                    if comp_key == 'Ref':
                        # Handcrafted
                        legend = rf'$F_{{\overline{{\mathcal{{V}}}}}}(\mathcal{{V}})$: Handcrafted'
                    else:
                        # Hybrid
                        legend = rf'$F_{{\mathcal{{V}}^{{\dagger}}}}(\mathcal{{V}})$: Hybrid'
                legend = legend if index == 1 else None
                if comp_key == 'Ref':
                    color = simcolor
                elif 'Only' in comp_key:
                    color = expcolor
                else:
                    color = dftcolor
                radar_plot(ax=axs[index], mins=min_values, maxs=max_values, data=mean_values, errors=std_values, color=color, labels=labels, legend=legend)


                
                index += 1
            plt.subplots_adjust(left=0.07, right=0.93, top=0.8, bottom=0.15, wspace=0.35)
            plt.savefig('{}/{}_comparision_to_{}_{}.png'.format(outputFolder, comp_key, ground_truth, layer), dpi=600)
            plt.savefig('{}/{}_comparision_to_{}_{}.svg'.format(outputFolder, comp_key, ground_truth, layer))
            plt.savefig('{}/{}_comparision_to_{}_{}.pdf'.format(outputFolder, comp_key, ground_truth, layer))
            if show: plt.show() 

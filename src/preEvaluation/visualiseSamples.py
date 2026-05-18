#!/usr/bin/env python

# %% 
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
def obtainFidelityValue(df, filename, className):
    """
    Obtain the fidelity value for a given filename and class name from the dataframe.
    """
    # Filter the dataframe for the given filename and class name
    filtered_df = df[(df['Filename'] == filename) & (df['Class'] == className)]
    
    # Check if any rows were found
    if filtered_df.empty:
        return None
    
    # Return the fidelity value
    return filtered_df.iloc[0]['Output Value']

# %%
if __name__ == "__main__":
    inputPath = "../../data/preEvaluate"
    fedilityFile = "../../processed_data/fidelityTest/fidelity_results.csv"
    resultsPath = "../../results/samples"
    if not os.path.exists(resultsPath): os.makedirs(resultsPath)

    # Read the fidelity results using pandas
    df = pd.read_csv(fedilityFile)
    print(df.head())
# %% 
    plt.rcParams['font.size']=14
    plt.rcParams['font.family']='Arial'
    plt.rcParams['pdf.fonttype']=42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['text.usetex'] = True # Render text with LaTeX

# %% 
    U = ["902.12.png", "374.03.png"] # "265.12.png", "155.14.png", "75.12.png",  "371.07.png"
    V = ["A190824.044700.dat.png", "H2O Au111-20210107-569.sxm.png"] # "A191211.113259.dat.png", "H2O Au111-20210107-067.sxm.png", "H2O Au111-20210107-564.sxm.png", "A180423.140652.dat.png", "A180423.171854.dat.png",

    #classes = ['realA', 'fakeB_Water-bilayerCGPPAFM2Exp_CoAllL30L10Elatest', 'noisedA_drift_0.3', 'realB', 'fakeA_lbd_50']
    classes = ['realA', 'fakeB_Water-bilayerCGPPAFM2Exp_CoAllL30L10Elatest', 'realB', 'fakeA_lbd_50']
    U_images = [f"{inputPath}/{classes[0]}/{u}" for u in U]
    V_tilde_images = [f"{inputPath}/{classes[1]}/{u}" for u in U]
    #E1_images = [f"{inputPath}/{classes[2]}/{u}" for u in U]

    V_images = [f"{inputPath}/{classes[2]}/{v}" for v in V]
    U_tilde_images = [f"{inputPath}/{classes[3]}/{v}" for v in V]

    #texts = [rf'$\mathcal{{U}}$', rf'$\tilde{{\mathcal{{V}}}}$', rf'$\mathcal{{V}}$', rf'$\tilde{{\mathcal{{U}}}}$']
    texts = [r'$u$', rf'$\tilde{{v}}$', r'$v$', rf'$\tilde{{u}}$']

    fig, axes = plt.subplots(nrows=len(U_images), ncols=4, figsize=(8, 6.25))
    
    # Add text above the top of the first row
    for j, text in enumerate(texts):
        axes[0, j].text(0.5, 1.01, text, ha='center', va='bottom', transform=axes[0, j].transAxes, fontsize=14)

    for i, (u_img, vt_img, v_img, ut_img) in enumerate(zip(U_images, V_tilde_images, V_images, U_tilde_images)):
        img_u = plt.imread(u_img)
        axes[i, 0].imshow(img_u, cmap='inferno')
        axes[i, 0].axis('off')
        fedility = obtainFidelityValue(df, U[i], classes[0])
        print(fedility)
        axes[i, 0].text(0.95, 0.95, rf'$u_{{{i+1}}}$', ha='right', va='top', transform=axes[i, 0].transAxes,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

        img_vt = plt.imread(vt_img)
        axes[i, 1].imshow(img_vt, cmap='inferno')
        axes[i, 1].axis('off')
        fedility = obtainFidelityValue(df, U[i], classes[1])
        axes[i, 1].text(0.95, 0.95, rf'$\tilde{{v}}_{{{i+1}}}$', ha='right', va='top', transform=axes[i, 1].transAxes,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
        
        img_v = plt.imread(v_img)
        axes[i, 2].imshow(img_v, cmap='inferno')
        axes[i, 2].axis('off')
        fedility = obtainFidelityValue(df, V[i], classes[2])
        axes[i, 2].text(0.95, 0.95, rf'$v_{{{i+1}}}$', ha='right', va='top', transform=axes[i, 2].transAxes,
                        bbox=dict(facecolor='white',edgecolor='none', alpha=0.5))

        img_ut = plt.imread(ut_img)
        axes[i, 3].imshow(img_ut, cmap='inferno')
        axes[i, 3].axis('off')
        fedility = obtainFidelityValue(df, V[i], classes[3])
        axes[i, 3].text(0.95, 0.95, rf'$\tilde{{u}}_{{{i+1}}}$', ha='right', va='top', transform=axes[i, 3].transAxes,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

    fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.0001, right=0.9999, top=0.95, bottom=0.001)
    plt.savefig(f"{resultsPath}/samples.png", dpi=600)
    plt.savefig(f"{resultsPath}/samples.pdf")
    plt.savefig(f"{resultsPath}/samples.svg")
    plt.show()

 # %%

#!/usr/bin/env python

import os
import json
import pandas as pd
import numpy as np
import torch
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from scipy.stats import wasserstein_distance

# Dataset for folder of 1-channel images (converted to RGB uint8)
class ImageFolderDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png'))
        ])
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((299, 299), interpolation=Image.BICUBIC),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        img = self.transform(img)
        img = torch.from_numpy(np.array(img))
        img = img.permute(2, 0, 1).to(torch.uint8)
        return img

def calculate_fid(real_folder, fake_folder, batch_size=32, feature_dim=2048):
    real_dataset = ImageFolderDataset(real_folder)
    fake_dataset = ImageFolderDataset(fake_folder)

    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=feature_dim).to(device)

    for real_batch, fake_batch in zip(real_loader, fake_loader):
        fid.update(real_batch.to(device), real=True)
        fid.update(fake_batch.to(device), real=False)

    return fid.compute().item()

# --------- Set seeds for reproducibility ---------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# CSV is for Wasserstein distance calculation
csv_df = pd.read_csv("../../processed_data/fidelityTest/fidelity_results.csv")
output_dir = "../../results/distance_analysis"
os.makedirs(output_dir, exist_ok=True)

U_values = csv_df[csv_df["Class"].str.contains("realA")]["Output Value"].dropna().values
V_values = csv_df[csv_df["Class"].str.contains("realB")]["Output Value"].dropna().values

# V_tilde values
vt_mask = csv_df["Class"].str.contains(r"fakeB_Water-bilayerCGPPAFM2Exp_CoAllL\d+L[\d\.]+Elatest")
vt_types = csv_df.loc[vt_mask, "Class"].unique()
vt_dfs = [csv_df[csv_df["Class"] == vt]["Output Value"].dropna().values for vt in vt_types]

#noiseTypes = ['gaussian', 'speckle', 'drift', 'blur', 'saltpepper', 'gamma', 'gradient_previous', 'noise_previous', 'cutout_previous', 'combined_previous']
noiseTypes = ['noise_previous', 'cutout_previous', 'gradient_previous', 'saltpepper', 'combined_previous']

results = {}

# Wasserstein distances
results["W(V, V)"] = (wasserstein_distance(V_values, V_values), 0.0)
results["W(U, V)"] = (wasserstein_distance(U_values, V_values), 0.0)

if vt_dfs:
    vt_distances = [wasserstein_distance(vt, V_values) for vt in vt_dfs if len(vt) > 0]
    mean_vt = np.mean(vt_distances)
    se_vt = np.std(vt_distances, ddof=1) / np.sqrt(len(vt_distances))
    results["W(V_tilde, V)"] = (mean_vt, se_vt)

for noise_type in noiseTypes:
    subclass_mask = csv_df["Class"].str.contains(f"noisedA_{noise_type}")
    subclass_classes = csv_df.loc[subclass_mask, "Class"].unique()

    distances = []
    for cls in subclass_classes:
        values = csv_df[csv_df["Class"] == cls]["Output Value"].dropna().values
        if len(values) > 0:
            distances.append(wasserstein_distance(values, V_values))

    if distances:
        mean_d = np.mean(distances)
        se_d = np.std(distances, ddof=1) / np.sqrt(len(distances))
        results[f"W(noisedA_{noise_type}, V)"] = (mean_d, se_d)

# Compute FID values from images
U_path = "../../data/preEvaluate/realA"
V_path = "../../data/preEvaluate/realB"

# 1. FID(U, V): Simulation - Experiment gap
fid_U_V = calculate_fid(U_path, V_path)
results["FID(U, V)"] = (fid_U_V, 0.0)

# 2. FID(V_tilde, V): New gap
vt_fids = []
for vt_folder in vt_types:
    vt_path = os.path.join("../../data/preEvaluate", vt_folder)
    if os.path.isdir(vt_path):
        try:
            vt_fids.append(calculate_fid(vt_path, V_path))
        except:
            continue

if vt_fids:
    fid_mean = np.mean(vt_fids)
    fid_se = np.std(vt_fids, ddof=1) / np.sqrt(len(vt_fids))
    results["FID(V_tilde, V)"] = (fid_mean, fid_se)

# 3. FID(noisedA_type, V): New gap
for noise_type in noiseTypes:
    noise_folders = [f for f in os.listdir("../../data/preEvaluate") if f.startswith(f"noisedA_{noise_type}_")]
    fids = []
    for folder in noise_folders:
        path = os.path.join("../../data/preEvaluate", folder)
        if os.path.isdir(path):
            try:
                fids.append(calculate_fid(path, V_path))
            except:
                continue

    if fids:
        fid_mean = np.mean(fids)
        fid_se = np.std(fids, ddof=1) / np.sqrt(len(fids))
        results[f"FID(noisedA_{noise_type}, V)"] = (fid_mean, fid_se)

# Print and save all results
for k, (v, se) in results.items():
    print(f"{k}: {v:.4f} ± {se:.4f}")

with open(os.path.join(output_dir, "all_distances.txt"), "w") as f:
    for k, (v, se) in results.items():
        f.write(f"{k}: {v:.4f} ± {se:.4f}\n")

# Save to JSON for Python
json_path = os.path.join(output_dir, "all_distances.json")
with open(json_path, "w") as fjson:
    json.dump({k: {"mean": v, "stderr": se} for k, (v, se) in results.items()}, fjson, indent=2)

# Save to CSV
csv_path = os.path.join(output_dir, "all_distances.csv")
df_results = pd.DataFrame([
    {"Metric": k, "Mean": v, "StdErr": se} for k, (v, se) in results.items()
])
df_results.to_csv(csv_path, index=False)

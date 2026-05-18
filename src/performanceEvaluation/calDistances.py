#!/usr/bin/env python
import os, json
import numpy as np
from water import sinkhorn_2d_distance, energy_distance, mmd_rbf_distance
from scipy.stats import wasserstein_distance

def write_similarity_to_file(file_path, similarities):
    with open(file_path, 'w') as f:
        json.dump(similarities, f, indent=4)

def normalize(data, property):
    ranges = {'OO': (0, 3.5), 'OH': (0, 1.25), 'HOH': (0, 180), 'ZOH': (0, 180), 'Hbond': [[0, 3.5], [120, 180]], 'OrderP': [[0, 1], [0, 1]]}
    if property != 'Hbond' and property != 'OrderP':
        nor_data = (data - ranges[property][0]) / (ranges[property][1] - ranges[property][0])
    else:
        nor_data_col1 = (data[:, 0] - ranges[property][0][0]) / (ranges[property][0][1] - ranges[property][0][0])
        nor_data_col2 = (data[:, 1] - ranges[property][1][0]) / (ranges[property][1][1] - ranges[property][1][0])
        nor_data = np.column_stack((nor_data_col1, nor_data_col2))
    return nor_data


if __name__ == '__main__':
    inputFolder = '../../processed_data/structure_properties'  # Predictions
    theoryFolder = '../../processed_data/theory_distributions' # Reference DFT 
    outputFolder = '../../processed_data/distribution_distances'
    os.makedirs(outputFolder, exist_ok=True)
    
    for ground_truth in ['Label']: # or ['Label', 'P']
        for layer in ['All', 'Top']:
            results_file = os.path.join(outputFolder, 'similarities_{}_{}.json'.format(ground_truth, layer))
            structures = [f for f in os.listdir(inputFolder) if os.path.isdir(os.path.join(inputFolder, f))]
            all_similarities = {}
            for structure in structures:
                print('Calculating for structure: {}'.format(structure))
                similarities = {}
        
                # O-O
                data = np.load(f'{theoryFolder}/{ground_truth}/OO_{layer}.npz')
                distances = data['OO']
                datac = np.load('{}/{}/OO.npz'.format(inputFolder, structure))
                distancesc = datac['OO']
                wdistancec = wasserstein_distance(distances, distancesc)
                # Normalize 
                distances_nor = normalize(distances, 'OO')
                distancesc_nor = normalize(distancesc, 'OO')
                wdistancec_nor = wasserstein_distance(distances_nor, distancesc_nor)
                edistancec_nor = energy_distance(distances_nor, distancesc_nor)
                mdistancec_nor = mmd_rbf_distance(distances_nor, distancesc_nor)
                similarities['OO_dist'] = {'wdistancec': wdistancec, 'wdistancec_nor': wdistancec_nor, 'edistancec_nor': edistancec_nor, 'mdistancec_nor': mdistancec_nor}
        
                # O-H
                data = np.load(f'{theoryFolder}/{ground_truth}/OH_{layer}.npz')
                distances = data['OH']
                datac = np.load('{}/{}/OH.npz'.format(inputFolder, structure))
                distancesc = datac['OH']
                wdistancec = wasserstein_distance(distances, distancesc)
                # Normalize
                distances_nor = normalize(distances, 'OH')
                distancesc_nor = normalize(distancesc, 'OH')
                wdistancec_nor = wasserstein_distance(distances_nor, distancesc_nor)
                edistancec_nor = energy_distance(distances_nor, distancesc_nor)
                mdistancec_nor = mmd_rbf_distance(distances_nor, distancesc_nor)
                similarities['OH_dist'] = {'wdistancec': wdistancec, 'wdistancec_nor': wdistancec_nor, 'edistancec_nor': edistancec_nor, 'mdistancec_nor': mdistancec_nor}
        
                # H-O-H 
                data = np.load(f'{theoryFolder}/{ground_truth}/HOH_{layer}.npz')
                angles = data['HOH']
                datac = np.load('{}/{}/HOH.npz'.format(inputFolder, structure))
                anglesc = datac['HOH'] 
                wdistancec = wasserstein_distance(angles, anglesc)
                # Normalize
                angles_nor = normalize(angles, 'HOH')
                anglesc_nor = normalize(anglesc, 'HOH')
                wdistancec_nor = wasserstein_distance(angles_nor, anglesc_nor)
                edistancec_nor = energy_distance(angles_nor, anglesc_nor)
                mdistancec_nor = mmd_rbf_distance(angles_nor, anglesc_nor)
                similarities['HOH_dist'] = {'wdistancec': wdistancec, 'wdistancec_nor': wdistancec_nor, 'edistancec_nor': edistancec_nor, 'mdistancec_nor': mdistancec_nor}
        
                # Z-O-H 
                data = np.load(f'{theoryFolder}/{ground_truth}/ZOH_{layer}.npz')
                angles = data['ZOH']
                datac = np.load('{}/{}/ZOH.npz'.format(inputFolder, structure))
                anglesc = datac['ZOH']
                wdistancec = wasserstein_distance(angles, anglesc)
                # Normalize
                angles_nor = normalize(angles, 'ZOH')
                anglesc_nor = normalize(anglesc, 'ZOH')
                wdistancec_nor = wasserstein_distance(angles_nor, anglesc_nor)
                edistancec_nor = energy_distance(angles_nor, anglesc_nor)
                mdistancec_nor = mmd_rbf_distance(angles_nor, anglesc_nor)
                similarities['ThetaOH_dist'] = {'wdistancec': wdistancec, 'wdistancec_nor': wdistancec_nor, 'edistancec_nor': edistancec_nor, 'mdistancec_nor': mdistancec_nor}
        
                # Hbond
                data = np.load(f'{theoryFolder}/{ground_truth}/Hbond_{layer}.npz')['OO_OHO']
                datac = np.load('{}/{}/Hbond.npz'.format(inputFolder, structure))['OO_OHO']
                wdistancec  = sinkhorn_2d_distance(data, datac)
                # Normalize
                data_nor = normalize(data, 'Hbond')
                datac_nor = normalize(datac, 'Hbond')
                wdistancec_nor  = sinkhorn_2d_distance(data_nor, datac_nor)
                edistancec_nor = energy_distance(data_nor, datac_nor)
                mdistancec_nor = mmd_rbf_distance(data_nor, datac_nor)
                similarities['Hbonds'] = {'wdistancec': wdistancec, 'wdistancec_nor': wdistancec_nor, 'edistancec_nor': edistancec_nor, 'mdistancec_nor': mdistancec_nor}
        
                # Order parameter 2d
                data = np.load(f'{theoryFolder}/{ground_truth}/OrderP_{layer}.npz')['sk_sg']
                datac = np.load('{}/{}/OrderP.npz'.format(inputFolder, structure))['sk_sg']
                wdistancec  = sinkhorn_2d_distance(data, datac)
                # Normalize
                data_nor = normalize(data, 'OrderP')
                datac_nor = normalize(datac, 'OrderP')
                wdistancec_nor  = sinkhorn_2d_distance(data_nor, datac_nor)
                edistancec_nor = energy_distance(data_nor, datac_nor)
                mdistancec_nor = mmd_rbf_distance(data_nor, datac_nor)
                similarities['OrderP'] = {'wdistancec': wdistancec, 'wdistancec_nor': wdistancec_nor, 'edistancec_nor': edistancec_nor, 'mdistancec_nor': mdistancec_nor}
        
                # Store similarities
                all_similarities[structure] = similarities
        
        
            write_similarity_to_file(results_file, all_similarities)


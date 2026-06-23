#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from water import read_samples_from_folder 
from water import mean_rdf, mean_adf, mean_distance_distribution, mean_adf_OH, cal_all_hydrogen_bonds
from water import compute_sk_sg_all  
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution


if __name__ == '__main__':
    structurePath = '../../data/structures/predictions' 
    baseOut = '../../processed_data/structure_properties'
    structures = [f for f in os.listdir(structurePath) if
               os.path.isdir(os.path.join(structurePath, f))]
    for structure in structures:
        print('Calculating for structure: {}'.format(structure))
        # Create the output folders
        outputFolder = os.path.join(baseOut, structure)
        print(outputFolder)
        os.makedirs(outputFolder, exist_ok=True)
        print(f"Folder created: {os.path.exists(outputFolder)}")
        # Read the samples
        sampleFolder = os.path.join(structurePath, structure, 'Prediction_c')
        samples = read_samples_from_folder(sampleFolder)
        print('Number of input samples: {}'.format(len(samples)))
        print('Calculating for structure: {}'.format(structure))

        # RDF and OO, OH distances distirbution
        print('Calulating RDF ...')
        # O-O 
        r_max = 3.5
        mic = False
        onlyDistances = True
        OO_distances = mean_rdf(samples, 'O', 'O', r_max=r_max, mic=mic, onlyDistances=onlyDistances)
        np.savez('{}/OO.npz'.format(outputFolder), OO=OO_distances, r_max=r_max)
    
        # O-H 
        r_max = 1.25
        onlyDistances = True
        OH_distances = mean_rdf(samples, 'O', 'H', r_max=r_max, mic=mic, onlyDistances=onlyDistances)
        np.savez('{}/OH.npz'.format(outputFolder), OH=OH_distances, r_max=r_max)

        # ADF
        r_max = 1.25
        print('Calulating ADF ...')
        firstTwo = False
        onlyAngle = True
        # H-O-H: Water angle
        y_lim = 0.3 if firstTwo else 0.02
        angles = mean_adf(samples, 'H', 'O', 'H', r_max=r_max, firstTwo=firstTwo, mic=mic, onlyAngle=onlyAngle)
        np.savez('{}/HOH.npz'.format(outputFolder), HOH=angles, r_max=r_max)
     
        # Theta of OH and z-axis 
        r_max = 1.25
        y_lim = 0.04
        angles = mean_adf_OH(samples, r_max = r_max, firstTwo=False, mic=False, onlyAngle=True, onlyFree=True)
        np.savez('{}/ZOH.npz'.format(outputFolder), ZOH=angles, r_max = r_max)

        # H-bonds 
        print('Calculating hydrogen bonds ...')
        hbonds = cal_all_hydrogen_bonds(samples)
        distances_da = [hb[3] for hb in hbonds]
        angles_dha = [hb[4] for hb in hbonds]
        distance_angle = np.array([distances_da, angles_dha]).T
        np.savez('{}/Hbond.npz'.format(outputFolder), hbond=hbonds, OO_OHO=distance_angle)

        # Order parameters: sg-sk 
        r_max = 3.5
        print('Calculating order parameters ...')
        sks, sgs = compute_sk_sg_all(samples, r_max=r_max)
        sk_sg = np.array([sks, sgs]).T
        np.savez('{}/OrderP.npz'.format(outputFolder), sk_sg=sk_sg)

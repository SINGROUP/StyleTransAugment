from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
# ASE viewer
from ase.visualize import view
from scipy.stats import gaussian_kde, wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from geomloss import SamplesLoss
import torch

def read_xyz_with_atomic_numbers(file_path):
    '''
    Read coordinates from an XYZ file and return an Atoms object.
    '''
    atomic_numbers_to_symbols = {1: 'H', 8: 'O', 29: 'Cu',  79: 'Au'}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Read number of atoms and comment line
    num_atoms = int(lines[0])
    comment = lines[1].strip()
    #print(comment)

    symbols = []
    positions = []
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        atomic_number = int(parts[0])
        x, y, z = map(float, parts[1:4])
        
        symbol = atomic_numbers_to_symbols[atomic_number]
        symbols.append(symbol)
        positions.append([x, y, z])
    positions = np.array(positions)
    return Atoms(symbols=symbols, positions=positions)

def calculate_lattice_vectors(atoms):
    '''
    Calculate lattice vectors for a system 
    with non-orthogonal XY plane and orthogonal Z
    input: 
        atoms: ASE Atoms object for the substrate
    return: 
        Lattice vectors
    '''
    positions = atoms.positions
    xy_positions = positions[:, :2]
    atomNum = xy_positions.shape[0]
    # print(f'Number of atoms: {atomNum}')

    miny = np.min(xy_positions[:, 1])
    maxy = np.max(xy_positions[:, 1])
    first_row = xy_positions[np.where(xy_positions[:, 1] == miny)]
    last_row = xy_positions[np.where(xy_positions[:, 1] == maxy)]
    #print(f'First row:\n {first_row}', first_row.shape)
    #print(f'Last row:\n {last_row}', last_row.shape)
    if first_row.shape[0] != last_row.shape[0]:
        raise ValueError("The number of atoms in the first row is not equal to the number of atoms in the last row.")

    atomNumFirstRow = first_row.shape[0]
    spaceX = (first_row[-1][0] - first_row[0][0]) / (atomNumFirstRow - 1)
    a = np.array([atomNumFirstRow * spaceX, 0, 0])
    #print('Vector x', a)

    atomNumPerRow = first_row.shape[0]
    atomNumPerCol = atomNum // atomNumPerRow
    #print(f'Number of atoms per column: {atomNumPerCol}')
    spaceY = (last_row[0][1] - first_row[0][1]) / (atomNumPerCol - 1)
    b = np.array([last_row[0][0] + spaceX/2, atomNumPerCol * spaceY, 0])
    #print('Vector y', b)
    # The z vector is orthogonal and fixed
    c = np.array([0, 0, 30.0])
    # Combine vectors to form the lattice matrix
    lattice_vectors = np.array([a, b, c])
    return lattice_vectors

def read_samples(simulation, toSee):
    '''
    Read a list of sample paths from a list of simulations
    Input: 
        simulations: a folder
        toSee: tuple of two integers, number of folders and number of xyz files to see  
    Output: a list of sample paths
    '''
    samples = []
    train_folders = [f for f in os.listdir(simulation) if 'train' in f]
    folderToSee = int(toSee[0]) 
    for train_folder in train_folders[0:folderToSee]:
        train_folder_path = os.path.join(simulation, train_folder)
        xyz_files = [f for f in os.listdir(train_folder_path) if f.endswith('.xyz')]
        xyzToSee = int(toSee[1])
        for xyz_file in  xyz_files[0:xyzToSee]:
            structure = os.path.join(train_folder_path, xyz_file) 
            samples.append(structure)
    return samples

def read_samples_from_folder(folder):
    '''
    Read a list of sample paths from a folder
    Input: 
        folder: a folder
        toSee: a integer, number of xyz files to see  
    Output: a list of sample paths
    '''
    samples = []
    xyz_files = [f for f in os.listdir(folder) if f.endswith('.xyz')]
    for xyz_file in  xyz_files:
        structure = os.path.join(folder, xyz_file) 
        samples.append(structure)
    return samples


def find_closest_neighbors(atoms, B_idx, A, num_neighbors=2, mic=True):
    '''
    Find the closest neighbor atoms with type A near atom B (Assume A, C are the same type of atoms)
    '''
    neighbor_indices = [i for i, atom in enumerate(atoms) if (i!=B_idx and atom.symbol==A)]
    distances = atoms.get_distances(B_idx, neighbor_indices, mic=mic)

    # Sort based on distance and pick the closest two
    closest_neighbors = sorted(distances)[:num_neighbors]

    # Find the indices of the closest hydrogens
    closest_neighbors_indices = [neighbor_indices[i] for i, d in enumerate(distances) if d in closest_neighbors]
    return closest_neighbors_indices

def find_neighbors(atoms, B_idx, A, r_max=10, mic=True):
    '''
    Find  neighbor atoms whithin r_max with type A near atom B (Assume A, C are the same type of atoms)
    Return:
        indices: np.array, indices of the neighbors
    '''
    # Find all neighbors of type A and exclude the atom itself 
    neighbor_indices = np.array([i for i, atom in enumerate(atoms) if (i!=B_idx and atom.symbol==A)])
    
    # Leave the neighbors within r_max
    distances = atoms.get_distances(B_idx, neighbor_indices, mic=mic)
    mask = (distances <= r_max) 
    neighbor_indices = neighbor_indices[mask]

    # Find all pairs of neighbors and return the indices
    close_neighbors_indices = []
    for i in range(len(neighbor_indices)):
        for j in range(i+1, len(neighbor_indices)):
            close_neighbors_indices.append([neighbor_indices[i], B_idx, neighbor_indices[j]])
    return np.array(close_neighbors_indices)


def cal_distances(atoms, A, B, r_max=4, mic=True, onlyDistances=False, zThresholdO=4.85, aboveZthres=None):
    '''
    Calculate distances between atom type A and B
    input: 
        atoms: ASE Atoms object
        A: str, atom type A
        B: str, atom type B
        r_max: float, maximum distance to consider
        mic: bool, whether to consider PBC
        onlyDistances: bool, whether to return only distances
        zThresholdO: float, z threshold of O atom (related to the Au surface) to separate the first and second layers water molecules
        aboveZthres: None: select all; True: select above the z threshold; False: select below the z threshold
    return:
        if onlyDistances:
            AB_all_distance: list of float, all distances between atom type A and B
        else:
            AB_all_distance: list of float, all distances between atom type A and B
            rho: float, density of atom type B
            refNum: int, reference number of atom type A
    '''
    positions = atoms.positions
    symbols = atoms.get_chemical_symbols()
    if aboveZthres is not None and A == 'O':
        #print('Debug: z threshold O', zThresholdO)
        # Calculate the mean value of z positions of Au atoms 
        Au_indices = [i for i, s in enumerate(symbols) if s == 'Au']
        Au_positions = positions[Au_indices]
        z_mean = np.max(Au_positions[:, 2]) # as the reference plane
        #print('Debug: Au_indices', Au_indices)
        #print('Debug: z mean Au surface', z_mean)
        if aboveZthres == True:
            A_indices = [i for i, s in enumerate(symbols) if s == A and positions[i][2] - z_mean > zThresholdO]
        elif aboveZthres == False:
            A_indices = [i for i, s in enumerate(symbols) if s == A and positions[i][2] - z_mean <= zThresholdO] 
        #print('Debug: A_indices', A_indices)
    else:   
        A_indices = [i for i, s in enumerate(symbols) if s == A]
    B_indices = [i for i, s in enumerate(symbols) if s == B]
    B_positions = positions[B_indices]

    AB_all_distance = []
    for count, A_idx in enumerate(A_indices):
        # Exclude the self pair
        B_indices_ = [i for i in B_indices if i != A_idx] if A == B else B_indices 
        distances = atoms.get_distances(A_idx, B_indices_, mic=mic) 
        # Only leave the distances within r_max
        distances = [d for d in distances if d <= r_max]
        AB_all_distance.extend(distances)
    if onlyDistances:
        return AB_all_distance
    else:
        # Obtain the density rho of B particles 
        z_max = np.max(B_positions[:, 2])
        z_min = np.min(B_positions[:, 2])
        ratio = (z_max - z_min) / atoms.cell[2][2]
        effective_volume = atoms.get_volume() * ratio
        rho = len(B_indices) / effective_volume
        # Reference numbers
        refNum = len(A_indices)
        return AB_all_distance, rho,  refNum 

def cal_angles(atoms, A, B, C, r_max=10.0, firstTwo=False, mic=True, zThresholdB=4.85, aboveZthres=None):
    '''
    Calculate angles between atom type A, B, and C, where B is the central atom, A and C are the first and second neighbors

    Input:
        atoms: ASE Atoms object
        A: str, atom type A
        B: str, atom type B
        C: str, atom type C
        r_max: float, maximum distance to consider
        firstTwo: bool, whether to consider only the first two neighbors
        mic: bool, whether to consider PBC
        zThresholdB: float, z threshold of B atom (related to the Au surface) to separate the first and second layers water molecules
        aboveZthres: None: select all; True: select above the z threshold; False: select below the z threshold

    Return:
        angles: list of angles
    '''
    symbols = atoms.get_chemical_symbols()
    if aboveZthres is not None:
        # Calculate the mean value of z positions of Au atoms 
        Au_indices = [i for i, s in enumerate(symbols) if s == 'Au']
        Au_positions = atoms.positions[Au_indices]
        z_mean = np.max(Au_positions[:, 2])
        if aboveZthres == True:
            B_indices = [i for i, s in enumerate(symbols) if s == B and atoms.positions[i][2] - z_mean > zThresholdB]
        elif aboveZthres == False:
            B_indices = [i for i, s in enumerate(symbols) if s == B and atoms.positions[i][2] - z_mean <= zThresholdB] 
    else:
        B_indices = [i for i, s in enumerate(symbols) if s == B]
    angles = []
    for B_idx in B_indices:
        if firstTwo:
            A_idx, C_idx = find_closest_neighbors(atoms, B_idx, A, num_neighbors=2, mic=mic)
            angle = atoms.get_angle(A_idx, B_idx, C_idx, mic=mic)
            angles.append(angle)
        else:
            neighbor_indices = find_neighbors(atoms, B_idx, A, r_max=r_max, mic=mic)
            #print('Debug: \n', neighbor_indices)
            if len(neighbor_indices) != 0:
                temp_angles = atoms.get_angles(neighbor_indices, mic=mic)
                angles.extend(list(temp_angles))
    return angles 

def find_hydrogen_bonds(structure, distance_cutoff=3.5, angle_cutoff=120, zThresholdO=4.85, aboveZthres=None):
    """
    function to find hydrogen bonds in a structure
    :param structure: ASE Atoms object
    :param distance_cutoff: float, maximum distance between donor and acceptor O atoms
    :param angle_cutoff: float, minimum angle D-H...A
    :return: list of tuples with the indices of the donor O, H, acceptor O atoms, the distance O...O, the angle D-H...A and the angle H-D...A
    """
    hydrogen_bonds = []
    # If aboveZthres is not None, calculate the mean value of z positions of Au atoms

    if aboveZthres is not None:
        Au_indices = [i for i, s in enumerate(structure.get_chemical_symbols()) if s == 'Au']
        Au_positions = structure.positions[Au_indices]
        z_mean_Au = np.mean(Au_positions[:, 2])
        #print('Debug: z mean Au surface', z_mean_Au)

    for i, donor_atom in enumerate(structure):
        if donor_atom.symbol == 'O': # Donor atom is an oxygen
            donor_pos = donor_atom.position
            # Find hydrogens bonded to this oxygen (within a smaller distance)
            bonded_hydrogens = [h_atom for h_atom in structure if h_atom.symbol == 'H' and structure.get_distance(i, h_atom.index) < 1.2]
            for h_atom in bonded_hydrogens:
                h_pos = h_atom.position
                # Check potential acceptor atoms
                for j, acceptor_atom in enumerate(structure):
                    if acceptor_atom.symbol == 'O' and acceptor_atom.index != i:
                        acc_pos = acceptor_atom.position
                        # Use OO distance for the cutoff
                        distance_oo = structure.get_distance(i, j)
                        if distance_oo <= distance_cutoff:
                            # Calculate angle D-H...A
                            vector_hd = donor_pos - h_pos
                            vector_ha = acc_pos - h_pos
                            cosine_angle = np.dot(vector_hd, vector_ha) / (np.linalg.norm(vector_hd) * np.linalg.norm(vector_ha))
                            cosine_angle = np.clip(cosine_angle, -1.0, 1.0) # Avoid numerical floating-point precision errors
                            #print(f"vector_hd: {vector_hd}, vector_ha: {vector_ha}, cosine_angle: {cosine_angle}")  # Debug statement
                            angle = np.degrees(np.arccos(cosine_angle))
                            #print('Debug: ', cosine_angle, angle, angle >= angle_cutoff) if (cosine_angle > 1 or cosine_angle < -1) else None
                            if angle >= angle_cutoff:
                                # Calculate angle HD...A
                                vector_da = acc_pos - donor_pos
                                vector_dh = h_pos - donor_pos
                                cosine_angle = np.dot(vector_da, vector_dh) / (np.linalg.norm(vector_da) * np.linalg.norm(vector_dh))
                                cosine_angle = np.clip(cosine_angle, -1.0, 1.0) # Avoid numerical floating-point precision errors
                                angle_hda = np.degrees(np.arccos(cosine_angle))
                                if aboveZthres is  None: # all
                                    hydrogen_bonds.append((donor_atom.index, h_atom.index, acceptor_atom.index, distance_oo, angle, angle_hda))
                                elif aboveZthres is True: # top layer
                                    if donor_pos[2] - z_mean_Au > zThresholdO:
                                        hydrogen_bonds.append((donor_atom.index, h_atom.index, acceptor_atom.index, distance_oo, angle, angle_hda))
                                elif aboveZthres is False: # bottom layer
                                    if donor_pos[2] - z_mean_Au <= zThresholdO:
                                        hydrogen_bonds.append((donor_atom.index, h_atom.index, acceptor_atom.index, distance_oo, angle, angle_hda))
    return hydrogen_bonds

def cal_all_hydrogen_bonds(samples, distance_cutoff=3.5, angle_cutoff=120, zThresholdO=4.85, aboveZthres=None):
    '''
    Find all hydrogen bonds in a list of samples
    '''
    all_hydrogen_bonds = []
    for sample in tqdm(samples):
        atoms = read_xyz_with_atomic_numbers(sample)
        hydrogen_bonds = find_hydrogen_bonds(atoms, distance_cutoff=distance_cutoff, angle_cutoff=angle_cutoff, zThresholdO=zThresholdO, aboveZthres=aboveZthres)
        all_hydrogen_bonds.extend(hydrogen_bonds)
    return all_hydrogen_bonds

def cal_angles_OH(atoms, r_max=1, mic=True, zThresholdO=4.85, onlyFree=True, aboveZthres=None): 
    '''
    Calculate the angles between OH an z axis
    '''
    vectorUP = np.array([0, 0, 1])
    symbols = atoms.get_chemical_symbols()
    if aboveZthres is not None:
        # Calculate the mean value of z positions of Au atoms 
        Au_indices = [i for i, s in enumerate(symbols) if s == 'Au']
        Au_positions = atoms.positions[Au_indices]
        z_mean = np.max(Au_positions[:, 2])
        if aboveZthres == True:
            O_indices = [i for i, s in enumerate(symbols) if s == 'O' and atoms.positions[i][2] - z_mean > zThresholdO]
        elif aboveZthres == False:
            O_indices = [i for i, s in enumerate(symbols) if s == 'O' and atoms.positions[i][2] - z_mean <= zThresholdO]
    else:
        O_indices = [i for i, s in enumerate(symbols) if s == 'O'] 
    angles = []
    #view(atoms)
    #input('Press Enter to continue...')
    for O_idx in O_indices:
        neighbor_indices = find_neighbors(atoms, O_idx, 'H', r_max=r_max, mic=mic) # H-O-H
        #print('Debug: Neighbor indices', neighbor_indices)
        #print('Debug: Number of neighbors', len(neighbor_indices))
        if len(neighbor_indices) == 1:
            neighbor_indices = neighbor_indices[0] # [[ H1, O, H2]] -> [H1, O, H2]
            #print('Debug: Neighbor indices', neighbor_indices)
            for H_idx in (neighbor_indices[0], neighbor_indices[-1]):
                OH_vector = atoms.positions[H_idx] - atoms.positions[O_idx]
                angle_wrt_z = np.arccos(np.dot(OH_vector, vectorUP) / np.linalg.norm(OH_vector)) * 180 / np.pi
                if onlyFree:
                    is_free = True
                    for O_acc_idx in O_indices:
                        if O_acc_idx == O_idx:
                            continue
                        OO_vector = atoms.positions[O_acc_idx] - atoms.positions[O_idx]
                        d_OO = np.linalg.norm(OO_vector)
                        if d_OO < 3.5:
                            DHA_angle = np.arccos(np.dot(OH_vector, OO_vector) / (np.linalg.norm(OH_vector) * np.linalg.norm(OO_vector))) * 180 / np.pi
                            if DHA_angle > 120:
                                is_free = False
                                break
                    if is_free:
                        angles.append(angle_wrt_z)
                else:
                    angles.append(angle_wrt_z) 
    return angles

def adf(ABC_all_angles, dtheta=1.0):
    '''
    Calculate the angular distribution function (ADF) for two atom types A and B

    Input:
        ABC_all_angles (list): list of angles between atom type A, B, and C
        dtheta (float):  bin size
    
    Return:
        theta (np.array): angle values
        ntheta (np.array): ADF values
    '''
    num_bins = int(180 / dtheta) 
    bins = np.linspace(0, 180, num_bins + 1)

    ntheta, theta = np.histogram(ABC_all_angles, bins=bins)
    theta = 0.5 * (theta[1:] + theta[:-1]) # theta is the center of the bin

    return theta, ntheta

def mean_adf_OH(samples, r_max=1.0,  subNum=79, firstTwo=False, mic=True, onlyAngle=False, onlyFree=True, zThresholdO=4.85, aboveZthres=None):
    '''
    Calculate the mean OH ADF for a list of samples

    Return:
        list or tuple of theta and ntheta:
        List of all angles if onlyAngle is True, otherwise (theta, ntheta).
    '''
    OH_all_angles = []
    rhos = []
    refNums = []
    for sample in tqdm(samples):
        # Prepare the sample atoms with PBC cell 
        atoms = read_xyz_with_atomic_numbers(sample)
        if mic:
            substrate = atoms[atoms.numbers == subNum]
            lattice_vectors = calculate_lattice_vectors(substrate)
            atoms.set_pbc([True, True, True])
            atoms.set_cell(lattice_vectors)
        else:
            x_min, x_max = min(atoms.positions[:,0]), max(atoms.positions[:,0])
            y_min, y_max  = min(atoms.positions[:,1]), max(atoms.positions[:,1])
            z_min, z_max  = min(atoms.positions[:,2]), max(atoms.positions[:,2])
            a, b, c = x_max-x_min, y_max-y_min, z_max-z_min
            origin = np.array([x_min, y_min, z_min])
            atoms.positions -= origin # Shift the atoms to the origin
            lattice_vectors = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
            atoms.set_pbc([False, False, False])
            atoms.set_cell(lattice_vectors)

        # Calculate angles between atom type A, B, and C. 
        angles = cal_angles_OH(atoms, r_max=r_max, mic=mic, zThresholdO=zThresholdO, onlyFree=onlyFree, aboveZthres=aboveZthres)
        OH_all_angles.extend(angles)
    if onlyAngle:
        return np.array(OH_all_angles)
    else:
        theta, ntheta = adf(OH_all_angles)
        return theta, ntheta 

def mean_adf(samples, A, B, C, r_max=10.0,  subNum=79, firstTwo=False, mic=True, onlyAngle=False, zThresholdB=4.85, aboveZthres=None):
    '''
    Calculate the mean ADF for a list of samples

    Input:
        samples: list of sample paths
        A: str, atom type A
        B: str, atom type B
        C: str, atom type C
        r_max: float, maximum distance to search for neighbors
        firstTwo: bool, whether to consider only the first two neighbors
        subNum: int, atomic number of the substrate
        mic: bool, whether to consider PBC
        onlyAngle: bool, whether to return only angles
        zThresholdB: float, z threshold of atom B (related to the Au surface) to separate the first and second layers water molecules
        aboveZthres: None: select all; True: select above the z threshold; False: select below the z threshold

    Return:
        list or tuple of theta and ntheta:
        List of all angles if onlyAngle is True, otherwise (theta, ntheta).
    '''
    ABC_all_angles = []
    rhos = []
    refNums = []
    for sample in tqdm(samples):
        # Prepare the sample atoms with PBC cell 
        atoms = read_xyz_with_atomic_numbers(sample)
        if mic:
            substrate = atoms[atoms.numbers == subNum]
            lattice_vectors = calculate_lattice_vectors(substrate)
            atoms.set_pbc([True, True, True])
            atoms.set_cell(lattice_vectors)
        else:
            x_min, x_max = min(atoms.positions[:,0]), max(atoms.positions[:,0])
            y_min, y_max  = min(atoms.positions[:,1]), max(atoms.positions[:,1])
            z_min, z_max  = min(atoms.positions[:,2]), max(atoms.positions[:,2])
            a, b, c = x_max-x_min, y_max-y_min, z_max-z_min
            origin = np.array([x_min, y_min, z_min])
            atoms.positions -= origin # Shift the atoms to the origin
            lattice_vectors = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
            atoms.set_pbc([False, False, False])
            atoms.set_cell(lattice_vectors)

        # Calculate angles between atom type A, B, and C. 
        angles = cal_angles(atoms, A, B, C, r_max=r_max, firstTwo=firstTwo, mic=mic, zThresholdB=zThresholdB, aboveZthres=aboveZthres)
        ABC_all_angles.extend(angles)
    if onlyAngle:
        return np.array(ABC_all_angles)
    else:
        theta, ntheta = adf(ABC_all_angles)
        return theta, ntheta 

def distance_dis(AB_all_distances, dr=0.1, r_max=4):
    '''
    Calculate the distance distribution for two atom types A and B
    input:
        AB_all_distances: np.array, all distances between atom type A and B
        dr: float, bin size
        r_max: float, maximum distance to consider
    return:
        r: np.array, distance values
        nr: np.array, distance distribution values
    '''
    # Extract positions of atom type A and B
    num_bins = int(r_max / dr) 
    bins = np.linspace(0, r_max, num_bins + 1)

    # Calculate  distance distribution
    nr, r = np.histogram(AB_all_distances, bins=bins, range=(0, r_max))
    r = 0.5 * (r[1:] + r[:-1]) # r is the center of the bin

    return r, nr 

def mean_distance_distribution(samples, A, B, subNum=79, dr=0.1, r_max=4, mic=True, onlyDistances=False):
    '''
    Calculate the mean ADF for a list of samples
    '''
    AB_all_distances = []
    refNums = []
    for sample in tqdm(samples):
        # Prepare the sample atoms with PBC cell 
        atoms = read_xyz_with_atomic_numbers(sample)
        substrate = atoms[atoms.numbers == subNum]
        lattice_vectors = calculate_lattice_vectors(substrate)
        atoms.set_pbc([True, True, True])
        atoms.set_cell(lattice_vectors)

        # Calculate distances between atom type A and B. 
        distances = cal_distances(atoms, A, B, r_max=r_max, mic=True, onlyDistances=True)

        # Collect all distances
        AB_all_distances.extend(distances)
    if onlyDistances:
        return AB_all_distances
    else:
        AB_all_distances = np.concatenate([arr.flatten() for arr in AB_all_distances])
        r, nr = distance_dis(AB_all_distances, dr=dr, r_max=r_max)
        return r, nr

def rdf(AB_all_distances, meanRho, refNum, r_max=10.0, bins=120):
    '''
    Calculate the radial distribution function for two atom types A and B
    input:
        AB_all_distances: list of float, all distances between atom type A and B
        meanRho: float, mean density of atom type B
        refNum: int, reference number of atom type A
        r_max: float, maximum distance to calculate RDF
        bins: int, bin number 
    return:
        r: np.array, distance values
        g: np.array, RDF values
    '''
    # Calculate RDF
    nr, r = np.histogram(AB_all_distances, bins=bins, range=(0, r_max))
    r_center = 0.5 * (r[1:] + r[:-1]) # r is the center of the bin
    dr = r[1] - r[0]
    gr = nr / (meanRho*4*np.pi*r_center**2*dr) / refNum  # Todo   
    return r, gr

def mean_rdf(samples, A, B, r_max=10.0, bins=120, mic=True, subNum=79, zThresholdO=4.85, aboveZthres=None, onlyDistances=False):
    '''
    Calculate the mean RDF for a list of samples.
    input:
        samples: list of sample paths
        A: str, atom type A
        B: str, atom type B
        r_max: float, maximum distance to calculate RDF
        bins: int, bin number
        mic: bool, whether to consider PBC
        subNum: int, atomic number of the substrate
        zThresholdO: float, z threshold of O atom (related to the Au surface) to separate the first and second layers water molecules
        aboveZthres: None: select all; True: select above the z threshold; False: select below the z threshold
        onlyDistances: bool, whether to return only distances
    return:
        r: np.array, distance values
        gr: np.array, RDF values
    '''
    AB_all_distances = []
    rhos = []
    refNums = []
    for sample in tqdm(samples):
        # Prepare the sample atoms with PBC cell 
        atoms = read_xyz_with_atomic_numbers(sample)
        if mic:
            substrate = atoms[atoms.numbers == subNum]
            lattice_vectors = calculate_lattice_vectors(substrate)
            atoms.set_pbc([True, True, True])
            atoms.set_cell(lattice_vectors)
        else:
            x_min, x_max = min(atoms.positions[:,0]), max(atoms.positions[:,0])
            y_min, y_max  = min(atoms.positions[:,1]), max(atoms.positions[:,1])
            z_min, z_max  = min(atoms.positions[:,2]), max(atoms.positions[:,2])
            a, b, c = x_max-x_min, y_max-y_min, z_max-z_min
            origin = np.array([x_min, y_min, z_min])
            atoms.positions -= origin # Shift the atoms to the origin
            lattice_vectors = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
            atoms.set_pbc([False, False, False])
            atoms.set_cell(lattice_vectors)

        # Calculate distances between atom type A and B
        AB_all_distance, rho,  refNum = cal_distances(atoms, A, B, r_max=r_max, mic=mic, onlyDistances=False, zThresholdO=zThresholdO, aboveZthres=aboveZthres)

        # Collect all distances
        AB_all_distances.extend(AB_all_distance)
        rhos.append(rho)
        refNums.append(refNum)
    if onlyDistances:
        return AB_all_distances
    else:
        meanRho = np.mean(rhos)
        refNum = np.sum(refNums) 
        r, gr = rdf(AB_all_distances, meanRho, refNum, r_max=r_max, bins=bins)
        return r, gr

def all_distances(samples, A, B, r_max=10.0, bins=120, mic=True, subNum=79):
    '''
    Obtain all distances between atom type A and B for a list of samples
    input:
        samples: list of sample paths
        A: str, atom type A
        B: str, atom type B
        r_max: float, maximum distance to calculate RDF
        bins: int, bin number
        mic: bool, whether to consider PBC
        subNum: int, atomic number of the substrate
    return:
        AB_all_distances: list of float, all distances between atom type A and B
    '''
    AB_all_distances = []
    rhos = []
    refNums = []
    for sample in tqdm(samples):
        # Prepare the sample atoms with PBC cell 
        atoms = read_xyz_with_atomic_numbers(sample)
        if mic:
            substrate = atoms[atoms.numbers == subNum]
            lattice_vectors = calculate_lattice_vectors(substrate)
            atoms.set_pbc([True, True, True])
            atoms.set_cell(lattice_vectors)
        else:
            x_min, x_max = min(atoms.positions[:,0]), max(atoms.positions[:,0])
            y_min, y_max  = min(atoms.positions[:,1]), max(atoms.positions[:,1])
            z_min, z_max  = min(atoms.positions[:,2]), max(atoms.positions[:,2])
            a, b, c = x_max-x_min, y_max-y_min, z_max-z_min
            origin = np.array([x_min, y_min, z_min])
            atoms.positions -= origin # Shift the atoms to the origin
            lattice_vectors = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
            atoms.set_pbc([False, False, False])
            atoms.set_cell(lattice_vectors)

        # Calculate distances between atom type A and B
        AB_all_distance = cal_distances(atoms, A, B, r_max=r_max, mic=mic, onlyDistances=True)

        # Collect all distances
        AB_all_distances.extend(AB_all_distance)
    return  AB_all_distances


def cal_d5(atoms, r_max=4.5, zThresholdO=4.85, aboveZthres=None):
    '''
    Calculate the 5th nearest neighbor distance of O atoms
    '''
    symbols = atoms.get_chemical_symbols()
    # Find all the O atoms
    if aboveZthres is not None:
        # Calculate the mean value of z positions of Au atoms 
        Au_indices = [i for i, s in enumerate(symbols) if s == 'Au']
        Au_positions = atoms.positions[Au_indices]
        z_mean = np.max(Au_positions[:, 2])
        if aboveZthres == True:
            O_indices = [i for i, s in enumerate(symbols) if s == 'O' and atoms.positions[i][2] - z_mean > zThresholdO]
        elif aboveZthres == False:
            O_indices = [i for i, s in enumerate(symbols) if s == 'O' and atoms.positions[i][2] - z_mean <= zThresholdO]
    else:
        O_indices = [i for i, s in enumerate(symbols) if s == 'O'] 
    # Loop through all the O atoms
    d5s = []
    for O_idx in O_indices:
        # Get all the other indices of O atoms
        O_indices_ = [i for i in O_indices if i != O_idx]
        # Get the distances between the central O atom and all the other O atoms
        distances = atoms.get_distances(O_idx, O_indices_)
        # Sort the distances from smallest to largest
        distances = sorted(distances)
        if len(distances) >= 5 and distances[4] <= r_max:
            d5 = distances[4]
            d5s.append(d5)
    return d5s

def cal_d5_all(samples, r_max=4.5, zThresholdO=4.85, aboveZthres=None):
    '''
    Calculate the 5th nearest neighbor distance of O atoms for all samples
    '''
    d5s = []
    for sample in tqdm(samples):
        atoms = read_xyz_with_atomic_numbers(sample)
        d5 = cal_d5(atoms, r_max=r_max, zThresholdO=zThresholdO, aboveZthres=aboveZthres)
        d5s.extend(d5)
    return np.array(d5s)

def compute_lsi(atoms, r_max=4.5, zThresholdO=4.85, aboveZthres=None):
    """
    Compute the Local Structure Index (LSI) for oxygen atoms.
    
    Parameters:
        atoms: ASE Atoms object
        r_max: float, maximum neighbor search radius (default: 4.5 Å)
    
    Returns:
        lsi_values: list of LSI values for each oxygen atom
    """
    symbols = atoms.get_chemical_symbols()
    # Find all the O atoms
    if aboveZthres is not None:
        # Calculate the mean value of z positions of Au atoms 
        Au_indices = [i for i, s in enumerate(symbols) if s == 'Au']
        Au_positions = atoms.positions[Au_indices]
        z_mean = np.max(Au_positions[:, 2])
        if aboveZthres == True:
            O_indices = [i for i, s in enumerate(symbols) if s == 'O' and atoms.positions[i][2] - z_mean > zThresholdO]
        elif aboveZthres == False:
            O_indices = [i for i, s in enumerate(symbols) if s == 'O' and atoms.positions[i][2] - z_mean <= zThresholdO]
    else:
        O_indices = [i for i, s in enumerate(symbols) if s == 'O'] 

    lsi_values = []
    for O_idx in O_indices:
        O_neighbors = [i for i in O_indices if i != O_idx]
        distances = atoms.get_distances(O_idx, O_neighbors, mic=True)

        # Select distances less than r_max
        distances = np.array([d for d in distances if d <= r_max])

        # Sort distances and find the index n(i) where r_n < 3.7 Å and r_n+1 > 3.7 Å
        sorted_distances = np.sort(distances)
        n_i = np.searchsorted(sorted_distances, 3.7)
        
        if n_i > 1:
            delta = np.diff(sorted_distances[:n_i])  # Compute Δ(j; i)
            delta_mean = np.mean(delta)
            lsi = np.mean((delta - delta_mean) ** 2)
            lsi_values.append(lsi)
    
    return lsi_values

def compute_lsi_all(samples, r_max=4.5, zThresholdO=4.85, aboveZthres=None):
    """
    Compute LSI for all oxygen atoms in multiple samples.
    
    Parameters:
        samples: list of file paths to XYZ structures
        r_max: float, neighbor search radius
    
    Returns:
        np.array of all computed LSI values
    """
    lsi_all = []
    for sample in tqdm(samples):
        atoms = read_xyz_with_atomic_numbers(sample)
        lsi_values = compute_lsi(atoms, r_max=r_max, zThresholdO=zThresholdO, aboveZthres=aboveZthres)
        lsi_all.extend(lsi_values)
    return np.array(lsi_all)

def compute_sk_sg(atoms, r_max=3.5, zThresholdO=4.85, aboveZthres=None):
    """
    Compute both tetrahedral order parameter S_g and distance-based order parameter S_k for oxygen atoms.
    
    Parameters:
        atoms: ASE Atoms object
        r_max: float, maximum neighbor search radius (default: 3.7 Å)
    
    Returns:
        sg_values: list of S_g values for each oxygen atom
        sk_values: list of S_k values for each oxygen atom
    """
    symbols = atoms.get_chemical_symbols()
    
    if aboveZthres is not None:
        Au_indices = [i for i, s in enumerate(symbols) if s == 'Au']
        Au_positions = atoms.positions[Au_indices]
        z_mean = np.max(Au_positions[:, 2])

    O_indices = [i for i, s in enumerate(symbols) if s == 'O'] 
    sg_values = []
    sk_values = []
    
    for O_idx in O_indices:
        O_neighbors = [i for i in O_indices if i != O_idx]
        distances = atoms.get_distances(O_idx, O_neighbors, mic=True)
        
        if len(distances) >= 4:
            nearest_indices = np.argsort(distances)[:4]
            neighbors = [O_neighbors[i] for i in nearest_indices]
            distances4 = [distances[i] for i in nearest_indices]
            
            if np.max(distances4) <= r_max:

                # Compute unit bond vectors
                r_j = [(atoms.positions[j] - atoms.positions[O_idx]) / np.linalg.norm(atoms.positions[j] - atoms.positions[O_idx]) for j in neighbors]
                
                # Compute S_g using the formula
                sg = 0
                for j in range(3):
                    for k in range(j + 1, 4):
                        cos_theta = np.dot(r_j[j], r_j[k])
                        sg += (cos_theta + 1/3) ** 2
                sg *= 3 / 8 
                
                # Compute S_k
                r_k = np.array(distances4)
                r_mean = np.mean(r_k)
                sk = (1/3) * np.sum((r_k - r_mean) ** 2 / (4 * r_mean ** 2))

                if aboveZthres is not None:
                    if aboveZthres is True: 
                        if atoms.positions[O_idx][2] - z_mean > zThresholdO:
                            sg_values.append(sg)
                            sk_values.append(sk)
                    elif aboveZthres is False:
                        if atoms.positions[O_idx][2] - z_mean <= zThresholdO:
                            sg_values.append(sg)
                            sk_values.append(sk)
                    else:
                        pass    
                else:
                    # All 
                    sg_values.append(sg)
                    sk_values.append(sk)
    
    return sk_values, sg_values

def compute_sk_sg_all(samples, r_max=3.5, zThresholdO=4.85, aboveZthres=None):
    """
    Compute S_g and S_k for all oxygen atoms in multiple samples.
    
    Parameters:
        samples: list of file paths to XYZ structures
        r_max: float, neighbor search radius
    
    Returns:
        np.array of all computed S_g values
        np.array of all computed S_k values
    """
    sg_all = []
    sk_all = []
    for sample in tqdm(samples):
        atoms = read_xyz_with_atomic_numbers(sample)
        sk_values, sg_values = compute_sk_sg(atoms, r_max=r_max, zThresholdO=zThresholdO, aboveZthres=aboveZthres)
        sg_all.extend(sg_values)
        sk_all.extend(sk_values)
    return np.array(sk_all), np.array(sg_all)

def plot_distance_distribution(distances, label, legend, r_max=10,  color='#299035', bins=120, y_lim=0.4, outfolder='output'):
    figure_size=(6, 2.5)
    plt.figure(figsize=figure_size)
    plt.tick_params(direction="in", axis='both', top=True)
    plt.hist(distances, density=True, range=(0, r_max), color = color, bins=bins, alpha=0.5, label=legend)
    plt.hist(distances, histtype='step', fill=False,  density=True, range=(0, r_max), color = color, bins=bins, alpha=1)
    plt.xlim(0, r_max)
    plt.ylim(0, y_lim)
    plt.xlabel(r"Distance [$\AA$]")
    plt.ylabel('Probability Density')
    plt.tight_layout()
    plt.legend(frameon=False, loc='upper right')
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    plt.savefig('{}/{}.png'.format(outfolder, label), dpi=300)
    plt.savefig('{}/{}.pdf'.format(outfolder, label))
    plt.clf()
    plt.close()

def plot_angle_distribution(angles, label, legend, color='#299035', bins=120, y_lim=0.4, outfolder='output', style='bar', loc='upper left', figure_size=(6, 2.5) , show=False):
    def plot_one(angles, color, legend, style='bar'):
        if style == 'bar':
          plt.hist(angles, bins=bins, density=True, range=(0, 180), color=color, alpha=0.5, label=legend)
          n, _, _ = plt.hist(angles, bins=bins, histtype='step', fill=False, density=True, range=(0, 180), color=color, alpha=1)
        elif style == 'step':
          n, _, _ = plt.hist(angles, bins=bins, histtype='step', fill=False, density=True, range=(0, 180), color=color, alpha=1, label=legend)
        else:
            raise ValueError("Invalid mode")
        return n

    figure_size=figure_size
    plt.figure(figsize=figure_size)
    plt.tick_params(direction="in", axis='both', top=True)
    if type(angles) == list:
        ns = []
        # Check if the style, and color  are the same as the length of angles
        if len(angles) != len(style) or len(angles) != len(color):
            raise ValueError("The length of angles, legend, and color should be the same.")
        for i, a in enumerate(angles):
            n = plot_one(a, color[i], legend[i], style[i])
            ns.append(n)
    else:
       n = plot_one(angles, color, legend, style)
    plt.xlim(0, 180)
    plt.ylim(0, y_lim)
    plt.xlabel(r"$\theta$ (degree)")
    plt.ylabel('Probability Density')
    plt.tight_layout()
    plt.legend(frameon=False, loc=loc)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    plt.savefig('{}/{}.png'.format(outfolder, label), dpi=300)
    plt.savefig('{}/{}.pdf'.format(outfolder, label))
    plt.savefig('{}/{}.svg'.format(outfolder, label))
    if show:
        plt.show()
    plt.clf()
    plt.close()

    if type(angles) == list:
        return ns
    else:
        return n

def plot_rdf(r, gr, label, legend, color='#299035', x_lim=10, y_lim=10, outfolder='output', style='bar', loc="upper right", figure_size=(6, 2.5), show=False):
    plt.figure(figsize=figure_size)
    plt.tick_params(direction="in", axis='both', top=True)
    def plot_one(r, gr, color, legend, style='bar'):
        if style == 'bar':
            plt.bar(r[:-1], gr, width=r[1]-r[0], alpha=0.5, edgecolor=color, align="edge", color=color, label=legend)
        elif style == 'line':
            plt.plot(r[:-1] + (r[1] - r[0]) / 2, gr, label=legend, color=color)
        elif style == 'step':
            plt.step(np.append(r[:-1], r[-1]), np.append(gr, gr[-1]), color=color, linestyle='-', lw=1, alpha=1, label=legend)
        else:
            raise ValueError("Invalid mode")
    # If gr is a list, loop over it
    if type(gr) == list:
        # Check if the style, and color  are the same as the length of gr
        if len(gr) != len(style) or len(gr) != len(color):
            raise ValueError("The length of gr, legend, and color should be the same.")
        for i, g in enumerate(gr):
            plot_one(r, g, color[i], legend[i], style[i])
    else:
        plot_one(r, gr, color, legend, style)
    plt.xlim(0, x_lim)
    plt.ylim(0, y_lim)
    plt.xlabel(r'r (Å)')
    plt.ylabel('g(r)')
    plt.tight_layout()
    plt.legend(frameon=False, loc=loc)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    plt.savefig('{}/{}.png'.format(outfolder, label), dpi=300)
    plt.savefig('{}/{}.pdf'.format(outfolder, label))
    plt.savefig('{}/{}.svg'.format(outfolder, label))
    if show:
        plt.show()
    plt.clf()
    plt.close()

def plot_hbond_distance_vs_angle_bak(hydrogen_bonds, angle_type='dha', label='hb-d-angle',  cmap='Greens', outfolder='output'):
    distances = [hb[3] for hb in hydrogen_bonds]
    angles = [hb[4] for hb in hydrogen_bonds] if angle_type == 'dha' else [hb[5] for hb in hydrogen_bonds]

    # Generate a 2D density estimate
    xy = np.vstack([distances, angles])
    kde = gaussian_kde(xy)
    # Generate a grid to evaluate the KDE over
    xmin, xmax = 1.0, 3.5
    (ymin, ymax) = (120, 180) if angle_type == 'dha' else (0, 60)    
    xgrid = np.linspace(xmin, xmax, 300)
    ygrid = np.linspace(ymin, ymax, 300)
    X, Y = np.meshgrid(xgrid, ygrid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)  # Evaluate the KDE on the grid

    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, Z, cmap=cmap, vmin=0, vmax=0.3, levels=30)
    plt.colorbar(contour, label=r'$\rho(d_\text{OO}, \theta)$')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # Labels and title
    plt.xlabel("$d_{OO}$ (Å)")
    plt.ylabel(r"$\angle$DHA (degree)") if angle_type == 'dha' else plt.ylabel(r"$\angle$HDA (degree)") 
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(outfolder, label), dpi=300)
    plt.savefig('{}/{}.pdf'.format(outfolder, label))
    plt.show()
    return X, Y, Z

def plot_hbond_distance_vs_angle(hydrogen_bonds, angle_type='dha', label='hb-d-angle', cmap='Greens', outfolder='output', figsize=(3, 3), nbin=100, vmax=0.22, show=False):
    # Extract distances and angles
    distances = [hb[3] for hb in hydrogen_bonds]
    angles = [hb[4] for hb in hydrogen_bonds] if angle_type == 'dha' else [hb[5] for hb in hydrogen_bonds]

    # Define plot range
    xmin, xmax = 1.0, 3.5
    ymin, ymax = (120, 180) if angle_type == 'dha' else (0, 60)
    xgrid = np.linspace(xmin, xmax, nbin)
    ygrid = np.linspace(ymin, ymax, nbin)
    X, Y = np.meshgrid(xgrid, ygrid)

    plt.figure(figsize=figsize)
    plt.tick_params(direction="in", axis='both', top=True, right=True)
    # Generate a 2D density estimate
    xy = np.vstack([distances, angles])
    kde = gaussian_kde(xy)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)  # Evaluate the KDE on the grid
    # Plot the density estimate
    contour = plt.pcolormesh(X, Y, Z, cmap=cmap, vmin=0, vmax=vmax)
    plt.colorbar(contour, label=r'$\rho(d, \theta)$')
    # Set plot limits and labels
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("$d_{OO}$ (Å)")
    plt.ylabel(r"$\angle$DHA (degrees)" if angle_type == 'dha' else r"$\angle$HDA (degree)")
    plt.tight_layout()
    plt.savefig(f'{outfolder}/{label}.png', dpi=300)
    plt.savefig(f'{outfolder}/{label}.pdf')
    plt.savefig(f'{outfolder}/{label}.svg')
    if show:
        plt.show()
    plt.close()

    return X, Y, Z

def plot_density_difference(X, Y, Z, angle_type='dha', cmap='coolwarm', label='density-difference', outfolder='output', show=False):
    contour = plt.pcolormesh(X, Y, Z, cmap=cmap, vmin=-0.05, vmax=0.05)
    plt.colorbar(contour, label=r'Probability Density Difference')
    plt.xlabel("$d_{OO}$ (Å)")
    plt.ylabel(r"$\theta$ (degree)") if angle_type == 'dha' else plt.ylabel(r"$\phi$ (degree)") 
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(outfolder, label), dpi=300)
    if show:
        plt.show()
    plt.close()


def compute_kde_wasserstein(data1, data2, bw_method=None, num_points=1000):
    """
    Compute KDE-smoothed Wasserstein Distance between two datasets.

    Parameters:
    - data1: array-like
        First dataset.
    - data2: array-like
        Second dataset.
    - bw_method: float or None
        Bandwidth method for KDE (e.g., 0.5 for smaller bandwidth, 2 for larger bandwidth).
        If None, the default bandwidth is used.
    - num_points: int
        Number of points for KDE evaluation.

    Returns:
    - wdistance_smoothed: float
        Wasserstein distance between the smoothed PDFs.
    """
    # Perform KDE smoothing
    kde1 = gaussian_kde(data1, bw_method=bw_method)
    kde2 = gaussian_kde(data2, bw_method=bw_method)

    # Create a common range for comparison
    x_range = np.linspace(min(min(data1), min(data2)) - 1,
                          max(max(data1), max(data2)) + 1, num_points)

    # Evaluate KDE PDFs
    pdf1 = kde1(x_range)
    pdf2 = kde2(x_range)

    # Compute Wasserstein distance
    wdistance_smoothed = wasserstein_distance(x_range, x_range, u_weights=pdf1, v_weights=pdf2)

    return wdistance_smoothed



def sinkhorn_2d_distance(data1, data2, p=1, blur=0.0001, bw_method=None, grid_size=100):
    """
    For blur is small enough, the Sinkhorn distance is the Eurth Mover's distance 
    Ref: 
    1. SINKHORN DISTANCES: LIGHTSPEED COMPUTATION OF  OPTIMAL TRANSPORTATION DISTANCES
    2. https://dfdazac.github.io/sinkhorn.html 
    """
    # Convert data to PyTorch tensors
    x = torch.tensor(data1, dtype=torch.float32)
    y = torch.tensor(data2, dtype=torch.float32)

    loss = SamplesLoss("sinkhorn", p=p, blur=blur)
    distance = loss(x, y)
    
    return distance.item()

def kde(data1, data2, bw_method=None, grid_size=100):
    # Perform KDE smoothing for plotting
    kde1 = gaussian_kde(data1.T, bw_method=bw_method)  # Transpose required for gaussian_kde
    kde2 = gaussian_kde(data2.T, bw_method=bw_method)

    # Create a common grid for comparison
    minX = min(data1[:, 0].min(), data2[:, 0].min())
    maxX = max(data1[:, 0].max(), data2[:, 0].max())
    minY = min(data1[:, 1].min(), data2[:, 1].min())
    maxY = max(data1[:, 1].max(), data2[:, 1].max())
    x_range = np.linspace(minX, maxX, grid_size, endpoint=True)
    y_range = np.linspace(minY, maxY, grid_size, endpoint=True)
    X, Y = np.meshgrid(x_range, y_range)
    grid_coords = np.vstack([X.ravel(), Y.ravel()])

    # Evaluate KDE PDFs
    pdf1 = kde1(grid_coords).reshape(grid_size, grid_size)
    pdf2 = kde2(grid_coords).reshape(grid_size, grid_size)

    return x_range, y_range, pdf1, pdf2

def SinkhornDistance(samples1, samples2, eps=0.1, max_iter=100, tol=1e-6):
    """
    Note: This self-implemented function isn't numerical stable yet. I tried to make it work for 1D, but for some samples, 0 could be found at the denominator.
    Then I use the Wasserstein distance function in Scipy and use Sinkhorn distance with small weight loss for the entropy part for the 2D samples.

    Compute the Sinkhorn distance between two distributions and return the distance, transport plan, and cost matrix.
    Ref: Sinkhorn distances: lightspeed computation of optimal transport
    Args:
        samples1 (torch.Tensor): First distribution samples (n x d).
        samples2 (torch.Tensor): Second distribution samples (m x d).
        epsilon (float): Regularization parameter.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        dist (torch.Tensor): Sinkhorn distance.
        transport_plan (torch.Tensor): Optimal transport plan matrix (n x m).
        cost_matrix (torch.Tensor): Cost matrix (n x m).
    """
    # If samples are not PyTorch tensors, convert them
    samples1 = torch.tensor(samples1, dtype=torch.float)
    samples2 = torch.tensor(samples2, dtype=torch.float)

    # If samples are 1D, convert them to 2D
    if samples1.dim() == 1:
        samples1 = samples1.unsqueeze(1)
    if samples2.dim() == 1:
        samples2 = samples2.unsqueeze(1)

    if torch.isnan(samples1).any() or torch.isnan(samples2).any():
        raise ValueError("Input samples contain NaN.")
    if torch.isinf(samples1).any() or torch.isinf(samples2).any():
        raise ValueError("Input samples contain infinity.")

    # Compute cost matrix (pairwise squared Euclidean distances)
    cost_matrix = torch.cdist(samples1, samples2, p=2) ** 2  # Cost matrix (n x m)
    #print('Cost_matrix', cost_matrix)
    # Initialize uniform weights for the two distributions
    n, m = cost_matrix.shape
    a = torch.ones(n, dtype=torch.float) / n  # Uniform weights for samples1
    b = torch.ones(m, dtype=torch.float) / m  # Uniform weights for samples2

    # Initialize u and v (scaling factors for Sinkhorn-Knopp)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    # Kernel matrix (regularized by epsilon)
    K = torch.exp(-cost_matrix / eps)

    # Sinkhorn iterations
    for _ in range(max_iter):
        u_prev = u.clone()
        u = a / (K @ v)
        v = b / (K.T @ u)
        if torch.isnan(u).any() or torch.isnan(v).any():
            print("NaN detected in scaling factors.")
            break
        # Check for convergence
        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    # Compute transport plan
    transport_plan = torch.diag(u) @ K @ torch.diag(v)

    # Compute Sinkhorn distance
    dist = torch.sum(transport_plan * cost_matrix)

    # Return number and np.array of the ransport plan, and cost matrix
    dist, transport_plan, cost_matrix = dist.item(), transport_plan.detach().numpy(), cost_matrix.detach().numpy()
    # Check if distance is NaN
    if np.isnan(dist):
        print("Sinkhorn distance is NaN.")
    return dist, transport_plan, cost_matrix


def energy_distance(data1, data2):
    """
    Compute the Energy Distance between two datasets.

    Supports:
        - 1D arrays: shape (N,)
        - 2D arrays: shape (N, D)

    Parameters:
        data1: numpy array of shape (N,) or (N, D)
        data2: numpy array of shape (M,) or (M, D)

    Returns:
        A float representing the Energy Distance.
    """
    # Convert 1D to 2D if needed
    if data1.ndim == 1:
        data1 = data1[:, np.newaxis]
    if data2.ndim == 1:
        data2 = data2[:, np.newaxis]

    n, m = len(data1), len(data2)

    # Pairwise distances
    d1d2 = cdist(data1, data2, metric='euclidean')   # Cross terms
    d1d1 = cdist(data1, data1, metric='euclidean')   # Within data1
    d2d2 = cdist(data2, data2, metric='euclidean')   # Within data2

    # Compute energy distance using the definition
    term_1 = 2 * np.sum(d1d2) / (n * m)
    term_2 = np.sum(d1d1) / (n * n)
    term_3 = np.sum(d2d2) / (m * m)

    return term_1 - term_2 - term_3

def mmd_rbf_distance(data1, data2, gamma=None):
    """
    Compute Maximum Mean Discrepancy (MMD) with RBF kernel.

    Parameters:
        data1, data2 : np.ndarray
            Input sample arrays. Shape (n,) or (n,d)
        gamma : float or None
            Kernel width parameter. If None, use median heuristic.

    Returns:
        float : MMD distance value
    """
    # Convert 1D arrays to 2D (n_samples, 1)
    if data1.ndim == 1:
        data1 = data1[:, np.newaxis]
    if data2.ndim == 1:
        data2 = data2[:, np.newaxis]

    # Use median heuristic for gamma if not provided
    if gamma is None:
        combined = np.vstack([data1, data2])
        pairwise_dists = cdist(combined, combined, 'euclidean')
        median = np.median(pairwise_dists)
        gamma = 1 / (2 * median**2 + 1e-10)  # add small epsilon to avoid div 0

    # Compute RBF kernel matrices
    Kxx = rbf_kernel(data1, data1, gamma=gamma)
    Kyy = rbf_kernel(data2, data2, gamma=gamma)
    Kxy = rbf_kernel(data1, data2, gamma=gamma)

    # Calculate MMD^2
    mmd2 = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return np.sqrt(mmd2)

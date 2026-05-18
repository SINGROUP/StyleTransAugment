#!/usr/bin/env python3
from ase import Atoms
from ase.visualize import view
import numpy as np
import sys 


def read_xyz_with_atomic_numbers(file_path, restrict_to_scan_xy=False):
    '''
    Read coordinates from an XYZ file and return an Atoms object.
    If restrict_to_scan_xy is True, only include atoms within the x-y scan window.
    '''
    atomic_numbers_to_symbols = {1: 'H', 6: 'C', 8: 'O', 29: 'Cu', 79: 'Au'}

    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_atoms = int(lines[0])
    comment = lines[1].strip()
    print("Comment line:", comment)

    # Parse scan window if restriction is enabled
    if restrict_to_scan_xy:
        import re
        match = re.search(r'Scan window: \[\[([^\]]+)\], \[([^\]]+)\]', comment)
        if not match:
            raise ValueError("Scan window not found in comment line.")
        xymin = list(map(float, match.group(1).split()))
        xymax = list(map(float, match.group(2).split()))
        xmin, ymin = xymin[0], xymin[1]
        xmax, ymax = xymax[0], xymax[1]
        print('xmin, ymin', xmin, ymin)
        print('xmax, ymax', xmax, ymax)

    symbols = []
    positions = []

    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        atomic_number = int(parts[0])
        x, y, z = map(float, parts[1:4])

        if restrict_to_scan_xy:
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue  # skip this atom

        symbol = atomic_numbers_to_symbols[atomic_number]
        symbols.append(symbol)
        positions.append([x, y, z])

    positions = np.array(positions)
    return Atoms(symbols=symbols, positions=positions)

if __name__ == '__main__':
    filePath = '0.xyz'
    restrict = True
    atoms = read_xyz_with_atomic_numbers(filePath, restrict_to_scan_xy=restrict)
    view(atoms)

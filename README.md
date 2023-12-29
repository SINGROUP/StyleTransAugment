# AFM-Augmentation
Enhancing AFM Image Analysis and Prediction through Machine Learning and Data Augmentation

This project relies on the [ASD-AFM-dev](https://github.com/SINGROUP/ASD-AFM-dev) data-augmentation branch. 

Usage:


1. Preparations
Clone this project:
```bash
git clone git@github.com:HuangJiaLian/AFM-Augmentation.git
```

Create and activate a conda environment:
```bash
# Conda environment installation
cd ASD-AFM-dev/
conda env create -f environment.yml
conda activate ml
git checkout data-augmentation
```

Change to a GPU node (My case)
```bash
srun -p gpushort --gres=gpu:4 --constraint=pascal  --time=4:00:00 --mem=6000M --pty bash
```

Compile extensions by running 
```
cp /etc/OpenCL/vendors/nvidia.icd $CONDA_PREFIX'/etc/OpenCL/vendors/'
./build.sh
```

2. Dataload demostration
```
python 1_dataload.py 
```
Several input-label pairs that look like the following would be stored in the folder temp. 
![](temp/input_label.png)

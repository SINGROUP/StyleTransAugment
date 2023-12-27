# AFM-Augmentation
Enhancing AFM Image Analysis and Prediction through Machine Learning and Data Augmentation

This project relies on the [ASD-AFM-dev](https://github.com/SINGROUP/ASD-AFM-dev) data-augmentation branch. 

Usage:

```bash
git clone git@github.com:HuangJiaLian/AFM-Augmentation.git
# Conda environment installation
cd ASD-AFM-dev/
conda env create -f environment.yml
conda activate ml
cp /etc/OpenCL/vendors/nvidia.icd $CONDA_PREFIX'/etc/OpenCL/vendors/'
# Compile extensions by running 
./build.sh
```

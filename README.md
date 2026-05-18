# Improving atomic force microscopy structure discovery via style-translation
This is the source code for our paper [arXiv:2509.02240](https://arxiv.org/pdf/2509.02240).

<p align='center'>
<img src="manuscript/TOC/TOC.png" width="50%"/>
</p>

## Abstract

Atomic force microscopy (AFM) is a key tool for characterising nanoscale structures, with functionalised tips now offering detailed images of the atomic structure. In parallel, AFM simulations using the particle probe model provide a cost-effective approach for rapid AFM image generation. Using state-of-the-art machine learning models and substantial simulated datasets, properties such as molecular structure, electrostatic potential, and molecular graph can be predicted from AFM images. However, transferring model performance from simulated to experimental AFM images poses challenges due to the subtle variations in real experimental data compared to the seemingly flawless simulations. In this study, we explore style translation to augment simulated images and improve the predictive performance of machine learning models in surface property analysis. We reduce the style gap between simulated and experimental AFM images and demonstrate the method's effectiveness in enhancing structure discovery models through local structural property distribution comparisons. This research presents a novel approach to improving the efficiency of machine learning models in the absence of labelled experimental data.

## Installation
We use `conda` to manage all the packages. You can create and activate it by using:
```bash
conda env create -f myenv.yml
conda activate sta
```

## Structure
There are three parts in `src`:
1. `preEvaluation`: Data-driven approach to evaluate the performance of style translation.
2. `performanceEvaluation`: Performance evaluations of the structure models on the experimental AFM images based on the local structural properties. 
3. `StyleTrans` (submodule): Using CycleGAN framework for training the style translation model to obtain the style translation models.

where the `snakemake` file is used to show the logic and run the code. 


## Data
The relevant data used for training can be found at [Zenodo](https://doi.org/10.5281/zenodo.16828078).

## Citation
Cited as:

> Jie Huang, Niko Oinonen, Fabio Priante, Filippo Federici Canova, Lauri Kurki, Chen Xu, and Adam
S. Foster*, Improving atomic force microscopy structure discovery via style-translation, arXiv:2509.02240, 2025

Or

```
@misc{huang2025sta,
      title={Improving atomic force microscopy structure discovery via style-translation}, 
      author={Jie Huang and Niko Oinonen and Fabio Priante and Filippo Federici Canova and Lauri Kurki and Chen Xu and Adam S. Foster},
      year={2025},
      eprint={2509.02240},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2509.02240}, 
}
```

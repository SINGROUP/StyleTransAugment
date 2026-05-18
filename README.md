# Improving atomic force microscopy structure discovery via style-translation

<p align='center'>
<img src="manuscript/TOC/TOC.png" width="70%"/>
</p>

Source codes for the paper [Improving atomic force microscopy structure discovery via style-translation](https://arxiv.org/pdf/2509.02240).

## Abstract

Atomic force microscopy (AFM) is a key tool for characterising nanoscale structures, with functionalised tips now offering detailed images of the atomic structure. In parallel, AFM simulations using the particle probe model provide a cost-effective approach for rapid AFM image generation. Using state-of-the-art machine learning models and substantial simulated datasets, properties such as molecular structure, electrostatic potential, and molecular graph can be predicted from AFM images. However, transferring model performance from simulated to experimental AFM images poses challenges due to the subtle variations in real experimental data compared to the seemingly flawless simulations. In this study, we explore style translation to augment simulated images and improve the predictive performance of machine learning models in surface property analysis. We reduce the style gap between simulated and experimental AFM images and demonstrate the method's effectiveness in enhancing structure discovery models through local structural property distribution comparisons. This research presents a novel approach to improving the efficiency of machine learning models in the absence of labelled experimental data.

## Structure

1. StyleTrans (submodule): Using CycleGAN framework for training the style translation model to obtain the style translation models.
2. preEvaluation: Data-driven approach to evaluate the performance of style translation.
3. structureDiscovery: Training the structure discovery model to predict the atomic structures from AFM images.
4. performanceEvaluation: Performance evaluations of the structure models on the experimental AFM images based on the local structural properties. 


## Installation

```bash
conda env create -f myenv.yml
conda activate sta
```

**Notes: Lots of the analysis code is still in this repository [StructureMetrics](https://github.com/huangchieh/StructureMetrics). We are organising all the codes and moving them here.**

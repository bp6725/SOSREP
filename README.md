# SOSREP - Sobolev Space Regularised Pre Density Model
This repository contains the implementation of the methods described in our paper titled "Sobolev Space Regularised Pre Density Models" (\cite{}). We propose a novel non-parametric density estimation approach that utilizes the regularization of a Sobolev norm of the density. Our method not only ensures statistical consistency but also enhances model interpretability by making the inductive bias explicit. 

## Highlights
- Statistical Consistency and Interpretability: Our method leverages the Sobolev space to provide a clear, interpretable framework for density estimation.
- Evaluated on the comprehensive anomaly detection benchmark suite, ADBench, our model demonstrates superior capabilities, ranking second among more than 15 competing algorithms (First in the challenging setup of duplicated anomalies). This strong performance underscores our model's potential and makes it a compelling choice for researchers and practitioners working on anomaly detection.
- Innovative Optimization Approach: Utilizes natural gradients to effectively navigate the challenges of non-convex optimization.
- Adaptation of the Fisher-Divergence for hyperparaneters tuning for unnormalized densities : Introduces an alternative evaluation method using Fisher divergence for situations where densities do not sum to one.
- Supports Gaussian, Laplace, and a new SDO kernels (Torch implementation). 


## Installation
## Installation
Instructions on setting up the environment and installing necessary dependencies.

```bash
git clone https://github.com/bp6725/SOSREP.git 
cd SOSREP 
pip install -r requirements.txt 
```


# Example usage
TODO

##Contact
For any additional questions or feedback, please contact Benny Perets at sbp67250@campus.technion.ac.il.

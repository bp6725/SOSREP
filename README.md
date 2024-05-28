# SOSREP - Sobolev Space Regularised Pre Density Model
This repository contains the implementation of the methods described in our paper titled "Sobolev Space Regularised Pre Density Models" (\cite{}). We propose a novel non-parametric density estimation approach that utilizes the regularization of a Sobolev norm of the density. Our method not only ensures statistical consistency but also enhances model interpretability by making the inductive bias explicit. 

## Highlights
- Statistical Consistency and Interpretability: Our method leverages the Sobolev space to provide a clear, interpretable framework for density estimation.
- Evaluated on the comprehensive anomaly detection benchmark suite, ADBench, our model demonstrates superior capabilities, ranking second among more than 15 competing algorithms (First in the challenging setup of duplicated anomalies). This strong performance underscores our model's potential and makes it a compelling choice for researchers and practitioners working on anomaly detection.
- Innovative Optimization Approach: Utilizes natural gradients to effectively navigate the challenges of non-convex optimization.
- Adaptation of the Fisher-Divergence for HP tuning for unnormalized densities : Introduces an alternative evaluation method using Fisher divergence for situations where densities do not sum to one.
- Torch implmantation for the Gaussian, Laplace and Single Derivative Order kernels. 


## Installation

bash
Copy code
git clone https://github.com/yourusername/sobolev-space-density-model.git
cd sobolev-space-density-model
pip install -r requirements.txt
Usage
Quick start guide or examples on how to use the implemented methods with your own data or with provided example datasets.

python
Copy code
# Example usage
from sobolev_density import SobolevDensityEstimator
model = SobolevDensityEstimator(params)
model.fit(data)
Citation
If you find this work useful, please consider citing:

mathematica
Copy code
@article{yourpaper2024,
  title={Sobolev Space Regularised Pre Density Models},
  author={Author, First and Others, And},
  journal={Journal of Amazing Density Estimations},
  year={2024},
  volume={XX},
  number={YY},
  pages={123-456},
  publisher={Famous Publisher}
}
License
Specify the license under which your code is made available (e.g., MIT, GPL).

Contact
For any additional questions or feedback, please contact [Your Name] at [Your Email].

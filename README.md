# Variational Entropy Search

This repo contains code for our paper "Variational Entropy Search for Adjusting Expected Improvement". 

## Referencing this code

If you use this code in any of your own work, please refer our paper: 
```
@misc{cheng2024variational,
  title={Variational Entropy Search for Adjusting Expected Improvement},
  author={Cheng, Nuojin and Becker, Stephen},
  year={2024},
  eprint={2402.11345},
  archivePrefix={arXiv},
  primaryClass={stat.ML}
}
```

## Description of code
### Demos for Test Functions

- **BO-1D.ipynb:** A 1D toy case comparing VES-Gamma with other functions;
- **BO-2D.ipynb:** Demonstration for implementing VES-Gamma on 3 different test functions;

### Demos for Read Datasets
- **iris.ipynb:** Hyper-parameter tuning for XGBoost on the iris dataset;
- **digits.ipynb:** Hyper-parameter tuning for XGBoost on the digits dataset;
- **wine.ipynb:** Hyper-parameter tuning for XGBoost on the wine dataset;
- **california_housing.ipynb:** Hyper-parameter tuning for XGBoost on the Carlifornia housing dataset;

### Support

- **ves.py:** Main file containing *VariationalEntropySearch* class and functions for running 1D and 2D experiments;

## Instruction for creating plots
Follow the provided demos notebooks. The generated figures from read datasets are also provided in the repo.

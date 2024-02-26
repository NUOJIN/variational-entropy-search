# Variational Entropy Search

This repo contains code for our paper "Variational Entropy Search for Adjusting Expected Improvement". 

## Referencing this code

If you use this code in any of your own work, please refer our paper: 
```
@misc{cheng2024variational,
  title={Variational Entropy Search for Adjusting Expected Improvement},
  author={Cheng, Nuojin and Becker, Stephen},
  journal={arXiv preprint arXiv:2402.11345},
  year={2024},
  eprint={2402.11345},
  archivePrefix={arXiv},
  primaryClass={stat.ML}
}
```

## Description of code
### Demos

- **BO-1D.ipynb:** A 1D toy case comparing VES-Gamma with other functions;
- **BO-2D-updated.ipynb:** Demonstration for implementing VES-Gamma on 12 different test functions;
- **iris.ipynb:** Hyper-parameter tuning for XGBoost on iris dataset;

### Support

- **ves.py:** Main file containing *VariationalEntropySearch* class and functions for running 1D and 2D experiments;

## Instruction for creating plots
Follow the provided demos

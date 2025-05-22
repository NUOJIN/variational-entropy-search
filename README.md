# Variational Entropy Search

This repo contains code for our paper "A Unified Framework for Entropy Search and Expected Improvement in Bayesian Optimization". 

## Referencing this code

If you use this code in any of your own work, please refer our paper: 
```
@misc{cheng2025unified,
  title={A Unified Framework for Entropy Search and Expected Improvement in Bayesian Optimization},
  author={Cheng, Nuojin* and Papenmeier, Leonard* and Becker, Stephen and Nardi, Luigi},
  year={2025},
  eprint={2501.18756},
  archivePrefix={arXiv},
  primaryClass={stat.ML}
}
```

## Example run

Assuming you have a running [Bencher](https://github.com/LeoIV/bencher) container, you can run VES on the Mopta08 benchmark with the following command:

```bash
python3 -m ves.main --benchmark mopta08 --exponential_family False \
--num_paths 128 --lengthscale_prior vbo --num_bo_iter 200 --reg_lambda 1"
```


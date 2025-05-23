

# Variational Entropy Search

This repository contains the official implementation of our paper:

**"A Unified Framework for Entropy Search and Expected Improvement in Bayesian Optimization"** (arXiv:2501.18756, accepted at ICML 2025 as a spotlight poster)

## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{cheng2025unified,
    title     = {A Unified Framework for Entropy Search and Expected Improvement in Bayesian Optimization},
    author    = {Cheng, Nuojin* and Papenmeier, Leonard* and Becker, Stephen and Nardi, Luigi},
    year      = {2025},
    eprint    = {2501.18756},
    archivePrefix = {arXiv},
    primaryClass  = {stat.ML}
}
```

## 🚀 Getting Started

Assuming you have a running [Bencher](https://github.com/LeoIV/bencher) container, you can run VES on the Mopta08 benchmark with the following command:

## Additional Information

For more details on parameters and usage, refer to the code documentation and comments.

```bash
python3 -m ves.main --benchmark mopta08 --exponential_family False \
--num_paths 128 --lengthscale_prior vbo --num_bo_iter 200 --reg_lambda 1"
```


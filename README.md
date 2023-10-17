# Channel-Adaptive Early Exiting using Reinforcement Learning for Multivariate Time Series Classification

This repository contains the official implementation of the CHARLEE framework described
in [Channel-Adaptive Early Exiting using Reinforcement Learning for Multivariate Time Series Classification](https://research.vu.nl/en/publications/channel-adaptive-early-exiting-using-reinforcement-learning-for-m), accepted
at ICMLA 2023. The extended [arXiv version](https://arxiv.org/abs/2306.14606) has a different title and abstract for double-blind review purposes.

## Requirements

The code is written in Python 3.8.13 and uses the following main dependencies:

* torch==1.13.0
* tsai==0.3.4
* sktime==0.13.4
* numpy==1.21.5
* scikit-learn==1.1.3
* pandas==1.5.1

## Acknowledgements

This repository contains code from [the repository of Dhariyal et al.](https://github.com/mlgig/Channel-Selection-MTSC),
utilizing [their work on channel selection](https://project.inria.fr/aaltd21/files/2021/09/AALTD_21_paper_15.pdf) for
multivariate time series classification.

Moreover, the initial structure of the framework and files is
based [on the code for Stop&Hop](https://github.com/Thartvigsen/StopAndHop), an [early classification method for irregular time series](https://dl.acm.org/doi/abs/10.1145/3511808.3557460), by Hartvigsen et al. 

## Datasets

The models are evaluated on a subset of
the [UEA multivariate dataset collection](https://www.timeseriesclassification.com/dataset.php). Moreover, we utilized datasets based on the [MAFAULDA
Machinery Fault Database](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html) and the [Case Western Reserve University Bearing Data](https://engineering.case.edu/bearingdatacenter).

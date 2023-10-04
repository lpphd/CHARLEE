# CHARLEE

Placeholder repository for the CHARLEE framework. The repository will soon be updated with the actual code and metrics.

# Channel-Adaptive Early Exiting using Reinforcement Learning for Multivariate Time Series Classification

This repository contains the official implementation of the experiments described in [Channel-Adaptive Early Exiting using Reinforcement Learning for Multivariate Time Series Classification](), accepted at ICMLA 2023 .

## Requirements

The code is written in Python 3.8.13 and uses the following main dependencies:

* torch==1.13.0
* tsai==0.3.4
* sktime==0.13.4
* numpy==1.21.5
* scikit-learn==1.1.3
* pandas==1.5.1


## Notes

For the WEASEL+MUSE, ResNet and InceptionTime models, MPI is used to distribute the workload of the experiments across
different nodes, but no communication among nodes is necessary.

There are no separate files for the baseline experiments, but these can be derived from the existing files by skipping
the data scaling alltogether.

## Datasets

The models are evaluated on a subset of
the [UEA multivariate dataset collection](https://www.timeseriesclassification.com/dataset.php).

## Results

The experiment metrics can be found under [Results](Results/), in the format [model\_name]\_uea\_metrics\_[scaling\_method]\_[dimension].csv

The baseline metrics are in the format [model\_name]\_uea\_metrics\_none.csv

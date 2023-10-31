# ANCKA and ANCKA-GPU
This repository contains the extension research work for AHCKA

## Pre-requisites
unzip all zip file in data/

CPU based AHCKA (python 3.9)

numpy, scipy, scikit-learn

GPU based AHCKA

additional library: cupy-cuda116, faiss-gpu=1.7.3=py3.9

## Dataset (Multipile types of attributed network)
Multiplex dataset: ACM, DBLP, IMDB

Hyper dataset: Cora-CA, Cora-CC, Query, Citeseeer, 20News, DBLP 

See https://github.com/CyanideCentral/AHCKA for MAG and AMAZON datasets, and download scann/faiss KNN searching index for large-scale graphs.

Undirected/Directed dataset: Cora, Undirected Citeseer, Pubmed, Citeseer, Cora-ML, Directed Citeseer.

## Reproduce command and hyperparameter setting
Refer to [bash.sh](bash.sh) for cpu and gpu based ANCKA and our sample CPU based output [CPU_log.log](CPU_log.log) / GPU based output [GPU_log.log](GPU_log.log) 
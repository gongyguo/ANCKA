# ANCKA and ANCKA-GPU
This repository contains the extension research work for AHCKA

## Pre-requisites
unzip all zip file in data/

CPU based AHCKA (python 3.9)

numpy, scipy, scikit-learn

GPU based AHCKA

additional library: cupy-cuda116, faiss-gpu=1.7.3=py3.9

## Dataset (Multipile types of attributed network)

Hyper dataset available: Cora-CA, Cora-CC, Query, Citeseeer, 20News, DBLP 

See https://github.com/CyanideCentral/AHCKA for MAG and AMAZON datasets, and download scann/faiss KNN searching index and collect in file INDEX/.

Undirected/Directed dataset available: Cora, Undirected Citeseer, Wiki, Directed Citeseer.

See https://renchi.ac.cn/datasets/ for Amazon2M and Tweibo dataset and put in data/npz file, and download faiss KNN searching index and collect in file INDEX/.

Multiplex dataset available: ACM, DBLP, IMDB

## Reproduce command and hyperparameter setting

Refer to [command.sh](command.sh) for cpu and gpu based ANCKA 
# ANCKA: A Versatile Framework for Attributed Network Clustering via K-Nearest Neighbor Augmentation

This repository contains the ANCKA clustering framework for general attributed networks, extended from the previously published AHCKA algorithm for attributed hypergraph:

    @article{LiYS23,
      author       = {Yiran Li and
                      Renchi Yang and
                      Jieming Shi},
      title        = {Efficient and Effective Attributed Hypergraph Clustering via K-Nearest
                      Neighbor Augmentation},
      journal      = {Proc. {ACM} Manag. Data},
      volume       = {1},
      number       = {2},
      pages        = {116:1--116:23},
      year         = {2023}
    }

## Pre-requisites

1. Unzip all zip file in data/

2. Prepare Python environment with pip or conda:

- Python version 3.9

- numpy, scipy, scikit-learn

- Optional libraries for large-scale networks: faiss-cpu/scann

- Optional libraries for ANCKA-GPU: cupy-cuda116, faiss-gpu=1.7.3=py3.9

## Dataset (Multipile types of attributed network)

Available dataset in this repro:

Hyper dataset: Cora-CA, Cora-CC, Query, Citeseer, 20News, DBLP 

Undirected/Directed dataset: Cora, Undirected Citeseer, Wiki, Directed Citeseer.

Multiplex dataset: ACM, DBLP, IMDB

Download [four large-scale datasets and used scann/faiss KNN index](https://github.com/CyanideCentral/AHCKA) and put them in data/ file.

## Reproduce command and hyperparameter setting

Refer to [command.sh](command.sh) for cpu and gpu based ANCKA's running command and hyperparameter setting 

## Sample output

```
CPU based
dataset:cora data:none network_type:UG
parameter setting: k=50 init_iter=25 beta=0.5
ACC=0.723 F1=0.686 NMI=0.556 ARI=0.484 Time=1.110s

dataset:citeseer data:none network_type:DG
parameter setting: k=50 init_iter=25 beta=0.4
ACC=0.696 F1=0.651 NMI=0.444 ARI=0.460 Time=1.152s

dataset:acm data:none network_type:MG
parameter setting: k=50 init_iter=25 beta=0.7
ACC=0.928 F1=0.928 NMI=0.739 ARI=0.796 Time=1.602s


GPU based
dataset:cora data:none network_type:UG
parameter setting: k=50 init_iter=25 beta=0.4
ACC=0.683 F1=0.621 NMI=0.533 ARI=0.470 Time=0.377s

dataset:citeseer data:none network_type:DG
parameter setting: k=50 init_iter=25 beta=0.4
ACC=0.694 F1=0.652 NMI=0.441 ARI=0.454 Time=0.839s

dataset:acm data:none network_type:MG
parameter setting: k=50 init_iter=25 beta=0.7
ACC=0.924 F1=0.924 NMI=0.730 ARI=0.786 Time=0.332s
```
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

See https://github.com/CyanideCentral/AHCKA for preprocessed and raw MAG and AMAZON datasets and put them in data/npz file, and download scann/faiss KNN searching index and collect in file INDEX/.

Undirected/Directed dataset available: Cora, Undirected Citeseer, Wiki, Directed Citeseer.

See https://renchi.ac.cn/datasets/ for raw Amazon2M and Tweibo datasets and put them in data/npz file, and download faiss KNN searching index and collect in file INDEX/.

Multiplex dataset available: ACM, DBLP, IMDB

## Reproduce command and hyperparameter setting

Refer to [command.sh](command.sh) for cpu and gpu based ANCKA 

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
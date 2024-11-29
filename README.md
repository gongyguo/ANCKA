# ANCKA: A Versatile Framework for Attributed Network Clustering via K-Nearest Neighbor Augmentation

This repository contains the **ANCKA** clustering framework for general attributed networks, extended from the previously published AHCKA algorithm for attributed hypergraph. If you find it helpful, please cite our work below:

```bibtex
@article{LiGSYSLL24,
  author       = {Yiran Li and
                  Gongyao Guo and
                  Jieming Shi and
                  Renchi Yang and
                  Shiqi Shen and
                  Qing Li and
                  Jun Luo},
  title        = {A versatile framework for attributed network clustering via K-nearest
                  neighbor augmentation},
  journal      = {{VLDB} J.},
  volume       = {33},
  number       = {6},
  pages        = {1913--1943},
  year         = {2024}
}
```

```bibtex
@article{LiYS23,
  author       = {Yiran Li and Renchi Yang and Jieming Shi},
  title        = {Efficient and Effective Attributed Hypergraph Clustering via K-Nearest Neighbor Augmentation},
  journal      = {Proc. {ACM} Manag. Data},
  volume       = {1},
  number       = {2},
  pages        = {116:1--116:23},
  year         = {2023}
}
```

The following attributed network datasets used in this work are also made available:

- Attributed undirected/directed graphs: Cora, Undirected Citeseer, Wiki, Directed Citeseer, Tweibo*, Amazon2M*.

- Attributed hypergraphs: Cora-CA, Cora-CC, Query, Citeseer, 20News, DBLP, Amazon*, MAG-PM*.

- Attributed multiplex graphs: ACM, DBLP, IMDB.

*\*Available at [Zenodo upload](https://zenodo.org/records/10426624) due to space limit of GitHub repository.*

## Pre-requisites

1. Prepare Python environment with pip or conda:

    - Python version 3.9

    - Install required libraries: numpy, scipy, scikit-learn

1. (Optional) Extract attributed multiplex datasets:

    ```shell
    unzip data/multi.zip -d data/
    ```

1. (Optional) To run ANCKA-GPU, install cupy and faiss-gpu=1.7.3.

1. (Optional) For testing large-scale datasets:

    - Install KNN search libraries: faiss-cpu and scann

    - Download datasets and KNN index files from [Zenodo upload](https://zenodo.org/records/10426624)

    - Extract files:

        ```shell
        cd ANCKA
        unzip ~/Download_path/ANCKA_data.zip -d data/
        ```

## Usage

To run ANCKA clustering algorithm please specify the name of dataset by command-line parameter `--dataset`, and for hypergraph datasets also specify the type of dataset by `--data`. Besides, also provide the type of attributed network by `--network_type`. Datasets supported by our implementation include the following:

| Type of dataset                           |    --data    |           --dataset            | --network_type |
| ----------------------------------------- | :----------: | :----------------------------: | :------------: |
| Co-authorship hypergraph in .pickle files | coauthorship |           cora, dblp           |       HG       |
| Co-citation hypergraph in .pickle files   |  cocitation  |         cora, citeseer         |       HG       |
| Attributed hypergraph stored in .npz file |     npz      |  query, 20news, amazon, magpm  |       HG       |
| Attributed undirected graph               |      -       | cora,citeseer-UG,wiki,Amazon2M |       UG       |
| Attributed directed graph                 |      -       |       citeseer-DG,Tweibo       |       DG       |
| Attributed multiplex graph                |      -       |        ACM,IMDB,DBLP-MG        |       MG       |


Other parameters are optional:

| Parameter  | Default | Description                                                                                                   |
| ---------- | ------- | ------------------------------------------------------------------------------------------------------------- |
| --knn_k    | 10      | $K$, the size of neighborhood in KNN graph                                                                    |
| --alpha    | 0.2     | $\alpha$, restart probability in random walk with restart (RWR)                                               |
| --beta     | 0.5     | $\beta$, the weight of KNN random walk                                                                        |
| --tmax     | 200     | $T_{a}$, the maximum number of orthogonal iterations                                                          |
| --interval | 5       | $\tau$, the interval of computing discrete cluster labels                                                     |
| --scale    | -       | Apply settings for large-scale data: approximate KNN with ScaNN or Faiss; simplified initialization ($T_i$=1) |
| --gpu      | -       | Use ANCKA-GPU implementation for clustering                                                       |
| --times    | 10      | Rerun ANCKA-GPU algorithm to obtain average metrics                                                              |
| --caltime  | -       | Measure the time costs of different segments in ANCKA or ANCKA-GPU algorithm                                            |
| --verbose  | -       | Produce verbose command-line output                                                                           |

To **reproduce** the experiment results of ANCKA and ANCKA-GPU in our paper, please refer to [command.sh](command.sh) for corresponding datasets.

Sample output of running `bash command.sh`:

```text
CPU based

dataset:dblp data:coauthorship network_type:HG
parameter setting: k=10 init_iter=25 beta=0.5
ACC=0.797 F1=0.774 NMI=0.632 ARI=0.632 Time=30.899s

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

dataset:dblp data:coauthorship network_type:HG
parameter setting: k=10 init_iter=25 beta=0.5
ACC=0.808 F1=0.787 NMI=0.643 ARI=0.646 Time=0.663s

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

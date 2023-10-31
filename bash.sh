OUTPUT1="CPU_log.log"
OUTPUT2="GPU_log.log"
PYTHON=/home/gongyao/anaconda3/bin/python
set=set3
{
        echo "CPU based"
        echo "Hyper"
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset query --data npz --graph_type Hypergraph  --beta 0.5 --knn_k 10
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --data coauthorship --graph_type Hypergraph   --beta 0.5 --knn_k 10
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --data cocitation --graph_type Hypergraph   --beta 0.5 --knn_k 10
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --data cocitation --graph_type Hypergraph   --beta 0.5 --knn_k 10
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset 20news --data npz --graph_type Hypergraph   --beta 0.5 --knn_k 10
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset dblp --data coauthorship --graph_type Hypergraph   --beta 0.5 --knn_k 10

        echo "Undirected"
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --graph_type Undirected  --beta 0.5 --knn_k 50
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --graph_type Undirected  --beta 0.4 --knn_k 50
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset pubmed --graph_type Undirected  --beta 0.06 --knn_k 50
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset wiki --graph_type Undirected  --beta 0.5 --knn_k 50

        echo "Directed"
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --graph_type Directed  --beta 0.4 --knn_k 50
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora_ml --graph_type Directed  --beta 0.5 --knn_k 50

        echo "Multi"
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset acm --graph_type Multi  --beta 0.7 --knn_k 50
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset imdb --graph_type Multi  --beta 0.4 --knn_k 50
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset dblp --graph_type Multi  --beta 0.1 --knn_k 50

}|tee $OUTPUT1

{
        echo "GPU based"
        echo "Hyper"
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset query --data npz --graph_type Hypergraph --beta 0.5 --knn_k 10 --gpu
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --data coauthorship --graph_type Hypergraph --beta 0.5 --knn_k 10 --gpu
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --data cocitation --graph_type Hypergraph   --beta 0.5 --knn_k 10 --gpu
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --data cocitation --graph_type Hypergraph   --beta 0.5 --knn_k 10 --gpu
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset 20news --data npz --graph_type Hypergraph   --beta 0.5 --knn_k 10 --gpu
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset dblp --data coauthorship --graph_type Hypergraph   --beta 0.5 --knn_k 10 --gpu
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset amazon --data npz --graph_type Hypergraph --scale --beta 0.4 --knn_k 10 --gpu --times 1
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset magpm --data npz --graph_type Hypergraph  --scale --beta 0.4 --knn_k 10 --gpu --times 1

        echo "Undirected"
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --graph_type Undirected  --beta 0.4 --knn_k 50 --gpu
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --graph_type Undirected  --beta 0.4 --knn_k 50 --gpu
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset pubmed --graph_type Undirected  --beta 0.06 --knn_k 50 --gpu
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset wiki --graph_type Undirected  --beta 0.5 --knn_k 50 --gpu

        echo "Directed"
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --graph_type Directed  --beta 0.4 --knn_k 50 --gpu
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora_ml --graph_type Directed  --beta 0.5 --knn_k 50 --gpu

        echo "Multi"
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset acm --graph_type Multi  --beta 0.7 --knn_k 50 --gpu
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset imdb --graph_type Multi  --beta 0.4 --knn_k 50 --gpu
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset dblp --graph_type Multi  --beta 0.1 --knn_k 50 --gpu

}|tee $OUTPUT2
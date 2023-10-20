OUTPUT1="CPU_log.log"
OUTPUT2="GPU_log.log"
PYTHON=/home/gongyao/anaconda3/bin/python
set=set3
{
        echo "CPU based"
        echo "Hyper"
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset query --data npz --graph_type Hypergraph  --beta 0.5 --knn_k 10
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --data coauthorship --graph_type Hypergraph   --beta 0.5 --knn_k 10
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --data cocitation --graph_type Hypergraph   --beta 0.5 --knn_k 10
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --data cocitation --graph_type Hypergraph   --beta 0.5 --knn_k 10
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset 20news --data npz --graph_type Hypergraph   --beta 0.5 --knn_k 10
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset dblp --data coauthorship --graph_type Hypergraph   --beta 0.5 --knn_k 10

        echo "Undirected"
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --graph_type Undirected  --beta 0.5 --knn_k 50
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --graph_type Undirected  --beta 0.4 --knn_k 50
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset pubmed --graph_type Undirected  --beta 0.06 --knn_k 50
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset wiki --graph_type Undirected  --beta 0.5 --knn_k 50

        echo "Directed"
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --graph_type Directed  --beta 0.4 --knn_k 50
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora_ml --graph_type Directed  --beta 0.5 --knn_k 50

        echo "Multi"
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset acm --graph_type Multi  --beta 0.7 --knn_k 50
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset imdb --graph_type Multi  --beta 0.4 --knn_k 50
        cset proc -s $set -e $PYTHON -- ANCKA.py --dataset dblp --graph_type Multi  --beta 0.1 --knn_k 50

}|tee  $OUTPUT1

{
        echo "GPU based"
        echo "Hyper"
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset query --data npz --graph_type Hypergraph  --beta 0.5 --knn_k 10
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset cora --data coauthorship --graph_type Hypergraph   --beta 0.5 --knn_k 10
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset cora --data cocitation --graph_type Hypergraph   --beta 0.5 --knn_k 10
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset citeseer --data cocitation --graph_type Hypergraph   --beta 0.5 --knn_k 10
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset 20news --data npz --graph_type Hypergraph   --beta 0.5 --knn_k 10
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset dblp --data coauthorship --graph_type Hypergraph   --beta 0.5 --knn_k 10

        echo "Undirected"
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset cora --graph_type Undirected  --beta 0.4 --knn_k 50
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset citeseer --graph_type Undirected  --beta 0.4 --knn_k 50
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset pubmed --graph_type Undirected  --beta 0.06 --knn_k 50
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset wiki --graph_type Undirected  --beta 0.5 --knn_k 50

        echo "Directed"
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset citeseer --graph_type Directed  --beta 0.4 --knn_k 50
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset cora_ml --graph_type Directed  --beta 0.5 --knn_k 50

        echo "Multi"
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset acm --graph_type Multi  --beta 0.7 --knn_k 50
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset imdb --graph_type Multi  --beta 0.4 --knn_k 50
        cset proc -s $set -e $PYTHON -- gANCKA.py --dataset dblp --graph_type Multi  --beta 0.1 --knn_k 50

}|tee  $OUTPUT2





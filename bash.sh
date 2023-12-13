OUTPUT1="CPU_log.log"
OUTPUT2="GPU_log.log"
OUTPUT3="test.log"
PYTHON=/home/gongyao/anaconda3/bin/python
set=set3
# {
        # echo "CPU based"

        # echo "HG"
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset query --data npz --network_type HG  --beta 0.5 --knn_k 10
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --data coauthorship --network_type HG   --beta 0.5 --knn_k 10
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --data cocitation --network_type HG   --beta 0.5 --knn_k 10
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --data cocitation --network_type HG   --beta 0.5 --knn_k 10
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset 20news --data npz --network_type HG   --beta 0.5 --knn_k 10
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset dblp --data coauthorship --network_type HG   --beta 0.5 --knn_k 10

        # echo "UG"
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --network_type UG  --beta 0.5 --knn_k 50
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --network_type UG  --beta 0.4 --knn_k 50
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset wiki --network_type UG  --beta 0.5 --knn_k 50
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset amazon2m --network_type UG  --beta 0.4 --knn_k 10 --scale

        # echo "DG"
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --network_type DG  --beta 0.4 --knn_k 50
        sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset tweibo --network_type DG  --beta 0.4 --knn_k 10 --scale

        # echo "MG"
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset acm --network_type MG  --beta 0.7 --knn_k 50
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset imdb --network_type MG  --beta 0.4 --knn_k 50
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset dblp --network_type MG  --beta 0.1 --knn_k 50

# }|tee $OUTPUT3

# {
#         echo "GPU based"

#         echo "HG"
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset query --data npz --network_type HG --beta 0.5 --knn_k 10 --gpu
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --data coauthorship --network_type HG --beta 0.5 --knn_k 10 --gpu
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --data cocitation --network_type HG   --beta 0.5 --knn_k 10 --gpu
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --data cocitation --network_type HG   --beta 0.5 --knn_k 10 --gpu
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset 20news --data npz --network_type HG   --beta 0.5 --knn_k 10 --gpu
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset dblp --data coauthorship --network_type HG   --beta 0.5 --knn_k 10 --gpu
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset amazon --data npz --network_type HG --scale --beta 0.4 --knn_k 10 --gpu --times 1
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset magpm --data npz --network_type HG  --scale --beta 0.4 --knn_k 10 --gpu --times 1

#         echo "UG"
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset cora --network_type UG  --beta 0.4 --knn_k 50 --gpu
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --network_type UG  --beta 0.4 --knn_k 50 --gpu
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset wiki --network_type UG  --beta 0.5 --knn_k 50 --gpu
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset amazon2m --network_type UG  --beta 0.4 --knn_k 10 --scale --gpu --times 1

#         echo "DG"
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset citeseer --network_type DG  --beta 0.4 --knn_k 50 --gpu
        # sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset tweibo --network_type DG  --beta 0.4 --knn_k 10 --scale --gpu --times 1

#         echo "MG"
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset acm --network_type MG  --beta 0.7 --knn_k 50 --gpu
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset imdb --network_type MG  --beta 0.4 --knn_k 50 --gpu
#         sudo cset proc -s $set -e $PYTHON -- ANCKA.py --dataset dblp --network_type MG  --beta 0.1 --knn_k 50 --gpu

# }|tee $OUTPUT3
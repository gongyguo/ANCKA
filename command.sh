{
    echo "CPU based"

    echo "HG"
    python ANCKA.py --dataset query --data npz --network_type HG  --beta 0.5 --knn_k 10
    python ANCKA.py --dataset cora --data coauthorship --network_type HG   --beta 0.5 --knn_k 10
    python ANCKA.py --dataset cora --data cocitation --network_type HG   --beta 0.5 --knn_k 10
    python ANCKA.py --dataset citeseer --data cocitation --network_type HG   --beta 0.5 --knn_k 10
    python ANCKA.py --dataset 20news --data npz --network_type HG   --beta 0.5 --knn_k 10
    python ANCKA.py --dataset dblp --data coauthorship --network_type HG   --beta 0.5 --knn_k 10

    echo "UG"
    python ANCKA.py --dataset cora --network_type UG  --beta 0.5 --knn_k 50
    python ANCKA.py --dataset citeseer --network_type UG  --beta 0.4 --knn_k 50
    python ANCKA.py --dataset wiki --network_type UG  --beta 0.5 --knn_k 50
    python ANCKA.py --dataset amazon2m --network_type UG  --beta 0.4 --knn_k 10 --scale

    echo "DG"
    python ANCKA.py --dataset citeseer --network_type DG  --beta 0.4 --knn_k 50
    python ANCKA.py --dataset tweibo --network_type DG  --beta 0.4 --knn_k 10 --scale

    echo "MG"
    python ANCKA.py --dataset acm --network_type MG  --beta 0.7 --knn_k 50
    python ANCKA.py --dataset imdb --network_type MG  --beta 0.4 --knn_k 50
    python ANCKA.py --dataset dblp --network_type MG  --beta 0.1 --knn_k 50
}

{
    echo "GPU based"

    echo "HG"
    python ANCKA.py --dataset query --data npz --network_type HG --beta 0.5 --knn_k 10 --gpu
    python ANCKA.py --dataset cora --data coauthorship --network_type HG --beta 0.5 --knn_k 10 --gpu
    python ANCKA.py --dataset cora --data cocitation --network_type HG   --beta 0.5 --knn_k 10 --gpu
    python ANCKA.py --dataset citeseer --data cocitation --network_type HG   --beta 0.5 --knn_k 10 --gpu
    python ANCKA.py --dataset 20news --data npz --network_type HG   --beta 0.5 --knn_k 10 --gpu
    python ANCKA.py --dataset dblp --data coauthorship --network_type HG   --beta 0.5 --knn_k 10 --gpu
    python ANCKA.py --dataset amazon --data npz --network_type HG --scale --beta 0.4 --knn_k 10 --gpu --times 1
    python ANCKA.py --dataset magpm --data npz --network_type HG  --scale --beta 0.4 --knn_k 10 --gpu --times 1

    echo "UG"
    python ANCKA.py --dataset cora --network_type UG  --beta 0.4 --knn_k 50 --gpu
    python ANCKA.py --dataset citeseer --network_type UG  --beta 0.4 --knn_k 50 --gpu
    python ANCKA.py --dataset wiki --network_type UG  --beta 0.5 --knn_k 50 --gpu
    python ANCKA.py --dataset amazon2m --network_type UG  --beta 0.4 --knn_k 10 --scale --gpu --times 1

    echo "DG"
    python ANCKA.py --dataset citeseer --network_type DG  --beta 0.4 --knn_k 50 --gpu
    python ANCKA.py --dataset tweibo --network_type DG  --beta 0.4 --knn_k 10 --scale --gpu --times 1

    echo "MG"
    python ANCKA.py --dataset acm --network_type MG  --beta 0.7 --knn_k 50 --gpu
    python ANCKA.py --dataset imdb --network_type MG  --beta 0.4 --knn_k 50 --gpu
    python ANCKA.py --dataset dblp --network_type MG  --beta 0.1 --knn_k 50 --gpu
}
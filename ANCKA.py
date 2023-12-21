#-- coding:utf-8 --
import config
import numpy as np
import scipy.sparse as sp
from data import data
from sklearn.preprocessing import normalize
import argparse
import random

p = argparse.ArgumentParser(description='Set parameter')
p.add_argument('--data', type=str, default='none', help='data type (coauthorship/cocitation) for hypergraph')
p.add_argument('--dataset', type=str, default='acm', help='dataset name (e.g.: cora/dblp for coauthorship, cora/citeseer for cocitation)')
p.add_argument('--tmax', type=int, default=200, help='t_max parameter')
p.add_argument('--seeds', type=int, default=0, help='seed for randomness')
p.add_argument('--alpha', type=float, default=0.2, help='mhc parameter')
p.add_argument('--beta', type=float, default=0.5, help='weight of knn random walk')
p.add_argument('--metric', type=bool, default=False, help='calculate additional metrics: modularity')
p.add_argument('--weighted_p', type=int, default=0, help='use transition matrix p weighted by attribute similarity')
p.add_argument('--verbose', action='store_true', help='print verbose logs')
p.add_argument('--scale', action='store_true', help='use configurations for large-scale data')
p.add_argument('--caltime', action='store_true', help='calculate time of different part of ANCKA')
p.add_argument('--gpu_usage', action='store_true', help='calculate gpu usage')
p.add_argument('--gpu', action='store_true', help='use gpu ANCKA version for clustering')
p.add_argument('--interval', type=int, default=5, help='interval between cluster predictions during orthogonal iterations')
p.add_argument('--times', type=int, default=10, help='rerun gpu ANCKA version to calculate avg time and metric')
p.add_argument('--knn_k', type=int, default=10, help='knn graph neighbors')
p.add_argument('--init_iter', type=int, default=25, help='BCM iteration')
p.add_argument('--network_type', type=str, default='MG', help='network type'
                '(e.g.: HG, MG, UG, DG)')
args = p.parse_args()

def random_walk(adj,type):

    if type == "HG":
        p_mat = [normalize(adj.T, norm='l1', axis=1), normalize(adj, norm='l1', axis=1)]
    elif type =="MG":
        config.num_view = len(adj)
        P = [normalize(layer_adj, norm='l1', axis=1) for layer_adj in adj]
        p_mat = [sum([pm*1./config.num_view for i, pm in enumerate(P)])]
    elif type =="UG" or type=="DG":
        p_mat = [normalize(adj, norm='l1', axis=1)]
    else:
        raise NotImplementedError
    return p_mat

def run_ancka():
        
    dataset = data.load(config.dataset,config.data,config.network_type)
    features = dataset['features_sp']
    labels = dataset['labels']
    labels = np.asarray(np.argmax(labels, axis=1)) if labels.ndim == 2 else labels
    config.labels = labels
    num_nodes = dataset['n']
    k = len(np.unique(labels))

    seed = config.seeds
    np.random.seed(seed)
    random.seed(seed)

    adj = dataset['adj_sp']       
    config.adj = adj
    config.features = features.copy()
    p_mat = random_walk(config.adj,config.network_type)
    
    if config.network_type == 'MG': 
        config.num_view = len(adj)
        d_tvec = [np.asarray(layer_adj.sum(0)).flatten() for layer_adj in adj]
        deg_dict = {i: sum([layer_dvec[i] for layer_dvec in d_tvec]) for i in range(num_nodes)}
    
    else:    
        d_vec = np.asarray(config.adj.sum(0)).flatten()
        deg_dict = {i: d_vec[i] for i in range(len(d_vec))}
        
    results = None

    if not config.gpu:
        from cluster import cluster
        results = cluster(p_mat, num_nodes, features, k, deg_dict, alpha=config.alpha, beta=config.beta, tmax=config.tmax, ri=False, weighted_p=config.weighted_p)
    
    else:
        from gcluster import cluster
        times = args.times
        if sp.issparse(features):
            features = features.todense()
        features = np.asarray(features,order='C').astype('float32')
        results = cluster(times, p_mat, features, k, num_nodes, deg_dict, alpha=config.alpha, beta=config.beta, tmax=config.tmax, ri=False, weighted_p=config.weighted_p)

    return results

if __name__ == '__main__':
    config.data = args.data
    config.dataset = args.dataset
    config.metric = args.metric
    config.tmax = args.tmax
    config.beta = args.beta
    config.alpha = args.alpha
    config.seeds = args.seeds
    config.verbose = args.verbose
    config.caltime = args.caltime
    config.cluster_interval = args.interval
    config.network_type = args.network_type
    config.knn_k = args.knn_k
    config.init_iter = args.init_iter

    if args.scale:
        config.approx_knn = True
        config.init_iter = 1

    if args.gpu:
        config.gpu = True

    if args.gpu_usage:
        config.gpu_usage = True
    
    print(f"dataset:{config.dataset} data:{config.data} network_type:{config.network_type}")
    print(f"parameter setting: k={config.knn_k} init_iter={config.init_iter} beta={config.beta}")
    acc, nmi, f1, adj_s, time, _= run_ancka()
    print(f"ACC={format(acc,'.3f')} F1={format(f1,'.3f')} NMI={format(nmi,'.3f')} ARI={format(adj_s,'.3f')} Time={format(time,'.3f')}s")



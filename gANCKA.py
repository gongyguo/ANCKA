#-- coding:utf-8 --
import config
import numpy as np
from data import data
import argparse
import random

p = argparse.ArgumentParser(description='Set parameter')
p.add_argument('--data', type=str, default='none', help='data type (coauthorship/cocitation)')
p.add_argument('--dataset', type=str, default='acm', help='dataset name (e.g.: cora/dblp for coauthorship, cora/citeseer for cocitation)')
p.add_argument('--tmax', type=int, default=200, help='t_max parameter')
p.add_argument('--seeds', type=int, default=0, help='seed for randomness')
p.add_argument('--alpha', type=float, default=0.2, help='mhc parameter')
p.add_argument('--beta', type=float, default=0.5, help='weight of knn random walk')
p.add_argument('--metric', type=bool, default=False, help='calculate additional metrics: modularity')
p.add_argument('--weighted_p', type=int, default=0, help='use transition matrix p weighted by attribute similarity')
p.add_argument('--verbose', action='store_true', help='print verbose logs')
p.add_argument('--scale', action='store_true', help='use configurations for large-scale data')
p.add_argument('--caltime', action='store_true', help='calculate time of each part')
p.add_argument('--gpu_usage', action='store_true', help='calculate gpu usage')
p.add_argument('--dis', type=str, default= 'gpu', help='type of method to assign label')
p.add_argument('--interval', type=int, default=5, help='interval between cluster predictions during orthogonal iterations')
p.add_argument('--times', type=int, default=11, help='times of running cluster')
p.add_argument('--param', type=str, default=' ', help='faiss index')
p.add_argument('--index', action='store_true', help='use previous index')
p.add_argument('--gamma', nargs='+',action='append', help='laplacian weights')
p.add_argument('--knn_k', type=int, default=10, help='knn graph neighbors')
p.add_argument('--init_iter', type=int, default=25, help='BCM iteration')
p.add_argument('--graph_type', type=str, default='Multi', help='graph type'
                '(e.g.: Hypergraph, Multi, Undirected, Directed)')
args = p.parse_args()

def run_ancka_gpu():

    if config.graph_type=='Hypergraph':

        from gcluster import cluster
        dataset = data.load(config.data, config.dataset)
        features = dataset['features_sp']
        labels = dataset['labels']

        labels = np.asarray(np.argmax(labels, axis=1)) if labels.ndim == 2 else labels
        config.labels = labels
        k  = len(np.unique(labels))

        seed = config.seeds
        np.random.seed(seed)
        random.seed(seed)

        hg_adj = dataset['adj_sp']
        n = hg_adj.shape[1]
        config.hg_adj = hg_adj
        config.features = features.copy()
        d_vec = np.asarray(config.hg_adj.sum(0)).flatten()
        deg_dict = {i: d_vec[i] for i in range(len(d_vec))}

        results = None

        times = args.times

        results = cluster(times, hg_adj, features, k, n, deg_dict, alpha=config.alpha, beta=config.beta, tmax=config.tmax, ri=False, weighted_p=config.weighted_p)
        return results
    
    if config.graph_type == 'Directed' or config.graph_type == 'Undirected':

        from gcluster import cluster_graph
        dataset = data.load_simple(config.graph_type,config.dataset)
        features = dataset['features']
        labels = dataset['labels']
        adj = dataset['adj_sp']
        num_nodes = dataset['n']
        if config.graph_type == 'Undirected':
            config.adj = adj
        else:
            config.adj = adj+adj.T

        config.adj.data[config.adj.data>0]=1.0

        labels = np.asarray(np.argmax(labels, axis=1)) if labels.ndim == 2 else labels
        config.labels = labels
        k = len(np.unique(labels))

        seed = config.seeds
        np.random.seed(seed)
        random.seed(seed)

        diagonal_indices = (np.arange(num_nodes), np.arange(num_nodes))
        config.adj[diagonal_indices] = 0.0
        config.features = features.copy()
        d_vec = np.asarray(config.adj.sum(0)).flatten()
        deg_dict = {i: d_vec[i] for i in range(len(d_vec))}

        results = None

        times = args.times

        results = cluster_graph(times,config.adj, features, k, num_nodes, deg_dict, alpha=config.alpha, beta=config.beta, tmax=config.tmax, ri=False, weighted_p=config.weighted_p)

        return results
    
    if config.graph_type == 'Multi':
        
        from gcluster import cluster_multi
        if config.dataset == "acm":
            dataset = data.load_acm()
        if config.dataset == "imdb":
            dataset = data.load_imdb()
        if config.dataset == "dblp":
            dataset = data.load_dblp()

        features = dataset['features']
        labels = dataset['labels']
        labels = np.asarray(np.argmax(labels, axis=1)) if labels.ndim == 2 else labels
        config.labels = labels
        num_nodes = dataset['n']
        k = len(np.unique(labels))

        seed = config.seeds
        np.random.seed(seed)
        random.seed(seed)

        adj_list = dataset['adj']
        config.num_view = len(adj_list)
        config.adj = adj_list
        config.features = features.copy()
        d_tvec = [np.asarray(adj.sum(0)).flatten() for adj in adj_list]
        deg_dict = [{i: d_vec[i] for i in range(len(d_vec))} for d_vec in d_tvec]

        results = None

        times = args.times

        results = cluster_multi(times,config.adj, features, k, num_nodes, deg_dict, alpha=config.alpha, beta=config.beta, gamma=config.gamma,tmax=config.tmax, ri=False, weighted_p=config.weighted_p)

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
    config.dis = args.dis
    config.param =args.param

    config.graph_type = args.graph_type
    config.knn_k = args.knn_k

    config.init_iter = args.init_iter
    if args.gamma != None:
        config.gamma = [float(x) for x in args.gamma[0]]
    else:
        config.gamma = []

    if args.scale:
        config.approx_knn = True
        config.init_iter = 1

    if args.scale:
        config.approx_knn = True
        config.init_iter = 1
    if args.gpu_usage:
        config.gpu_usage = True
        config.index = True
        
    print(config.dataset+' '+config.data+": knn neighbors = " + str(config.knn_k))
    print(f"{config.knn_k} {config.init_iter} {config.beta}")
    run_ancka_gpu()



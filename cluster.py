from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
import numpy as np
import resource
import scipy.sparse as sp
from munkres import Munkres
import config
import heapq
from spectral import discretize
from scipy.linalg import qr
import time
from scipy.sparse import csc_matrix
from numpy import linalg as LA
import operator
import random

def early_stop(stats):
    return len(stats)>3 and stats[-1]>stats[-2] and stats[-2]>stats[-3]

class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label


    def clusteringAcc(self):
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal!!!!')
            c1_clusters = {c: set() for c in set(l1)}
            c2_clusters = {c: set() for c in set(l2)}
            
            for i in range(len(self.true_label)):
                c1 = self.true_label[i]
                c2 = self.pred_label[i]
                c1_clusters[c1].add(i)
                c2_clusters[c2].add(i)

            c2_c1 = {}
            for c2 in set(l2):
                for c1 in set(l1):
                    c2_c1[str(c2)+","+str(c1)]=0


            for (c1, s1) in c1_clusters.items():
                for (c2, s2) in c2_clusters.items():
                    num_com_s1s2 = len(s1.intersection(s2))
                    c2_c1[str(c2)+","+str(c1)]=num_com_s1s2

            sorted_x = sorted(c2_c1.items(), key=operator.itemgetter(1), reverse=True)
            
            c2_c1_map = {}
            c1_flag = {c: True for c in set(l1)}
            c2_flag = {c: True for c in set(l2)}
            for (k, v) in sorted_x:
                if len(c2_c1_map.keys())==numclass1:
                    break
                c2, c1 = k.split(',')
                c2, c1 = int(c2), int(c1)
                if c1_flag[c1] and c2_flag[c2]:
                    c2_c1_map[c2]=c1

                c1_flag[c1] = False
                c2_flag[c2] = False
            
            new_predict = np.zeros(len(self.pred_label))
            for i in range(len(l2)):
                new_predict[i] = c2_c1_map[self.pred_label[i]]
                
        else:
            cost = np.zeros((numclass1, numclass2), dtype=np.float64)
            for i, c1 in enumerate(l1):
                mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
                for j, c2 in enumerate(l2):
                    mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                    cost[i][j] = len(mps_d)

            m = Munkres()
            cost = cost.__neg__().tolist()

            indexes = m.compute(cost)

            new_predict = np.zeros(len(self.pred_label))
            for i, c in enumerate(l1):
                c2 = l2[indexes[i][1]]

                ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
                new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        return acc, f1_macro, precision_macro, recall_macro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1, pre, rc = self.clusteringAcc()

        return acc, nmi, f1, pre, adjscore, rc

def cluster(P, n, X, num_cluster, deg_dict, alpha=0.2, beta = 0.5, t=5, tmax=200, ri=False, weighted_p=0):

    start_time = time.time()

    #use identical setting as AHCKA
    if not config.network_type == "HG" or config.approx_knn:
        config.knn_k-=1

    if config.approx_knn and config.network_type == "HG":
        import scann
        ftd = X.todense()
        if config.dataset.startswith('amazon'):
            searcher = scann.scann_ops_pybind.load_searcher('scann_amazon')
        else:
            searcher = scann.scann_ops_pybind.load_searcher('scann_magpm')
        neighbors, distances = searcher.search_batched_parallel(ftd)
        del ftd
        knn = sp.csr_matrix(((distances.ravel()), neighbors.ravel(), np.arange(0, neighbors.size+1, neighbors.shape[1])), shape=(n, n))
        knn.setdiag(0.0)

    elif config.approx_knn and (config.network_type == "UG" or config.network_type == "DG"):
        import faiss
        X= X.astype(np.float32).tocoo()
        ftd = np.zeros(X.shape, X.dtype)
        ftd[X.row, X.col] = X.data
        faiss.normalize_L2(ftd)
        index = faiss.read_index(f"INDEX/{config.dataset}_cpu.index")
        distances, neighbors = index.search(ftd, config.knn_k+1)
        knn = sp.csr_matrix(((distances.ravel()), neighbors.ravel(), np.arange(0, neighbors.size+1, neighbors.shape[1])), shape=(n, n))
        knn.setdiag(0.0)

    else:
        knn = kneighbors_graph(X, config.knn_k, metric="cosine", mode="distance", n_jobs=16)
        knn.data = 1.0-knn.data
    knn = knn + knn.T #A_k
    Q = normalize(knn, norm='l1')

    if config.caltime:
        t1 = time.time()
        print(f"knn_time: {t1-start_time}")

    num_topk_deg = num_cluster
    topk_deg_nodes = heapq.nlargest(int(num_topk_deg), deg_dict, key=deg_dict.get)
    if config.network_type=="HG":
        PC = P[0]@P[1][:,topk_deg_nodes]
    else:
        PC = P[0][:,topk_deg_nodes]
    M = PC
    for i in range(config.init_iter):
        if config.network_type=="HG":
            M = (1-alpha)*P[0]@(P[1].dot(M))+PC
        else:
            M = (1-alpha)*P[0].dot(M)+PC
    class_evdsum = M.sum(axis=0).flatten().tolist()[0]
    newcandidates = np.argpartition(class_evdsum, -num_cluster)[-num_cluster:]
    M = M[:,newcandidates]
    labels = np.asarray(np.argmax(M, axis=1)).flatten()
    
    if config.random_init is True:
        lls = np.unique(labels)
        for i in range(n):
            ll = random.choice(lls)
            labels[i] = ll
    
    M = csc_matrix((np.ones(len(labels)), (np.arange(0, M.shape[0]), labels)),shape=(M.shape))
    M = M.todense()

    Mss = np.sqrt(M.sum(axis=0))
    Mss[Mss==0]=1
    q = M*1.0/Mss

    e1 = np.ones(shape = (n,1))
    q = np.hstack([e1,q])

    predict_clusters_best=labels
    iter_best = 0
    conductance_best=100
    conductance_best_acc = [0]*3

    err = 1

    if config.caltime:
        t2 = time.time()
        print(f"init_time: {t2-t1}")

    if beta>0.0 and config.network_type=="HG":
        unconnected = np.asarray(config.adj.sum(0)).flatten()==0
        Q[unconnected, :] *= (1./beta)
    if config.approx_knn and config.network_type=="HG":
        mask = np.ones(P[0].shape[0])
        mask[np.argwhere(X.sum(1)==0)[:,0]]*=(1./(1-beta))
        P = [sp.diags(mask)@P[0], P[1]]

    conductance_stats = []

    for i in range(tmax):
        if config.network_type=="HG":
            z = (1-beta)*P[0]@(P[1].dot(q))+ (beta)*Q.dot(q)
        else:
            z = (1-beta)*P[0].dot(q)+ (beta)*Q.dot(q)
        q_prev = q
        q, _ = qr(z, mode='economic')
        err = LA.norm(q-q_prev)/LA.norm(q)

        if (i+1)%config.cluster_interval==0:
            leading_eigenvectors = q[:,1:num_cluster+1]
            predict_clusters, y = discretize(leading_eigenvectors)
            
            conductance_cur = 0
            z_0 = config.alpha * y
            z = z_0
            for j in range(config.num_hop):
                if config.network_type=="HG":
                    z = (1-config.alpha)*((1-beta)*P[0]@(P[1].dot(z))+ (beta)*Q.dot(z)) + z_0
                else:
                    z = (1-config.alpha)*((1-beta)*P[0].dot(z)+ (beta)*Q.dot(z)) + z_0
            ct=y.T@z
            conductance_cur = 1.0-np.trace(ct)/num_cluster

            if config.verbose:
                print(i, err, conductance_cur)
            conductance_stats.append(conductance_cur)
                
            if conductance_cur<conductance_best:
                conductance_best = conductance_cur
                predict_clusters_best = predict_clusters
                iter_best = i

            if config.cond_early_stop and early_stop(conductance_stats):
                break

            if err <= config.q_epsilon:
                break

    if config.caltime:
        t3 = time.time()
        print(f"orth_time: {t3-t2}")
        
    end_time = time.time()
    peak_memory=0
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    if config.verbose:
        print("%f seconds in clustering"%(end_time-start_time))
        print(np.unique(predict_clusters_best))
        print("best iter: %d, best mhc: %f, acc: %f, %f, %f"%(iter_best, conductance_best, conductance_best_acc[0], conductance_best_acc[1], conductance_best_acc[2]))
    cm = clustering_metrics(config.labels, predict_clusters_best)
    acc, nmi, f1, pre, adj_s, rec = cm.evaluationClusterModelFromLabel()

    # print(f"{acc} {f1} {nmi} {adj_s} {end_time-start_time} {peak_memory}")
    return [acc, nmi, f1, adj_s, end_time-start_time, peak_memory]


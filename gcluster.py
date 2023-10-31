from sklearn import metrics
from sklearn.preprocessing import normalize 
import resource
import numpy as np
import cupy as cp
import scipy.sparse as sp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse import csc_matrix 
from cupy import linalg as cLA
from munkres import Munkres
import heapq
from gspectral import gdiscretize
import time
import operator
import random
import config
import os
import signal
import subprocess
import faiss 

switch_gpu = 1
gpu_usage = []
flops = []
total_time = []
pure_time = []
acc_list=[]
f1_list=[]
nmi_list=[]
ari_list=[]
orth_time = [] 
copy_time = []
dis_time = []
mhc_time = []
knn_time= []
init_time= []
res = faiss.StandardGpuResources()
co = faiss.GpuClonerOptions()

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
    
def cluster(times, P, X, num_cluster, num_node, deg_dict, alpha=0.2, beta = 0.35, t=5, tmax=1000, ri=False, weighted_p=0):
    
    n = num_node
    if sp.issparse(X):
        ftd = np.array(X.todense(),order='C').astype('float32')
    else:
        ftd = np.array(X,order='C').astype('float32')
    faiss.normalize_L2(ftd)

    if config.approx_knn:

        index =faiss.read_index(f'INDEX/{config.dataset}.index')

    else:
        index = faiss.index_factory(ftd.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(ftd)
    
    gpu_index = faiss.index_cpu_to_gpu(res, 0 , index, co)

    for test_time in range(times):
    
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()
        start_gpu.record()
        start_time = time.time()

        if config.gpu_usage:
            outfile = f"profile/{config.dataset}_{config.data}_{test_time}.csv"
            if config.approx_knn:
                ms = 500
            else:
                ms = 1
            cmd = "nvidia-smi -i " +str(switch_gpu)+ " -lms " +str(ms) +" --query-gpu=timestamp,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,clocks.current.graphics,clocks.current.sm --format=csv |tee " + outfile
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid) 

        orth_temp = dis_temp = mhc_temp= 0

        if config.approx_knn and config.beta<1:
            mask = np.ones(P[0].shape[0])
            mask[np.argwhere(X.sum(1)==0)[:,0]]*=(1./(1-beta))
            P1 = [sp.diags(mask)@P[0], P[1]]
        
        predict_clusters_best=None
        iter_best = 0
        conductance_best=100
        conductance_best_acc = [0]*3
        err = 1
        conductance_stats = []

        t0 = time.time()
        distances, neighbors = gpu_index.search(ftd, config.knn_k+1)       
        knn = sp.csr_matrix((distances.ravel(), neighbors.ravel(), np.arange(0, neighbors.size+1, neighbors.shape[1])), shape=(num_node,num_node))
        knn.setdiag(0.0)
        knn = knn + knn.T #A_k
        Q = normalize(knn, norm='l1')
        t1= time.time()
        knn_time.append(t1-t0)

        if beta>0.0 and config.network_type=="HG":
            unconnected = np.asarray(P[1].sum(0)).flatten()==0
            Q[unconnected, :] *= (1./beta)

        # memcpy data to cupy supported matrix on GPU
        t2=time.time()
        Q_g = csr_matrix(Q)
        P0_g = csr_matrix(P[0])
        if config.network_type =="HG":
            P1_g = csr_matrix(P[1])
        stamp1=time.time()
        copy_time.append(stamp1-t2)
    
        t3=time.time()
        topk_deg_nodes = heapq.nlargest(int(num_cluster), deg_dict, key=deg_dict.get)
        if config.network_type == "HG":
            PC = P0_g@P1_g[:,topk_deg_nodes]
        else:
            PC = P0_g[:,topk_deg_nodes]
        M = PC
        for i in range(config.init_iter):
            if config.network_type == "HG":
                M = (1-alpha)*P0_g@(P1_g.dot(M))+PC
            else:
                M = (1-alpha)*P0_g.dot(M)+PC
        class_evdsum = M.sum(axis=0)[0]
        newcandidates = np.argpartition(class_evdsum.get(), -num_cluster)[-num_cluster:]
        M = M[:,newcandidates]
        labels = M.argmax(axis=1).flatten()
        
        M = csc_matrix((cp.ones(len(labels)), (cp.arange(0, M.shape[0]), labels)),shape=(M.shape))
        Mss = cp.sqrt(M.sum(axis=0))
        Mss[Mss==0]=1
        q = M*1.0/Mss
        e1 = cp.ones(shape = (n,1))
        q = cp.hstack([e1,q])
        stamp2 = time.time()
        init_time.append(stamp2-t3)

        if config.approx_knn:
            P0_g = csr_matrix(P1[0])
            
        # orthogonal iterations
        for i in range(tmax):

            stamp3 = time.time()
            if config.network_type =="HG":
                z = (1-beta)*P0_g@(P1_g.dot(q))+ (beta)*Q_g.dot(q)
            else:
                z = (1-beta)*P0_g.dot(q)+ (beta)*Q_g.dot(q)

            if i == 0:
                if config.dataset.startswith('20news') or config.dataset.startswith('query'):
                    z = cp.around(z,14)
                else:
                    z = cp.around(z,15)

            q_prev = q
            q, r = cLA.qr(z, mode='reduced')
            err = cLA.norm(q-q_prev)/cLA.norm(q)
            stamp4 = time.time()
            orth_temp += stamp4-stamp3

            if (i+1)%config.cluster_interval==0:

                leading_eigenvectors = q[:,1:num_cluster+1]

                stamp5 =time.time()
                predict_clusters, y = gdiscretize(leading_eigenvectors)
                stamp6 =time.time()
                dis_temp +=stamp6-stamp5

                z_0 = config.alpha * y
                z = z_0
                for j in range(config.num_hop):
                    if config.network_type =="HG":
                        z = (1-config.alpha)*((1-beta)*P0_g@(P1_g.dot(z))+ (beta)*Q_g.dot(z)) + z_0
                    else:
                        z = (1-config.alpha)*((1-beta)*P0_g.dot(z)+ (beta)*Q_g.dot(z)) + z_0
                ct=y.T@z
                conductance_cur = 1.0-cp.trace(ct)/num_cluster
                stamp7 = time.time()
                mhc_temp +=stamp7-stamp6

                conductance_stats.append(conductance_cur)
                    
                if conductance_cur<conductance_best:
                    conductance_best = conductance_cur
                    predict_clusters_best = predict_clusters
                    iter_best = i

                if config.cond_early_stop and early_stop(conductance_stats):
                    break

                if err <= config.q_epsilon:
                    break

        orth_time.append(orth_temp)
        dis_time.append(dis_temp)
        mhc_time.append(mhc_temp)    
        end_time = time.time()
        end_gpu.record()
        end_gpu.synchronize()
        gpu_time = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        peak_memory=0
        peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        if config.gpu_usage:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            import pandas as pd
            df = pd.read_csv(outfile,encoding="utf-8")
            gpu_utilization = np.array(df[' utilization.gpu [%]'].str.extract('(\d+)')).astype(np.float32).flatten()
            gpu_usage.append(np.mean(gpu_utilization))
            gpu_clock =np.array(df[' clocks.current.graphics [MHz]'].str.extract('(\d+)')).astype(np.float32).flatten()
            gpu_flops = (gpu_utilization/1700)*gpu_clock
            flops.append(np.mean(gpu_flops))
        
        predict_clusters_best=cp.asnumpy(predict_clusters_best)
        cm = clustering_metrics(config.labels, predict_clusters_best)
        acc, nmi, f1, pre, adj_s, rec = cm.evaluationClusterModelFromLabel()

        if config.caltime and test_time == times-1:
            print("knn time: " + f"{np.mean(knn_time[1:])}" )
            print(knn_time)
            print("copy time: " + f"{np.mean(copy_time[1:])}" )
            print(copy_time)
            print("init time: " + f"{np.mean(init_time[1:])}" )
            print(init_time)
            print("orthogonal time: " + f"{np.mean(orth_time[1:])}" )
            print(orth_time)
            print("discretize time: " + f"{np.mean(dis_time[1:])}" )
            print(dis_time)
            print("calmhc time: " + f"{np.mean(mhc_time[1:])}" )
            print(mhc_time)
            if config.gpu_usage:
                print("gpu_utilization_avg: " + f"{np.mean(gpu_usage)} {np.mean(gpu_usage[1:])}" )
                print(gpu_usage)
                print("gpu_flops_percent: " + f"{np.mean(flops)} {np.mean(flops[1:])}" )
                print(flops)
        
        acc_list.append(acc)
        f1_list.append(f1)
        nmi_list.append(nmi)
        ari_list.append(adj_s)
        total_time.append(gpu_time/1000)
        pure_time.append(end_time-start_time-(stamp1-t2))

        if test_time == times-1:

            print("time:"+ f"{np.mean(total_time[1:])} {np.mean(pure_time[1:])}")
            print("mean:"+ f"{np.mean(acc_list)} {np.mean(f1_list)} {np.mean(nmi_list)} {np.mean(ari_list)}")
            print("std :"+ f"{np.std(acc_list)} {np.std(f1_list)} {np.std(nmi_list)} {np.std(ari_list)}")
            
            print(f"{acc} {f1} {nmi} {adj_s} {end_time-start_time} {peak_memory}")
            return [acc, nmi, f1, adj_s, np.mean(total_time),peak_memory]


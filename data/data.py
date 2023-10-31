import os, pickle
import numpy as np, scipy.sparse as sp
import pickle
import scipy.io as sio
import config

def load(dataset,data,type):
    if type=="Hypergraph":
        data_dict=load_hyper(data,dataset)
    elif type=="Undirected" or type=="Directed":
        data_dict=load_simple(type,dataset)
    else:
        if dataset == "acm":
            data_dict=load_acm()
        if dataset == "imdb":
            data_dict=load_imdb()
        if dataset == "dblp":
            data_dict=load_dblp()
    return data_dict

#load undirected npz
def load_npz_dataset(file_name):
    """Load a graph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'W' : The adjacency matrix in sparse matrix format
            * 'fea' : The attribute matrix in sparse matrix format
            * 'gnd' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        edge_index = loader['adj_indices'].copy()
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'W': A,
            'fea': X,
            'gnd': z
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph

#load directed and undirected graph
def load_simple(type,dataset):
    path_directed = 'data/directed/'
    path_undirected = 'data/undirected/'
    if type=='Undirected':
        print(f'===== loading undirected {dataset} =====')
        data = sio.loadmat('{}{}.mat'.format(path_undirected, dataset))
        feature = data['fea']

    elif type=='Directed':
        print(f'===== loading directed {dataset} =====')
        data = load_npz_dataset('{}{}.npz'.format(path_directed, dataset))
        feature = data['fea']
        if sp.issparse(feature):
            feature = feature.todense()
        feature=sp.csr_matrix(feature)

    adj = sp.csr_matrix(data['W'])
    if type=='Directed':
        adj = adj + adj.T
    adj.data[adj.data>0]=1.0
    diagonal_indices = (np.arange(feature.shape[0]), np.arange(feature.shape[0]))
    adj[diagonal_indices] = 0.0

    labels = data['gnd']
    if type == 'Undirected':
        labels = labels.T
        labels = labels - 1
        labels = labels[0, :]

    data_dict = {'features_sp': feature, 'labels': labels, 'n': feature.shape[0], 'adj_sp': adj}

    return data_dict

#load hyper graph
def load_hyper(data, dataset):

    if data == 'npz':
        data_dict = load_npz(dataset)
    else:
        ps = parser(data, dataset)

        with open(os.path.join(ps.d, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)

        with open(os.path.join(ps.d, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle)

        with open(os.path.join(ps.d, 'labels.pickle'), 'rb') as handle:
            labels = ps._1hot(pickle.load(handle))

        adj = np.zeros((len(hypergraph), features.shape[0]))
        for index, edge in enumerate(hypergraph):
            hypergraph[edge] = list(hypergraph[edge])
            adj[index, hypergraph[edge]] = 1
        if config.remove_unconnected:
            nonzeros = adj.sum(0).nonzero()[0]
            adj = adj[:, nonzeros]
            features = features[nonzeros, :]
            labels = labels[nonzeros, :]
            pairs = adj.nonzero()
            hypergraph = {}
            for index, edge in enumerate(pairs[0]):
                if edge not in hypergraph:
                    hypergraph[edge]=[]
                hypergraph[edge].append(pairs[1][index])
        adj_sp = sp.csr_matrix(adj)
        data_dict = {'features': features.todense(), 'features_sp':features, 'labels': labels, 'n': features.shape[0], 'e': len(hypergraph), 'name': dataset, 'adj': adj, 'adj_sp': adj_sp}
    return data_dict

def load_npz(dataset):
    hg_adj = sp.load_npz(f'data/npz/{dataset}/hypergraph.npz')
    np.clip(hg_adj.data, 0, 1, out=hg_adj.data)
    features = sp.load_npz(f'data/npz/{dataset}/features.npz')
    labels = np.load(f'data/npz/{dataset}/labels.npy')
    return {'features_sp': features, 'labels': labels, 'n': features.shape[0], 'e': hg_adj.shape[0], 'name': dataset, 'adj': hg_adj, 'adj_sp': hg_adj}

def load_acm():
    feature = 'feature'
    path = 'data/acm/'
    feature = sio.loadmat('{}{}.mat'.format(path, feature))
    adj1 = 'PAP'
    adj2 = 'PLP'

    adj1 = sio.loadmat('{}{}.mat'.format(path, adj1))
    adj2 = sio.loadmat('{}{}.mat'.format(path, adj2))

    adj1 = sp.coo_matrix(adj1['PAP'])
    adj2 = sp.coo_matrix(adj2['PLP'])

    feature = feature['feature']

    list_of_adj = [adj1, adj2]

    lines = 'ground_truth'
    gt = []
    with open('{}{}.txt'.format(path, lines)) as f:
        lines = f.readlines()
    for line in lines:
        gt.append(int(line))

    gt = np.array(gt)

    data_dict={'features_sp': feature, 'labels': gt, 'n': feature.shape[0], 'adj_sp': list_of_adj}
    return data_dict

def load_imdb():
    path = 'data/imdb/'
    dataset = 'imdb'
    ids = 'ids'

    adj = sio.loadmat('{}{}.mat'.format(path, dataset))
    ids = sio.loadmat('{}{}.mat'.format(path, ids))

    adj1 = sp.coo_matrix(adj['MDM'])
    adj2 = sp.coo_matrix(adj['MAM'])

    feature = adj['feature']

    list_of_adj = [adj1, adj2]

    lines = 'ground_truth'
    gt = []
    count = 0
    with open('{}{}.txt'.format(path, lines)) as f:
        lines = f.readlines()

    for line in lines:
        count += 1
        if isinstance(line, str):
            gt.append(int(line))
        else:
            gt.append(line)
    gt = np.array(gt)
    data_dict = {'features_sp': feature, 'labels': gt, 'n': feature.shape[0], 'adj_sp': list_of_adj}
    return data_dict

def load_dblp():
    path = 'data/dblpAttributed/'
    adj1 = 'apa'
    adj2 = 'apcpa'
    adj3 = 'aptpa'
    feature = 'a_feat'
    gt = 'labels'

    adj1 = sp.load_npz('{}{}.npz'.format(path, adj1))
    adj2 = sp.load_npz('{}{}.npz'.format(path, adj2))
    adj3 = sp.load_npz('{}{}.npz'.format(path, adj3))

    list_of_adj = [adj1, adj2, adj3]

    feature = sp.load_npz('{}{}.npz'.format(path, feature))

    gt = np.load('{}{}.npy'.format(path, gt)).astype('int32')

    data_dict={'features_sp': feature, 'labels': gt, 'n': feature.shape[0], 'adj_sp': list_of_adj}
    return data_dict

class parser(object):

    def __init__(self, data, dataset):

        import inspect
        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.d = os.path.join(current, data, dataset)
        self.data, self.dataset = data, dataset

    

    def parse(self):
        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()



    def _load_data(self):

        with open(os.path.join(self.d, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)

        with open(os.path.join(self.d, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle).todense()

        with open(os.path.join(self.d, 'labels.pickle'), 'rb') as handle:
            labels = self._1hot(pickle.load(handle))

        return {'hypergraph': hypergraph, 'features': features, 'labels': labels, 'n': features.shape[0]}



    def _1hot(self, labels):
        classes = set(labels)
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)
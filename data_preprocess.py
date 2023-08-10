import os
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
import torch as t
import torch_geometric
torch_geometric.typing.WITH_PYG_LIB = False
from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
import config_load

from utils import *

EPS = 1e-8

def arg_parse():
    parser = argparse.ArgumentParser(description="Data Preprocess.")
    parser.add_argument('-d', '--data_dir', type=str, default='data/Breast_Cancer_Matrix',)
    parser.add_argument('-op', action='store_true', default=False,)
    parser.add_argument('-n', '--n2v', dest='n2v', action='store_true')
    return parser.parse_args()


def get_ppi_mat(ppi_name='CPDB', drop_rate=0.0, from_list=False, pan=False, random_seed=42):
    """
    Read PPI data from a csv file and construct PPI adjacent matrix.

    Parameters:
    ----------
    ppi_name:   str, {'CPDB', 'IRef', 'Multinet', 'PCNet', 'STRING'}, default='CPDB'. 
                Name of PPI network dataset. Corresponding matrix should be put into certain directory first.
    drop_rate:  float, default=0.0. 
                Drop rate for robustness study.
    from_list:  bool, default=False.
                Whether the PPI data is loaded from a preprocessed adjacency list instead of a matrix.

    Returns:
    numpy.ndarray
        A ndarray(num_nodes, num_nodes) contains PPI adjacent matrix.
    """
    prefix = 'pan_data' if pan else 'data'
    # Load PPI data from an edge list
    if from_list:
        ppi_dir = f"{prefix}/{ppi_name}/{ppi_name}_edgelist.csv"
        print(f"Loading PPI matrix from {ppi_dir} ......")
        data = pd.read_csv(
            f"{prefix}/{ppi_name}/{ppi_name}_edgelist.csv", sep='\t')
        # Load the gene names
        gene_list, gene_set = get_all_nodes(pan=pan)

        # Extract the edges that are also in the list of gene names
        adj = [(row[1], row[2], row[3]) for row in data.itertuples()
               if row[1] in gene_set and row[2] in gene_set]
        conf = [row[4] for row in data.itertuples() if row[1]
                in gene_set and row[2] in gene_set]
        if drop_rate:
            # Drop samples with stratification by confidence score
            adj, drop_adj = train_test_split(
                adj, test_size=drop_rate, random_state=random_seed, stratify=conf)
        # Construct the adjacency matrix from the edges
        adj_matrix = pd.DataFrame(0, index=gene_list, columns=gene_list)
        for line in adj:
            adj_matrix.loc[line[0], line[1]] = line[2]
            adj_matrix.loc[line[1], line[0]] = line[2]
        print(f'Saving ppi matrix to {prefix}/{ppi_name}/{ppi_name}_matrix.csv ......')
        adj_matrix.to_csv(f'{prefix}/{ppi_name}/{ppi_name}_matrix.csv', sep='\t')
        data = adj_matrix.to_numpy().astype(float)

        return data

    # Load PPI data from a matrix
    ppi_dir = f"{prefix}/{ppi_name}/{ppi_name}_matrix.csv"
    print(f"Loading PPI matrix from {ppi_dir} ......")
    data = pd.read_csv(ppi_dir, sep="\t").to_numpy()[:, 1:].astype(float)

    return data


def get_label(data_dir='data/Breast_Cancer_Matrix'):
    """
    Read label data, where some nodes have labels and others do not. For the nodes with labels, change the labels from -1 to 0.

    Returns:
    labels:         ndarray(num_nodes, 2). First col is 1 for negative nodes, and second col is 1 for positive nodes.
    labeled_idx:    list(num_nodes). Indices of labeled nodes.
    """
    cell_line = get_cell_line(data_dir)
    data = read_table_to_np(os.path.join(data_dir, cell_line +
                            "-Label.txt"), dtype=int).transpose()[0]
    labeled_idx = []
    labels = np.zeros((len(data), 2), dtype=float)
    for i in range(len(data)):
        if data[i] != 0:
            labeled_idx.append(i)
            if data[i] == -1:
                labels[i][0] = 1
            else:
                labels[i][1] = 1
    return labels, labeled_idx


def get_node_feat(data_dir='data/Breast_Cancer_Matrix', n2v=False):
    """
    Read marker node features from a csv file.

    Parameters:
    ----------

    Returns:
    feat:       Ndarray(num_nodes, num_features). Node features.
    pos:        Ndarray(num_nodes,). Node indices.
    """
    cell_line = get_cell_line(data_dir)
    feat = read_table_to_np(os.path.join(data_dir, cell_line + "-Normalized-Nodefeature-Matrix.csv"), sep=',')
    feat = feat if cell_line == 'Pan' else feat[:, :10]
    if n2v: 
        n2v_feat = read_table_to_np(os.path.join(data_dir, "N2V_embedding_10.csv"), sep='\t')
        feat = np.concatenate((feat, n2v_feat), axis=1)
    print(f"Feature matrix shape: {feat.shape}")
    pos = np.arange(feat.shape[0])
    return feat, pos


def construct_edge(mat, weighted=False):
    """
    Construct edges from adjacent matrix.

    Parameters:
    ----------
    ppi_mat:    ndarray(num_nodes, num_nodes).
                PPI matrix from get_ppi_mat().

    Returns:
    edges:      list(num_edges, 2). 
    edge_dim:   int.
                Dim of edge features.
    val:        list(num_edges, ).
                Edge features(=[1] * num_edges in current version).
    """
    num_nodes = mat.shape[0]
    edges = []
    val = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if mat[i, j] > 0:
                edges.append([i, j])
                if weighted:
                    val.append(mat[i, j])
                else:
                    val.append(1)

    edge_dim = 1
    edges = np.transpose(edges)
    val = np.reshape(val, (-1, edge_dim))

    return edges, edge_dim, val


def build_pyg_data(node_mat, node_lab, mat, pos):
    x = t.tensor(node_mat, dtype=torch.float)
    y = t.tensor(node_lab, dtype=torch.long)
    pos = t.tensor(pos, dtype=torch.int)
    edge_index, edge_dim, edge_feat = construct_edge(mat, weighted=False)
    edge_index = t.tensor(edge_index, dtype=torch.long)
    edge_feat = t.tensor(edge_feat, dtype=torch.float)
    data = Data(x=x.clone(), y=y.clone(), edge_index=edge_index,
                edge_attr=edge_feat, pos=pos, edge_dim=edge_dim)
    print(
        f"Number of edges: {data.num_edges}, Dimensionality of edge: {edge_dim},\nNubmer of nodes: {data.num_nodes}")

    return data


class CancerDataset(InMemoryDataset):
    def __init__(self, data_list=None):
        super(CancerDataset, self).__init__('.', None, None)
        self.data_list = data_list
        self._data, self.slices = self.collate(self.data_list)
        self.num_slices = len(self.data_list)
        self.num_nodes = self.data_list[0].num_nodes
        self._data.num_classes = 2

    def get_idx_split(self, i):
        train_idx = torch.where(self.data_list[0].train_mask[:, i])[0]
        test_idx = torch.where(self.data_list[0].test_mask[:, i])[0]
        valid_idx = torch.where(self.data_list[0].valid_mask[:, i])[0]

        return {
            'train': train_idx,
            'test': test_idx,
            'valid': valid_idx
        }

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    

def create_cv_dataset(train_idx_list, valid_idx_list, test_idx_list, data_list):
    num_nodes = data_list[0].num_nodes
    num_folds = len(train_idx_list)

    train_mask = np.zeros((num_nodes, num_folds), dtype=bool)
    valid_mask = np.zeros((num_nodes, num_folds), dtype=bool)
    test_mask = np.zeros((num_nodes, num_folds), dtype=bool)
    for i in range(num_folds):
        train_mask[train_idx_list[i], i] = True
        valid_mask[valid_idx_list[i], i] = True
        test_mask[test_idx_list[i], i] = True

    for data in data_list:
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        data.unlabeled_mask = ~torch.logical_or(
            data.train_mask[:, 0], torch.logical_or(data.valid_mask[:, 0], data.test_mask[:, 0]))
    cv_dataset = CancerDataset(data_list=data_list)

    return cv_dataset


def get_data(configs, stable=True):
    PPI = configs['PPI']
    cv_folds = configs["cv_folds"]
    data_dir = configs["data_dir"]
    load_data = configs["load_data"]
    pre_drop_rate = configs["pre_drop_rate"]
    random_seed = configs["random_seed"]
    pan = 'Pan' in configs['data_dir']

    cell_line = get_cell_line(data_dir)

    def get_dataset_dir(overlap, n2v, stable):
        dataset_suffix = "_dataset_final" if stable else "_dataset"
        if overlap:
            dataset_suffix = '_op' + dataset_suffix
        if n2v:
            dataset_suffix = '_n2v' + dataset_suffix

        dataset_dir = os.path.join(
            data_dir, cell_line + dataset_suffix + '.pkl')
        
        return dataset_dir

    if load_data:
        dataset_dir = get_dataset_dir(configs['overlap'], configs['n2v'], stable)
        print(f"Loading dataset from: {dataset_dir} ......")
        with open(dataset_dir, 'rb') as f:
            cv_dataset = pickle.load(f)
            
        return cv_dataset
    
    if configs['overlap']:
        ppi_mat_list = None
        for ppi in PPI:
            if ppi_mat_list is None:
                ppi_mat_list = get_ppi_mat(ppi, drop_rate=pre_drop_rate, from_list=False, pan=pan, random_seed=random_seed)
            else:
                ppi_mat_list += get_ppi_mat(ppi, drop_rate=pre_drop_rate, from_list=False, pan=pan, random_seed=random_seed)
        ppi_mat_list = [ppi_mat_list]
    else:
        ppi_mat_list = [get_ppi_mat(ppi, drop_rate=pre_drop_rate, from_list=False, pan=pan, random_seed=random_seed) for ppi in PPI]
    node_mat, pos = get_node_feat(data_dir=data_dir, n2v=configs['n2v'])
    node_lab, labeled_idx = get_label(data_dir)
    labeled_lab = [node_lab[i][1] for i in labeled_idx]

    train_idx_list, valid_idx_list, test_idx_list = [], [], []
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    for train_labeled_idx, test_labeled_idx in skf.split(labeled_idx, labeled_lab):
        test_idx_list.append([labeled_idx[i] for i in test_labeled_idx])
        train_valid_idx = [labeled_idx[i] for i in train_labeled_idx]
        train_valid_lab = [labeled_lab[i] for i in train_labeled_idx]
        train_idx, valid_idx = train_test_split(
            train_valid_idx, test_size=0.125, stratify=train_valid_lab, random_state=random_seed)
        train_idx_list.append(train_idx)
        valid_idx_list.append(valid_idx)

    data_list = [build_pyg_data(node_mat, node_lab, ppi_mat, pos) for ppi_mat in ppi_mat_list]
    cv_dataset = create_cv_dataset(
        train_idx_list.copy(), valid_idx_list.copy(), test_idx_list.copy(), data_list)

    dataset_dir = get_dataset_dir(overlap=configs['overlap'], n2v=configs['n2v'], stable=False)
    print(f'Finished! Saving dataset to {dataset_dir} ......')
    with open(dataset_dir, 'wb') as f:
        pickle.dump(cv_dataset, f)

    return cv_dataset


if __name__ == "__main__":
    configs = config_load.get()
    args = arg_parse()
    configs['data_dir'] = args.data_dir
    configs['overlap'] = args.op
    configs["load_data"] = False
    configs['n2v'] = args.n2v
    data = get_data(configs)
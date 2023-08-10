import torch
import numpy as np
import pandas as pd
import pickle
import argparse
import os

from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.nn import Node2Vec
from torch_geometric.data import InMemoryDataset

import config_load
import data_preprocess as dp
from utils import *

def arg_parse():
    parser = argparse.ArgumentParser(description="Train Graph Embedding")
    parser.add_argument('-e', "--embedding_dim", dest="dim", type=int, help="Dim of embedding")
    parser.add_argument('-d', "--data_dir", dest="dir", help="train PPI embedding", default="data/Breast_Cancer_Matrix")
    parser.add_argument('-g', '--gpu', dest='gpu', default=3)
    return parser.parse_args()

class CancerN2VDataset(InMemoryDataset):
    def __init__(self, data):
        super(CancerN2VDataset, self).__init__('.', None, None)
        self.data = data
        self.data.num_classes = 2

        self.data, self.slices = self.collate([self.data])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def get_dataset(configs, load=True):
    data_dir = configs["data_dir"]
    cell_line = get_cell_line(data_dir)
    dir = data_dir + f"/{cell_line}_Node2Vec_dataset.pkl"
    if load:
        print("Loading dataset from:", dir)
        with open(dir, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    
    adj = None
    for ppi in configs['PPI']:
        if adj is None:
            adj = dp.get_ppi_mat(ppi, from_list=False)
        else:
            adj += dp.get_ppi_mat(ppi, from_list=False)
    edge_index, _, _ = dp.construct_edge(adj)
    node_lab, labeled_idx = dp.get_label(data_dir)
    labeled_lab = [node_lab[i][1] for i in labeled_idx]
    y = torch.tensor(node_lab, dtype=torch.long)
    print(edge_index.shape)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(y=y, edge_index=edge_index, num_nodes=y.shape[0])
    train_idx, test_idx, _, _ = train_test_split(labeled_idx, labeled_lab, test_size=0.1, stratify=labeled_lab, random_state=42)
    num_nodes = adj.shape[0]
    train_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    for idx in train_idx:
        train_mask[idx] = True
    for idx in test_idx:
        test_mask[idx] = True
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    dataset = CancerN2VDataset(data)
    print('finished')
    f = open(dir, 'wb')
    pickle.dump(dataset, f)
    f.close()

    return dataset

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test(model):
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask, 1],
                     z[data.test_mask], data.y[data.test_mask, 1], max_iter=150)
    feat = z.cpu().detach().numpy()
    feat = dp.minmax(feat)
    return acc, feat

@torch.no_grad()
def plot_points(model, colors):
    model.eval()
    z = model(torch.arange(data.num_nodes, device=device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data.y[:, 1].cpu().numpy()
    num_classes = np.unique(y).size

    plt.figure(figsize=(8, 8))
    for i in range(num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.savefig("node2vec.png", transparent=False)

def get_node2vec_feat(data, epochs):
    model = Node2Vec(data.edge_index, embedding_dim=args.dim, walk_length=80, context_size=20, walks_per_node=10, num_negative_samples=3, num_nodes=data.num_nodes, p=1, q=2, sparse=True).to(device)
    loader = model.loader(batch_size=256, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    for epoch in range(1, epochs+1):
        loss = train(model, loader, optimizer)
        acc, feat = test(model)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    return feat, model

if __name__ == "__main__":
    configs = config_load.get() 
    args = arg_parse()
    configs["data_dir"] = args.dir
    device = f'cuda:{args.gpu}'
    data = get_dataset(configs, load=True)[0]
    feat, model = get_node2vec_feat(data, epochs=100)

    columns = [f'PPI-{i}' for i in range(model.embedding_dim)]
    gene_name = get_node_name([i for i in range(data.num_nodes)])
    feat = pd.DataFrame(feat, columns=columns, index=gene_name)
    feat.to_csv(os.path.join(configs["data_dir"] + f"N2V_embedding_{model.embedding_dim}.csv"), sep='\t')

    colors = ['#ffc0cb', '#bada55']
    plot_points(model, colors)
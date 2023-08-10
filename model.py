import torch_geometric
torch_geometric.typing.WITH_PYG_LIB = False

import torch as t
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv, GCNConv, GATConv, ChebConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import dropout_edge, negative_sampling, remove_self_loops, add_self_loops
from torch.nn import Dropout, MaxPool1d, AvgPool1d

from sklearn.svm import SVC


HIDDEN_DIM = 32
LEAKY_SLOPE = 0.2


class GTN(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, pooling, residual):
        super(GTN, self).__init__()
        self.drop_rate = drop_rate
        self.pooling = pooling
        self.residual = residual
        self.convs = t.nn.ModuleList()
        mid_channels = in_channels + hidden_channels if residual else hidden_channels
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=attn_drop_rate, edge_dim=edge_dim,
                                          concat=False, beta=True))
        self.ln1 = LayerNorm(in_channels=mid_channels)
        if pooling:
            self.convs.append(TransformerConv(mid_channels, hidden_channels, heads=heads,
                                            dropout=attn_drop_rate, edge_dim=edge_dim, concat=True, beta=True))
            self.ln2 = LayerNorm(in_channels=hidden_channels * heads // 2)
            self.pool = MaxPool1d(2, 2) if pooling == 'max' else AvgPool1d(2, 2) if pooling == 'avg' \
                        else Linear(hidden_channels * heads, hidden_channels * heads // 2)
        else:
            self.convs.append(TransformerConv(mid_channels, hidden_channels // 2, heads=heads,
                                            dropout=attn_drop_rate, edge_dim=edge_dim, concat=True, beta=True))
            self.ln2 = LayerNorm(in_channels=hidden_channels * heads // 2)

    def forward(self, data):
        x = data.x
        edge_index, edge_mask = dropout_edge(data.edge_index, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        edge_attr = data.edge_attr[edge_mask]
        res = x * self.residual
        x = F.leaky_relu(self.convs[0](x, edge_index, edge_attr), negative_slope=LEAKY_SLOPE, inplace=True)
        x = t.cat((x, res), dim=1) if self.residual else x
        x = self.ln1(x)
        edge_index, edge_mask = dropout_edge(data.edge_index, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        edge_attr = data.edge_attr[edge_mask]

        x = self.convs[1](x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE)
        x = t.squeeze(self.pool(t.unsqueeze(x, 1)), dim=1) if self.pooling else x
        x = self.ln2(x)
        
        return x[:data.batch_size]


class Multi_GTN(t.nn.Module):
    def __init__(self, gnn, in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, num_ppi, pooling, residual, learnable_weight, ):
        super(Multi_GTN, self).__init__()

        self.convs = t.nn.ModuleList()
        for _ in range(num_ppi):
            if 'GTN' in gnn:
                self.convs.append(GTN(in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, pooling, residual))
            # elif 'GAT' in gnn:
            #     self.convs.append(GAT(in_channels, hidden_channels, heads, attn_drop_rate, edge_dim=edge_dim))
            # elif 'GCN' in gnn:
            #     self.convs.append(GCN(in_channels, hidden_channels, attn_drop_rate))
            # elif 'MLP' in gnn:
            #     self.convs.append(MLP(drop_rate, in_channels, hidden_channels))

        if learnable_weight:
            self.ppi_weight = t.nn.ParameterList([t.nn.Parameter(t.Tensor(1, 1)) for _ in range(num_ppi)])
            for weight in self.ppi_weight:
                t.nn.init.constant_(weight, 1)
        else: self.ppi_weight = t.ones(num_ppi, 1)

        self.lins = t.nn.ModuleList()
        self.lins.append(Linear(int(num_ppi * hidden_channels * heads / 2), HIDDEN_DIM,
                        weight_initializer="kaiming_uniform"))
        self.dropout = Dropout(drop_rate)
        self.lins.append(Linear(HIDDEN_DIM, 1, weight_initializer="kaiming_uniform"))
        
    def forward(self, data_tuple):
        x_list = [self.convs[i](data) for i, data in enumerate(data_tuple)]
        x = t.cat(x_list, dim=1)
        x = self.lins[0](x).relu()
        x = self.dropout(x)
        x = self.lins[1](x)

        return t.sigmoid(x), x_list, self.ppi_weight
    

class SVM(t.nn.Module):
    def __init__(self, C=1.0, gamma='auto', kernel='rbf'):
        super().__init__()
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.svm = SVC(C=C, gamma=gamma, kernel=kernel, probability=True)

    def forward(self, data):
        data = data[0]
        return self.svm.predict(data.x)


class MLP(t.nn.Module):
    def __init__(self, drop_rate, in_channels, hidden_channels):
        super(MLP, self).__init__()
        self.dropout = Dropout(drop_rate)
        self.lins = t.nn.ModuleList()
        self.lins.append(
            Linear(in_channels, hidden_channels, weight_initializer="kaiming_uniform"))
        self.lins.append(
            Linear(hidden_channels, 1, weight_initializer="kaiming_uniform"))

    def forward(self, x):
        x = x[0].x if isinstance(x, tuple) else x
        x = self.lins[0](x).relu()
        x = self.dropout(x)
        return self.lins[1](x)
    

class GCN(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, drop_rate, residual):
        super(GCN, self).__init__()
        self.drop_rate = drop_rate
        self.residual = residual
        mid_channels = in_channels + hidden_channels if residual else hidden_channels

        self.convs = t.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, improved=False))
        self.convs.append(GCNConv(mid_channels, 1, improved=False))

    def forward(self, data):
        data = data[0]
        x = data.x
        res = x * self.residual
        x = self.convs[0](x, data.edge_index).relu()
        x = t.cat((x, res), dim=1) if self.residual else x
        
        return t.sigmoid(self.convs[1](x, data.edge_index))[:data.batch_size]
    

class GAT(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, drop_rate, edge_dim, residual):
        super(GAT, self).__init__()
        self.residual = residual
        mid_channels = in_channels + hidden_channels if residual else hidden_channels

        self.convs = t.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=drop_rate, edge_dim=edge_dim, concat=False))
        self.convs.append(GATConv(mid_channels, hidden_channels, heads=heads, dropout=drop_rate, edge_dim=edge_dim, concat=True))

        self.lins = MLP(drop_rate=drop_rate, in_channels=hidden_channels*heads, hidden_channels=hidden_channels)

    def forward(self, data):
        data = data[0]
        x = data.x
        res = x * self.residual
        x = self.convs[0](x, data.edge_index, data.edge_attr)
        x = t.cat((x, res), dim=1) if self.residual else x
        x = self.convs[1](x, data.edge_index, data.edge_attr)
        x = self.lins(x)[:data.batch_size]

        return t.sigmoid(x)
    
    
# according to https://github.com/Bibyutatsu/proEMOGI/blob/main/proEMOGI/proemogi.py
class EMOGI(t.nn.Module):
    def __init__(self, in_channels, drop_rate=0.5, hidden_dims=[20, 40], residual=0.):
        super(EMOGI, self).__init__()
        self.in_channels = in_channels

        # model params
        self.num_hidden_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims
        self.drop_rate = drop_rate
        self.residual = residual
        
        self.convs = t.nn.ModuleList()

        # add intermediate layers
        inp_dim = self.in_channels
        for l in range(self.num_hidden_layers):
            self.convs.append(GCNConv(inp_dim,
                                       self.hidden_dims[l]))
            inp_dim = self.hidden_dims[l] + self.in_channels if self.residual else self.hidden_dims[l]
            
        self.convs.append(GCNConv(inp_dim, 1))
        
    def forward(self, data):
        data = data[0]
        x = data.x
        res = x * self.residual
        for layer in self.convs[:-1]:
            x = layer(x, data.edge_index)
            x = t.cat((x, res), dim=1) if self.residual else x
            x = F.relu(x)
            if self.drop_rate is not None:
                x = F.dropout(x, self.drop_rate, training=self.training)
        x = self.convs[-1](x, data.edge_index)
        return t.sigmoid(x)[:data.batch_size]
    

class MTGCN(t.nn.Module):
    def __init__(self, in_channels, hidden_dims, residual):
        super(MTGCN, self).__init__()
        self.residual = residual
        mid_channels = in_channels + hidden_dims[0] if residual else hidden_dims[0]

        self.conv1 = ChebConv(in_channels, hidden_dims[0], K=2, normalization="sym")
        self.conv2 = ChebConv(mid_channels, hidden_dims[1], K=2, normalization="sym")
        self.conv3 = ChebConv(hidden_dims[1], 1, K=2, normalization="sym")

        self.lin1 = Linear(in_channels, 100)
        self.lin2 = Linear(in_channels, 100)

        self.c1 = t.nn.Parameter(t.Tensor([0.5]))
        self.c2 = t.nn.Parameter(t.Tensor([0.5]))

    def forward(self, data):
        data = data[0]
        res = data.x * self.residual
        edge_index, _ = dropout_edge(data.edge_index, p=0.5, force_undirected=True, training=self.training)

        x0 = F.dropout(data.x, training=self.training)
        x = t.relu(self.conv1(x0, edge_index))
        x = t.cat((x, res), dim=1) if self.residual else x
        x = F.dropout(x, training=self.training)
        x1 = t.relu(self.conv2(x, edge_index))

        x = x1 + t.relu(self.lin1(x0))
        z = x1 + t.relu(self.lin2(x0))

        pb, _ = remove_self_loops(data.edge_index)
        pb, _ = add_self_loops(pb)
        E = data.edge_index

        pos_loss = -t.log(t.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15).mean()
        neg_edge_index = negative_sampling(pb, data.x.shape[0], data.edge_index.shape[1])
        neg_loss = -t.log(
            1 - t.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean() if neg_edge_index.numel() != 0 else 0

        r_loss = pos_loss + neg_loss


        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return (x, r_loss, self.c1, self.c2)
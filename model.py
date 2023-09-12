import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, GraphConv, GATConv
import dgl.function as fn

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, dropout = 0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.dropout = nn.Dropout(dropout)
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = self.dropout(h) #dropout before relu
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, dropout = 0.5):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree = True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = GraphConv(h_feats, h_feats, allow_zero_in_degree = True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = self.dropout(h) #dropout before relu
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, dropout = 0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, 1, allow_zero_in_degree = True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = GATConv(h_feats, h_feats, 1, allow_zero_in_degree = True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = h.reshape(h.shape[0],h.shape[2]) #1 attention head
        h = self.dropout(h) #dropout before relu
        h = F.relu(h)
        h = self.conv2(g, h)
        h = h.reshape(h.shape[0],h.shape[2]) #1 attention head
        return h
        
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
        self.sig = nn.Sigmoid()

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.sig(self.W2(F.relu(self.W1(h)))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
        
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']
        
        
class HeteroMLPPredictor(nn.Module):
    def __init__(self, h_feats, edge_types, dropout = 0.5):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.dropout = nn.Dropout(dropout)
        self.W2 = nn.Linear(h_feats, edge_types)
        self.sig = nn.Sigmoid()
    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.sig(self.W2(F.relu(self.dropout(self.W1(h)))))} # dim: edge_types

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
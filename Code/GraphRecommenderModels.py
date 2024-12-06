# -*- coding: utf-8 -*-
"""

"""
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import dgl
from Demo_SEC_API_Ben import SecAPIDataset
from dgl.data import DGLDataset
import scipy.sparse as sp

class SecAPIGraphDataset(DGLDataset):
    def __init__(self, asset_freq_threshold, portfolio_asset_threshold, mask_ratio=0.2, num_asset_freq_buckets=10):
        self.portfolio_asset_threshold = portfolio_asset_threshold
        self.asset_freq_threshold = asset_freq_threshold
        self.mask_ratio = mask_ratio
        self.num_asset_freq_buckets = num_asset_freq_buckets
        super().__init__(name="sec_api")

    def process(self):
        dataset = SecAPIDataset(asset_freq_threshold=self.asset_freq_threshold, portfolio_asset_threshold=self.portfolio_asset_threshold)
        portfolio_data = dataset.portfolios
        portfolio_indices = torch.tensor(pd.factorize(portfolio_data['FUND_ID'].values)[0])
        portfolio_features = torch.ones_like(portfolio_indices.unique()).unsqueeze(1)   #for now no features of portfolio
        asset_indices = torch.tensor(portfolio_data['identifiers.isin.value'].values)
        asset_features = torch.tensor(portfolio_data.groupby('identifiers.isin.value').first().values.astype(float)[:,2:])  #we have different features for the same asset, which should normally not be the case
        
        self.graph = dgl.heterograph({
            ('portfolio', 'includes', 'asset'): (portfolio_indices, asset_indices)})
        self.graph.nodes['portfolio'].data['feat'] = portfolio_features
        self.graph.nodes['asset'].data['feat'] = asset_features
        # Add node types
        self.graph.nodes['portfolio'].data['type'] = torch.full((self.graph.num_nodes('portfolio'),), 0, dtype=torch.long)
        self.graph.nodes['asset'].data['type'] = torch.full((self.graph.num_nodes('asset'),), 1, dtype=torch.long)

        train_mask = dataset.val_mask == 0
        val_mask = dataset.val_mask == 1

        self.graph.edges['includes'].data['train_mask'] = train_mask
        self.graph.edges['includes'].data['val_mask'] = val_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    def get_split(self):
        u, v = self.graph.edges()
        val_mask = self.graph.edges['includes'].data['val_mask']
        train_mask = self.graph.edges['includes'].data['train_mask']
        train_pos_u, train_pos_v = u[train_mask], v[train_mask]
        val_pos_u, val_pos_v = u[val_mask], v[val_mask]

        # Find all negative edges and split them for training and testing
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy()))).todense()
        neg_u, neg_v = np.where(adj != 1)
        # Pick same amount of negative edges than positive edges (randomly, with new set for each call of this function)
        neg_eids = np.random.choice(len(neg_u), self.graph.number_of_edges())
        train_neg_u, train_neg_v = neg_u[neg_eids[train_mask]], neg_v[neg_eids[train_mask]]
        val_neg_u, val_neg_v = neg_u[neg_eids[val_mask], neg_v[neg_eids[val_mask]]

        train_graph = dgl.edge_subgraph(self.graph, self.graph.edges['includes'].data['train_mask'], relabel_nodes=False)
        val_graph = dgl.edge_subgraph(self.graph, self.graph.edges['includes'].data['val_mask'], relabel_nodes=False)
        return train_graph, val_graph
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import networkx as nx
    portfolio_asset_threshold = 5
    asset_freq_threshold = 25
    print('Loading Graph Dataset')
    dataset = SecAPIGraphDataset(asset_freq_threshold=asset_freq_threshold, portfolio_asset_threshold=portfolio_asset_threshold)
    g_train, g_val = dataset.get_split()
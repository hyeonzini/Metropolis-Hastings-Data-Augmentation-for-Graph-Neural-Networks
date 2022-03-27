import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, SGConv

import pdb
import math
import random
import numpy as np

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(
            SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_head, att_dropout):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, int(hidden_channels), heads=num_head, concat=True, dropout = att_dropout))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels*num_head, int(hidden_channels), heads=num_head, concat=True, dropout = att_dropout))
        self.convs.append(
            GATConv(hidden_channels*num_head, out_channels, heads=num_head, concat=True, dropout = att_dropout))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, use_bn=False):
        super(GCN, self).__init__()
        self.use_bn = use_bn
        self.convs = torch.nn.ModuleList()
        
        if self.use_bn:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        self.convs.append(
            GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
            if self.use_bn:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        self.convs.append(
            GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
        if self.bns:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index, get_h=False):
        for i, conv in enumerate(self.convs[:-1]):
            #x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            hidden = x
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        
        if get_h:
            return x, hidden
        else:
            return x


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(
            torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(
            torch.nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(
            nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout
    
    def reset_parameters(self):
        for layer in self.layers:
            layers.reset_parameters()
    
    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x
    
class AGG_NET(torch.nn.Module):
    def __init__(self, num_hop, in_channels = 1, hidden_channels = 1, out_channels = 1, dropout = 0, use_bn=False):
        super(AGG_NET, self).__init__()
        self.use_bn = use_bn
        self.convs = torch.nn.ModuleList()
        
        if self.use_bn:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        self.convs.append(
            GCNConv(in_channels, out_channels, bias = False, normalize = False, add_self_loops = True, aggr = 'add'))
        
        for _ in range(num_hop - 1):
            if self.use_bn:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.convs.append(
                GCNConv(out_channels, out_channels, bias = False, normalize = False, add_self_loops = True, aggr = 'add'))
            
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        from torch_geometric.nn.inits import ones
        for conv in self.convs:
            conv.reset_parameters()
            ones(conv.weight)
        
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index, get_h=False):
        for i, conv in enumerate(self.convs[:-1]):
            #x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        
        return x
import pickle
import os
import torch
import numpy as np
import pdb
import torch_geometric
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import KFold
            
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CoraFull, CitationFull
import torch_geometric.transforms as T

def load_nodes(args):
    
    assert args.dataset in ['CORA', 'CITESEER', 'PUBMED', 'CS', 'Physics', 'Computers', 'Photo']

    if args.dataset in ['CORA', 'CITESEER', 'PUBMED']:
        data = Planetoid('../data', args.dataset, pre_transform=T.Compose([T.NormalizeFeatures()])) 
        
        features = data.data.x
        edge_index = data.data.edge_index
        labels = data.data.y
        train_index = torch.where(data.data.train_mask)[0].tolist()
        train_label = labels[train_index]
        valid_index = torch.where(data.data.val_mask)[0].tolist()
        valid_label = labels[valid_index]
        test_index = torch.where(data.data.test_mask)[0].tolist()
        test_label = labels[test_index]
        num_classes = data.num_classes
        bn = False
        
    elif args.dataset in ['CS', 'Physics']:
        
        dataset = Coauthor('../data', args.dataset, pre_transform=T.Compose([T.NormalizeFeatures()])) 
        data = dataset.data
        
        test_mask = torch.ones(data.x.size(0), dtype=torch.bool)
        for i in range(dataset.num_classes):
            cls_idx = torch.where(data.y==i)[0]
            mask = torch.randperm(cls_idx.size(0))
            if i == 0:
                train_index = mask[:20]
                train_index = cls_idx[train_index]
                valid_index = mask[20:50]
                valid_index = cls_idx[valid_index]

            else:
                temp_train = mask[:20]
                temp_train = cls_idx[temp_train]
                temp_val = mask[20:50]
                temp_val = cls_idx[temp_val]

                train_index = torch.cat((train_index, temp_train))
                valid_index = torch.cat((valid_index, temp_val))

        test_mask[train_index] = False
        test_mask[valid_index] = False
        test_index = test_mask.nonzero(as_tuple=True)[0]


        features = data.x
        edge_index = data.edge_index
        labels = data.y

        train_label = labels[train_index]
        valid_label = labels[valid_index]
        test_label = labels[test_index]
        num_classes = dataset.num_classes
        bn = False
    
    elif args.dataset in ['Computers', 'Photo']:
        
        dataset = Amazon('../data', args.dataset, pre_transform=T.Compose([T.NormalizeFeatures()])) 
        data = dataset.data
        
        test_mask = torch.ones(data.x.size(0), dtype=torch.bool)
        for i in range(dataset.num_classes):
            cls_idx = torch.where(data.y==i)[0]
            mask = torch.randperm(cls_idx.size(0))
            if i == 0:
                train_index = mask[:20]
                train_index = cls_idx[train_index]
                valid_index = mask[20:50]
                valid_index = cls_idx[valid_index]

            else:
                temp_train = mask[:20]
                temp_train = cls_idx[temp_train]
                temp_val = mask[20:50]
                temp_val = cls_idx[temp_val]

                train_index = torch.cat((train_index, temp_train))
                valid_index = torch.cat((valid_index, temp_val))

        test_mask[train_index] = False
        test_mask[valid_index] = False
        test_index = test_mask.nonzero(as_tuple=True)[0]


        features = data.x
        edge_index = data.edge_index
        labels = data.y

        train_label = labels[train_index]
        valid_label = labels[valid_index]
        test_label = labels[test_index]
        num_classes = dataset.num_classes
        bn = False
    
    return features, edge_index, train_index, train_label, valid_index, valid_label, test_index, test_label, num_classes #edge_indices, node_features, train_target, None, None, valid_target, test_target, num_nodes, num_classes
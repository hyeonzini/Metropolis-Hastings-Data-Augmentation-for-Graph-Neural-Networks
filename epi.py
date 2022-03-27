import os
import pdb
import random
import datetime
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as u

from utils import n_f1_score
from model import GCN, MLP, GAT, SAGE, AGG_NET
from preprocess_nc import load_nodes
from sklearn.metrics import accuracy_score
from scipy.stats import truncnorm
from scipy.special import betaln

import torch
import torch.autograd.profiler as profiler

import pickle
import wandb
import time

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, (end_time-start_time)*100))
        return result
    return wrapper_fn


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, full=False):
        num_data = x.shape[0]
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        if full: 
            return -1.0 * b.sum(1)
        b = -1.0 * b.sum()
        b = b/num_data
        return b

class XeLoss(nn.Module):
    def __init__(self):
        super(XeLoss, self).__init__()
        
    def forward(self, y, x):
        num_data = x.shape[0]
        b = F.softmax(y, dim=1)*F.log_softmax(x, dim=1) - F.softmax(y, dim=1)*F.log_softmax(y, dim=1)
        b = -1.0 * b.sum()
        b = b/num_data
        return b
    
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon, self).__init__()
        
    def forward(self, y, x):
        num_data = x.shape[0]
        b = F.softmax(y, dim=1)*F.log_softmax(x, dim=1) - F.softmax(y, dim=1)*F.log_softmax(y, dim=1)
        b += F.softmax(x, dim=1)*F.log_softmax(y, dim=1) - F.softmax(x, dim=1)*F.log_softmax(x, dim=1)
        b = -0.5 * b.sum()
        b = b/num_data
        return b

def our_truncnorm(a, b, mu, sigma, x=None, mode='pdf'):
    a, b = (a - mu) / sigma, (b - mu) / sigma
    if mode=='pdf':
        return truncnorm.pdf(x, a, b, loc = mu, scale = sigma)
    elif mode=='rvs':
        return truncnorm.rvs(a, b, loc = mu, scale = sigma)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hard_xe_loss_op = nn.CrossEntropyLoss()   
soft_xe_loss_op = XeLoss()
h_loss_op = HLoss()
js_loss_op = Jensen_Shannon()

def aggregate(features, edge_index, agg_model, num_hop):
    n = features.shape[0]
    edge_index_w_sl = u.add_self_loops(edge_index, num_nodes = n)[0]
    s_vec = agg_model(features, edge_index_w_sl)
    return s_vec

def log_normal(a, b, sigma):
    return -1 * torch.pow(a - b, 2) / (2 * torch.pow(sigma, 2)) #/root2pi / sigma

def augment(args, org_edge_index, org_feature, delta_G_e, delta_G_v):
    m = org_edge_index.shape[1]
    num_edge_drop = int(m*delta_G_e)
    #######  flip_edge (A=1)  #######
    idx = torch.randperm(m, device='cuda')[:m-num_edge_drop]
    aug_edge_index = org_edge_index[:, idx]
    #################################    
    
    n = org_feature.shape[0]
    num_node_drop = int(n*delta_G_v)
    
    aug_feature = org_feature.clone()
    node_list = torch.ones(n, 1, device = device)
    ##########  flip_feat  ##########
    idx = torch.randperm(n, device='cuda')[:num_node_drop]
    aug_feature[idx] = 0
    node_list[idx] = 0

    if num_node_drop:
        aug_feature *= n / (n-num_node_drop)
    #################################
    return aug_edge_index, aug_feature, node_list



def train(args, feature, edge_index, aug_feature, aug_edge_index, node_list, train_idx, train_label, num_classes, model, opt_model, agg_model, epoch, cnt):    
    num_node = feature.shape[0]
    num_edge = edge_index.shape[1]
    
    delta_G_e = 1 - aug_edge_index.shape[1]/num_edge
    delta_G_e_aug = our_truncnorm(0, 1, delta_G_e, args.sigma_delta_e, mode='rvs')
    
    delta_G_v = 1 - node_list.sum().item()/num_node
    delta_G_v_aug = our_truncnorm(0, 1, delta_G_v, args.sigma_delta_v, mode='rvs')
    

    aug_edge_index2, aug_feature2, node_list2 = augment(args, edge_index, feature, delta_G_e_aug, delta_G_v_aug)
    
    
    model.train()
    output = model(feature, edge_index)

    feat_ones = torch.ones(num_node, 1, device = device)
    with torch.no_grad():
        delta_g_e     = 1 - (aggregate(feat_ones,  aug_edge_index,agg_model,  args.num_layers) / org_ego).squeeze(1) 
        delta_g_aug_e = 1 - (aggregate(feat_ones,  aug_edge_index2,agg_model, args.num_layers) / org_ego).squeeze(1)
        delta_g_v     = 1 - (aggregate(node_list,  edge_index,agg_model,      args.num_layers) / org_ego).squeeze(1) 
        delta_g_aug_v = 1 - (aggregate(node_list2, edge_index,agg_model,      args.num_layers) / org_ego).squeeze(1)
    
    
    max_ent = h_loss_op(torch.full((1, output.shape[1]), 1 / output.shape[1])).item()
    ent = h_loss_op(output.detach(), True) / max_ent
    
    
    p     = args.lam1_e * log_normal(delta_g_e,     args.mu_e, args.a_e * ent + args.b_e) + \
            args.lam1_v * log_normal(delta_g_v,     args.mu_v, args.a_v * ent + args.b_v)
    p_aug = args.lam1_e * log_normal(delta_g_aug_e, args.mu_e, args.a_e * ent + args.b_e) + \
            args.lam1_v * log_normal(delta_g_aug_v, args.mu_v, args.a_v * ent + args.b_v)
    
    q     = np.log(our_truncnorm(0, 1, delta_G_e_aug, args.sigma_delta_e, x=delta_G_e, mode='pdf')) + \
            args.lam2_e * betaln(num_edge - num_edge * delta_G_e + 1, num_edge * delta_G_e + 1) + \
            np.log(our_truncnorm(0, 1, delta_G_v_aug, args.sigma_delta_v, x=delta_G_v, mode='pdf')) + \
            args.lam2_v * betaln(num_node - num_node * delta_G_v + 1, num_node * delta_G_v + 1)
    q_aug = np.log(our_truncnorm(0, 1, delta_G_e, args.sigma_delta_e, x=delta_G_e_aug, mode='pdf')) + \
            args.lam2_e * betaln(num_edge - num_edge * delta_G_e_aug + 1, num_edge * delta_G_e_aug + 1) + \
            np.log(our_truncnorm(0, 1, delta_G_v, args.sigma_delta_v, x=delta_G_v_aug, mode='pdf')) + \
            args.lam2_v * betaln(num_node - num_node * delta_G_v_aug + 1, num_node * delta_G_v_aug + 1)
    
    
    acceptance = ( (torch.sum(p_aug) - torch.sum(p))  - (q_aug - q) )
    
    f1 = 0
    acc = 0
    
    if np.log(random.random()) < acceptance:
        aug_output = model(aug_feature, aug_edge_index)
        aug_output2 = model(aug_feature2, aug_edge_index2)
        
        sup_aug_score = aug_output[train_idx]

        loss_XE = hard_xe_loss_op(sup_aug_score, train_label)   
        if args.option_loss == 0:
            loss_KL = soft_xe_loss_op(aug_output, aug_output2)
        else:
            if delta_G_e + delta_G_v < delta_G_e_aug + delta_G_v_aug:
                loss_KL = js_loss_op(aug_output.detach(), aug_output2)
            else:
                loss_KL = js_loss_op(aug_output, aug_output2.detach())
        
        loss_H  = h_loss_op(output)

        total_loss = (loss_XE +
                      args.kl * loss_KL +
                      args.h  * loss_H)
        
        opt_model.zero_grad()
        total_loss.backward()
        opt_model.step()
        
        f1 = torch.mean(n_f1_score(torch.argmax(output[train_idx], dim=1), train_label, num_classes=num_classes)).item()
        acc = accuracy_score(torch.argmax(output[train_idx], dim=1).cpu(), train_label.cpu())
        
        next_aug_feature = aug_feature2
        next_aug_edge_index = aug_edge_index2
        next_node_list = node_list2
        cnt += 1
        accep = 1
        
    else:
        next_aug_feature = aug_feature
        next_aug_edge_index = aug_edge_index
        next_node_list = node_list
        accep = 0
    
    return next_aug_feature, next_aug_edge_index, next_node_list, f1, acc, cnt, delta_G_e, delta_G_v, delta_g_e.mean().item(), delta_g_v.mean().item(), accep

@torch.no_grad()
def evaluate(args, feature, edge_index, idx, label, num_classes, model):
    
    model.eval()
    eval_feature = feature
    output = model(feature, edge_index)
    score = output[idx]
    loss = hard_xe_loss_op(score, label)

    f1 = torch.mean(n_f1_score(torch.argmax(score, dim=1), label, num_classes=num_classes)).numpy()
    acc = accuracy_score(torch.argmax(score, dim=1).cpu(), label.cpu())
        
    return f1, acc

def episode(args):
    global org_ego

    features, edge_index, train_index, train_label, valid_index, valid_label, test_index, test_label, num_classes = load_nodes(args)
    
    features = features.to(device)
    edge_index = edge_index.to(device)
    
    train_label = train_label.to(device)
    valid_label = valid_label.to(device)
    test_label = test_label.to(device)
   
    in_channels = features.shape[1]
    hidden_channels = args.emb_dim
    num_layers = args.num_layers
    dropout = args.dropout
    learning_rate = args.lr
    weight_decay = args.decay
    
    
    agg_model = AGG_NET(num_hop = num_layers).cuda()
    agg_model.eval()
    with torch.no_grad():
        org_ego = aggregate(torch.ones(features.shape[0],1, device = device), edge_index, agg_model,args.num_layers)
    if args.model_name == 'GCN':
        model = GCN(in_channels, hidden_channels, num_classes, num_layers, dropout).to(device)
    elif args.model_name == 'GAT':
        model = GAT(in_channels, hidden_channels, num_classes, num_layers, dropout, args.num_heads, args.att_dropout).to(device)
    elif args.model_name == 'SAGE':
        model = SAGE(in_channels, hidden_channels, num_classes, num_layers, dropout).to(device)
    elif args.model_name == 'MLP':
        model = MLP(in_channels, hidden_channels, num_classes, num_layers, dropout).to(device)
    opt_model = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)    

    best_valid_acc, best_valid_f1, best_test_acc, best_test_f1, best_epoch = -1, -1, -1, -1, -1
    aug_feature, aug_edge_index, aug_node_list = features, edge_index, torch.ones(features.shape[0], 1, device = device)
    
    cnt = 0
    delta_G_e_list = []
    delta_G_v_list = []
    delta_g_e_list = []
    delta_g_v_list = []
    for epoch in range(args.num_epochs):
        if epoch % (args.print*5) == 0:
            print("epochs :", epoch)
        aug_feature, aug_edge_index, aug_node_list, train_f1, train_acc, cnt, delta_G_e, delta_G_v, delta_g_e, delta_g_v, accep = train(args, features, edge_index, aug_feature, aug_edge_index, aug_node_list, train_index, train_label, num_classes, model, opt_model, agg_model, epoch, cnt)
        
        if accep == 1:
            valid_f1, valid_acc = evaluate(args, features, edge_index, valid_index, valid_label, num_classes, model)
            test_f1, test_acc = evaluate(args, features, edge_index, test_index, test_label, num_classes, model)
        
            delta_G_e_list.append(delta_G_e)
            delta_G_v_list.append(delta_G_v)
            delta_g_e_list.append(delta_g_e)
            delta_g_v_list.append(delta_g_v)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_test_acc = test_acc
                best_epoch = cnt

            if valid_f1 > best_valid_f1:
                best_valid_f1 = valid_f1
                best_test_f1 = test_f1
                
            if cnt % args.print == 0:
                print("############################################")
                print("epochs:", epoch, "cnt:", cnt)
                print('#train: {0:0.4f}'.format(train_acc), 'val: {0:0.4f}'.format(valid_acc), 'test: {0:0.4f}'.format(test_acc), 'best_test: {0:0.4f}'.format(best_test_acc), 'best_val: {0:0.4f}'.format(best_valid_acc))
                print('#train: {0:0.4f}'.format(train_f1), 'val: {0:0.4f}'.format(valid_f1), 'test: {0:0.4f}'.format(test_f1), 'best_test: {0:0.4f}'.format(best_test_f1), 'best_val: {0:0.4f}'.format(best_valid_f1))
        
        if cnt > args.max_epochs:
            break
        
    accep = cnt
    return best_epoch, train_acc, train_f1, best_test_acc, best_valid_acc, best_test_f1, best_valid_f1, accep, np.mean(delta_G_e_list), np.mean(delta_G_v_list), np.mean(delta_g_e_list), np.mean(delta_g_v_list)



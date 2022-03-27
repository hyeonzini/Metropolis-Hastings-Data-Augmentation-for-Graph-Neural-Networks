import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

import pdb
import random
import torch_geometric.utils as u
import datetime
from sklearn.metrics import accuracy_score

from epi import episode

import wandb
import time
def main():
    parser = argparse.ArgumentParser(description="Graph Augmentation")
    
    # environment
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--use_seed', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='CORA')
    parser.add_argument('--num_epochs', type=int, default=7000, help='number of epochs')
    parser.add_argument('--max_epochs', type=int, default=2000, help='number of epochs')
    parser.add_argument('--num', type=int, default=2, help='number of episode')
    parser.add_argument('--print', type=int, default=1)
    parser.add_argument('--wandb', type=int, default=0)
    parser.add_argument('--wandb_name', type=str, default='ours_manyparam')
    
    # classifier model params
    parser.add_argument('--model_name', type=str, default='GCN', help='which model to use')#sgc, mlp
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--att_dropout', type=float, default=0.5)
    parser.add_argument('--num_heads', type=int, default=8)
    
    parser.add_argument('--a_e', type=float, default=100)
    parser.add_argument('--b_e', type=float, default=1)
    parser.add_argument('--a_v', type=float, default=100)
    parser.add_argument('--b_v', type=float, default=1)
    
    parser.add_argument('--kl', type=float, default=2.0)
    parser.add_argument('--h', type=float, default=0.2)
    
    parser.add_argument('--sigma_delta_e', type=float, default=0.03)
    parser.add_argument('--sigma_delta_v', type=float, default=0.03)
    parser.add_argument('--mu_e', type=float, default=0.6)
    parser.add_argument('--mu_v', type=float, default=0.2)
    parser.add_argument('--lam1_e', type=float, default=1)
    parser.add_argument('--lam1_v', type=float, default=1)
    parser.add_argument('--lam2_e', type=float, default=0.000)
    parser.add_argument('--lam2_v', type=float, default=0.000)
    
    parser.add_argument('--option_loss', type=int, default=0)
    
    args = parser.parse_args()
    
    def set_seed(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.wandb:
        wandb.init(project=args.wandb_name, reinit=True)
        wandb.config.update(args)

    if args.use_seed:
        set_seed(args.seed)
    
    best_epoch, tr_acc, tr_f1, best_te_acc, best_val_acc, best_te_f1, best_val_f1, accep, delta_G_e,delta_G_v, delta_g_e, delta_g_v  = episode(args)
    
if __name__ == "__main__":
    main()

    
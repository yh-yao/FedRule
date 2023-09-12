#!/usr/bin/env python
# coding: utf-8

import os
import yaml
from pathlib import Path
import torch
import argparse
import dgl
import json
import itertools
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from data_splitter import split_data
from utils import compute_loss, increment_dir, test_global_model, get_recommendation_result
from model import GraphSAGE, GCN, GAT, HeteroMLPPredictor


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='wyze', type=str)
    parser.add_argument('-l', '--logdir', default='./runs', type=str)
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float)
    parser.add_argument('-c', '--num_comms',default=100, type=int) # num_iterations in centralized training
    parser.add_argument('-m', '--model_type', default='graphsage', type=str)
    parser.add_argument('-neg', '--more_negative', action='store_true')

    seed = 0
    dgl.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    args = parser.parse_args()

    args.log_dir =  increment_dir(Path(args.logdir) / 'exp')
    args.log_dir += args.dataset + "_" + 'center'
    os.makedirs(args.log_dir)
    yaml_file = str(Path(args.log_dir) / "args.yaml")
    with open(yaml_file, 'w') as out:
        yaml.dump(args.__dict__, out, default_flow_style=False)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tb_writer = SummaryWriter(log_dir=args.log_dir)
    
    print("load data")
    user_graphs_list, _ = dgl.load_graphs("usergraphs.bin")
    user_id_dict = json.load(open('user_id_dict.json', 'r'))
    user_graphs = dict()
    user_ids = list(user_id_dict.keys())

    for i in range(len(user_ids)):
        user_graphs[user_ids[i]] = user_graphs_list[i]

    all_trigger_actions = json.load(open('all_trigger_actions.json', 'r'))
    all_devices = json.load(open('all_devices.json', 'r'))
    user_device_id_to_node_id = json.load(open('user_device_id_to_node_id.json', 'r'))
    
    train_gs, train_pos_gs, train_neg_gs, test_pos_gs, test_neg_gs = split_data(user_graphs, all_trigger_actions, args.more_negative)
    if args.model_type == 'graphsage':
        model = GraphSAGE(len(set(all_devices.values())), 32).to(args.device) #feature dim: len(set(all_devices.values()))
    elif args.model_type == 'gcn':
        model = GCN(len(set(all_devices.values())), 32).to(args.device) 
    elif args.model_type == 'gat':
        model = GAT(len(set(all_devices.values())), 32).to(args.device) 
        
    pred = HeteroMLPPredictor(32, len(set(all_trigger_actions.values()))).to(args.device)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.learning_rate)

    print("start training")
    for e in range(args.num_comms):
        model.train()
        pred.train()
        loss = None
        for user_index in train_gs:
            train_g = train_gs[user_index]
            train_pos_g = train_pos_gs[user_index]
            train_neg_g = train_neg_gs[user_index]
            
            h = model(train_g, train_g.ndata['feat'])
            pos_score = pred(train_pos_g, h)[list(range(len(train_pos_g.edata['etype']))), train_pos_g.edata['etype']]
            neg_score = pred(train_neg_g, h)[list(range(len(train_neg_g.edata['etype']))), train_neg_g.edata['etype']]
            if loss == None:
                loss = compute_loss(pos_score, neg_score)
            else:
                loss += compute_loss(pos_score, neg_score)
            
        tb_writer.add_scalar('Train/Loss',loss.item() / len(train_gs),e-1)  #-1 since it is before backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('In epoch {}, loss: {}'.format(e-1, loss.item() / len(train_gs)))
        
        if (e + 1) % 5 == 0:
            global_train_loss, global_train_AUC, global_train_MR, global_test_loss, global_test_AUC, global_test_MR = test_global_model(train_gs, train_pos_gs, train_neg_gs, test_pos_gs, test_neg_gs, model, pred)

            tb_writer.add_scalar('Global Train/Loss', global_train_loss, e)
            tb_writer.add_scalar('Global Train/AUC', global_train_AUC, e)
            tb_writer.add_scalar('Global Train/POS_MR', global_train_MR, e)

            tb_writer.add_scalar('Global Test/Loss', global_test_loss, e)
            tb_writer.add_scalar('Global Test/AUC', global_test_AUC, e)
            tb_writer.add_scalar('Global Test/POS_MR', global_test_MR, e)
            
    torch.save(model.state_dict(), args.dataset + "central_model_" + args.model_type)
    torch.save(pred.state_dict(), args.dataset + "central_pred_" + args.model_type)
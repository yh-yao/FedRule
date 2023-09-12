#!/usr/bin/env python
# coding: utf-8

# In[1]:


from huggingface_hub import login
login(token="XXXXXXXXXXXXXX")
from datasets import load_dataset
import collections
import torch
import torch.nn.functional as F
import dgl
import json
import numpy as np
import networkx as nx
from model import GraphSAGE, GCN, GAT, HeteroMLPPredictor

test_rule = load_dataset("wyzelabs/RuleRecommendation", data_files="test_rule.csv")
test_rule_df = test_rule['train'].to_pandas()


all_trigger_actions = json.load(open('all_trigger_actions.json', 'r'))
all_devices = json.load(open('all_devices.json', 'r'))


class_num = len(set(all_devices.values()))
one_hot_matrix = F.one_hot(torch.tensor(list(range(class_num))), num_classes=class_num)

test_device_df = load_dataset("wyzelabs/RuleRecommendation", data_files="test_device.csv")['train'].to_pandas()


user_device_id_to_node_id = collections.defaultdict(dict)
user_number_of_devices = collections.defaultdict(int)
test_user_graphs = collections.defaultdict(dgl.DGLGraph)

'''
# add devices that have no rules
for index, row in test_device_df.iterrows():
    user_device_id_to_node_id[row["user_id"]][row['device_id']] = user_number_of_devices[row["user_id"]]
    user_number_of_devices[row["user_id"]] += 1
    
    device_type = all_devices[row['device_model']]
    test_user_graphs[row["user_id"]].add_nodes(1, data = {'feat':one_hot_matrix[device_type].reshape(1,-1)})
'''

for index, row in test_rule_df.iterrows():
    trigger_action = str(row['trigger_state_id'])+' '+str(row['action_id'])
    if trigger_action in all_trigger_actions and row['trigger_device'] in all_devices and row['action_device'] in all_devices:
        if row['trigger_device_id'] not in user_device_id_to_node_id[row["user_id"]]:
            #assign id to the current device for supporting multiple devices with the same type
            user_device_id_to_node_id[row["user_id"]][row['trigger_device_id']] = user_number_of_devices[row["user_id"]]
            user_number_of_devices[row["user_id"]] += 1

            device = all_devices[row['trigger_device']]
            test_user_graphs[row["user_id"]].add_nodes(1, data = {'feat':one_hot_matrix[device].reshape(1,-1)})

        if row['action_device_id'] not in user_device_id_to_node_id[row["user_id"]]:
            user_device_id_to_node_id[row["user_id"]][row['action_device_id']] = user_number_of_devices[row["user_id"]]
            user_number_of_devices[row["user_id"]] += 1

            device = all_devices[row['action_device']]
            test_user_graphs[row["user_id"]].add_nodes(1, data = {'feat':one_hot_matrix[device].reshape(1,-1)})
        node1 = user_device_id_to_node_id[row["user_id"]][row['trigger_device_id']]
        node2 = user_device_id_to_node_id[row["user_id"]][row['action_device_id']]
        #the file contains same rules but with different devices
        test_user_graphs[row["user_id"]].add_edges(node1, node2, data = {'etype':torch.tensor([all_trigger_actions[trigger_action]])}) #directed

model = GraphSAGE(len(set(all_devices.values())), 32)
pred = HeteroMLPPredictor(32, len(set(all_trigger_actions.values())))
model.load_state_dict(torch.load("wyzecentral_model_graphsage"))
pred.load_state_dict(torch.load("wyzecentral_pred_graphsage"))


def get_recommendation_result(G, model, pred, topk, rule_dict=None):
    Complete_G = nx.complete_graph(list(G.nodes()), nx.MultiDiGraph())
    Complete_G = dgl.from_networkx(Complete_G)
    Complete_G = dgl.add_self_loop(Complete_G)
    
    model.eval()
    pred.eval()
    with torch.no_grad():
        h = model(G, G.ndata['feat'])
        #predictor, use node embeddings of source node and target node as input, predict the link probability of current edge
        #need a complete graph as input
        scores = pred(Complete_G, h)
    L = []
    edges = Complete_G.edges()
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            # trigger_device, action_device, trigger_action_pair, score
            L.append([int(edges[0][i]), int(edges[1][i]), j, float(scores[i][j])])

    # rule filter
    if rule_dict is not None:
        node_transfer = np.argmax(G.ndata['feat'].numpy(), axis = 1)
        for i in L:
            trigger_device_type = node_transfer[int(i[0])]
            action_device_type = node_transfer[int(i[1])]
            trigger_action_pair = int(i[2])
            rule = f"{trigger_device_type} {action_device_type} {trigger_action_pair}"
            # give more score if the rule in rule_dict
            if rule in rule_dict:
                i[3] += 1
    L = torch.tensor(sorted(L, key= lambda e:e[3], reverse = True))[:,:-1]

    return L[:topk]


train_rule = load_dataset("wyzelabs/RuleRecommendation", data_files="train_rule.csv")
train_df = train_rule['train'].to_pandas()

rule_dict = dict()
for index, row in train_df.iterrows():
    trigger_action = str(row['trigger_state_id'])+' '+str(row['action_id'])
    if trigger_action in all_trigger_actions:
        current_rule = f"{all_devices[row['trigger_device']]} {all_devices[row['action_device']]} {all_trigger_actions[trigger_action]}"
        rule_dict[current_rule] = 1


inv_all_trigger_actions_with_id = dict()
for key in all_trigger_actions:
    inv_all_trigger_actions_with_id[all_trigger_actions[key]] = key
    
a = open("recommended_result.csv", "w")
a.write("user_id,rule,rank\n")
for user_index in test_user_graphs:
    test_user_graphs[user_index].ndata['feat'] = test_user_graphs[user_index].ndata['feat'].float()
    L = get_recommendation_result(test_user_graphs[user_index], model, pred, topk = 50, rule_dict=rule_dict)
    current_user_devices = list(user_device_id_to_node_id[user_index].keys())
    for rank in range(len(L)):
        current_rule = L[rank]
        trigger_device = current_user_devices[int(current_rule[0])]
        action_device = current_user_devices[int(current_rule[1])]
        trigger_state, action = inv_all_trigger_actions_with_id[int(current_rule[2])].split(' ')
        a.write(f"{user_index},{trigger_device}_{trigger_state}_{action}_{action_device},{rank + 1}\n")
a.close()




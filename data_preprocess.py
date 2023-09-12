#!/usr/bin/env python
# coding: utf-8

from huggingface_hub import login
# enter your token of huggingface
# https://huggingface.co/docs/hub/security-tokens
login(token="XXXXXXXXXXXXXX")

from datasets import load_dataset
import collections
import torch
import torch.nn.functional as F
import dgl
import json

train_device = load_dataset("wyzelabs/RuleRecommendation", data_files="train_device.csv")
train_rule = load_dataset("wyzelabs/RuleRecommendation", data_files="train_rule.csv")
test_device = load_dataset("wyzelabs/RuleRecommendation", data_files="test_device.csv")
test_rule = load_dataset("wyzelabs/RuleRecommendation", data_files="test_rule.csv")

print("start preprocessing")
df = train_rule['train'].to_pandas()
all_devices = {'Camera': 0,
                 'ClimateSensor': 1,
                 'Cloud': 2,
                 'ContactSensor': 3,
                 'Irrigation': 4,
                 'LeakSensor': 5,
                 'Light': 6,
                 'LightStrip': 7,
                 'Lock': 8,
                 'MeshLight': 9,
                 'MotionSensor': 10,
                 'OutdoorPlug': 11,
                 'Plug': 12,
                 'RobotVacuum': 13,
                 'Switch': 14,
                 'Thermostat': 15}

all_trigger_actions = collections.defaultdict(int)
for index, row in df.iterrows():
                all_trigger_actions[str(row['trigger_state_id'])+' '+str(row['action_id'])] += 1

#filter some action 
for index in list(all_trigger_actions.keys()):
        if all_trigger_actions[index] <= 10:
            all_trigger_actions.pop(index)


for index, key in enumerate(all_trigger_actions):
            all_trigger_actions[key] = index

class_num = len(set(all_devices.values()))
one_hot_matrix = F.one_hot(torch.tensor(list(range(class_num))), num_classes=class_num)

user_device_id_to_node_id = collections.defaultdict(dict)
user_number_of_devices = collections.defaultdict(int)

user_graphs = collections.defaultdict(dgl.DGLGraph)

count = 0
for index, row in df.iterrows():
    trigger_action = str(row['trigger_state_id'])+' '+str(row['action_id'])
    if trigger_action in all_trigger_actions and row['trigger_device'] in all_devices and row['action_device'] in all_devices:

        if row['trigger_device_id'] not in user_device_id_to_node_id[row["user_id"]]:
            #assign id to the current device for supporting multiple devices with the same type
            user_device_id_to_node_id[row["user_id"]][row['trigger_device_id']] = user_number_of_devices[row["user_id"]]
            user_number_of_devices[row["user_id"]] += 1

            device = all_devices[row['trigger_device']]
            user_graphs[row["user_id"]].add_nodes(1, data = {'feat':one_hot_matrix[device].reshape(1,-1)})

        if row['action_device_id'] not in user_device_id_to_node_id[row["user_id"]]:
            user_device_id_to_node_id[row["user_id"]][row['action_device_id']] = user_number_of_devices[row["user_id"]]
            user_number_of_devices[row["user_id"]] += 1

            device = all_devices[row['action_device']]
            user_graphs[row["user_id"]].add_nodes(1, data = {'feat':one_hot_matrix[device].reshape(1,-1)})
        node1 = user_device_id_to_node_id[row["user_id"]][row['trigger_device_id']]
        node2 = user_device_id_to_node_id[row["user_id"]][row['action_device_id']]
        #the file contains same rules but with different devices
        user_graphs[row["user_id"]].add_edges(node1, node2, data = {'etype':torch.tensor([all_trigger_actions[trigger_action]])}) #directed
    count += 1
    if count > 1000:
        break

#filter, remove graph with devices < 3
for i in list(user_graphs.keys()):
    if user_graphs[i].number_of_nodes() <= 2:
        user_graphs.pop(i)

print("save processed data")

user_id_dict = dict()
for i in user_graphs.keys():
    user_id_dict[i] = i

with open('user_id_dict.json', 'w') as f:
    json.dump(user_id_dict, f)

dgl.save_graphs("usergraphs.bin", list(user_graphs.values()))

with open('all_trigger_actions.json', 'w') as f:
    json.dump(all_trigger_actions, f)
    
with open('all_devices.json', 'w') as f:
    json.dump(all_devices, f)
    
with open('user_device_id_to_node_id.json', 'w') as f:
    json.dump(user_device_id_to_node_id, f)


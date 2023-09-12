import numpy as np
import dgl
import torch
import torch.nn.functional as F

def split_data(user_graphs, all_trigger_actions, more_neg = False):
    train_gs = dict()
    train_pos_gs = dict()
    train_neg_gs = dict()
    test_pos_gs = dict()
    test_neg_gs = dict()

    for user_index in user_graphs:
        g = user_graphs[user_index]

        # Split edge set for training and testing
        u, v = g.edges()

        # eids = np.arange(g.number_of_edges())
        eids = torch.randperm(g.number_of_edges()).to(g.device)
        test_size = int(len(eids) * 0.2) 
        train_size = g.number_of_edges() - test_size
        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

        test_pos_edata = g.edata['etype'][eids[:test_size]]
        train_pos_edata = g.edata['etype'][eids[test_size:]]

        if len(test_pos_u) == 0:
            continue #remove some graph with few edges

        # Find all negative edges and split them for training and testing
        adj = g.adjacency_matrix().to(g.device)
        adj_neg = 1 - adj.to_dense() - torch.eye(g.number_of_nodes()).to(g.device) #no self loop, filtered before
        neg_inds = torch.nonzero(adj_neg)
        neg_edata = torch.randint(len(set(all_trigger_actions.values())), (neg_inds.shape[0],)).to(g.device)
        
        
        #add the same edge in dataset but different edge type for negative sampleing
        neg_u_different_type = u
        neg_v_different_type = v
        #neg_sampling
        if more_neg == True:
            for i in range(10):
                neg_u_different_type = torch.cat((neg_u_different_type, u), 0)
                neg_v_different_type = torch.cat((neg_v_different_type, v), 0)

        neg_edge_fetures = torch.randint(len(set(all_trigger_actions.values())), (len(neg_u_different_type),)).to(g.device)

        for i in range(len(u)):
            same_edges_with_different_type = g.edata['etype'][g.edge_ids(u[i],v[i], return_uv = True)[2]]
            while neg_edge_fetures[i] in same_edges_with_different_type:
                neg_edge_fetures[i] = np.random.choice(len(set(all_trigger_actions.values())), 1)[0]
                
        neg_u = torch.cat((neg_inds[:,0], neg_u_different_type), 0)
        neg_v = torch.cat((neg_inds[:,1], neg_v_different_type), 0)
        neg_edata = torch.cat((neg_edata, neg_edge_fetures), 0)
        
        #print(len(neg_u), len(neg_edata))
        
        if len(neg_u) == 0:
            continue #some graphs are too small, become complete graphs, skip it

        #neg_eids = list(range(len(neg_u)))#np.random.choice(len(neg_u), len(neg_u))   #### super negative sampling, many edges
        
        neg_eids = torch.randperm(len(neg_u))[:g.number_of_edges()]
        
        test_neg_edata = neg_edata[neg_eids[:test_size]]
        train_neg_edata = neg_edata[neg_eids[test_size:]]

        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
        train_pos_g.edata['etype'] = train_pos_edata

        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
        train_neg_g.edata['etype'] = train_neg_edata

        test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
        test_pos_g.edata['etype'] = test_pos_edata

        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
        test_neg_g.edata['etype'] = test_neg_edata

        train_g = dgl.remove_edges(g, eids[:test_size])

        train_g.ndata['feat'] = train_g.ndata['feat'].float()
        
        
        #add efeat
        edge_types = len(set(all_trigger_actions.values()))
        one_hot_matrix = F.one_hot(torch.tensor(list(range(edge_types))), num_classes=edge_types).to(g.device)
        train_g.edata['efeat'] = one_hot_matrix[train_g.edata['etype']]
        train_pos_g.edata['efeat'] = one_hot_matrix[train_pos_g.edata['etype']]
        train_neg_g.edata['efeat'] = one_hot_matrix[train_neg_g.edata['etype']]
        test_pos_g.edata['efeat'] = one_hot_matrix[test_pos_g.edata['etype']]
        test_neg_g.edata['efeat'] = one_hot_matrix[test_neg_g.edata['etype']]


        train_gs[user_index] = train_g
        train_pos_gs[user_index] = train_pos_g
        train_neg_gs[user_index] = train_neg_g
        test_pos_gs[user_index] = test_pos_g
        test_neg_gs[user_index] = test_neg_g
    return train_gs, train_pos_gs, train_neg_gs, test_pos_gs, test_neg_gs

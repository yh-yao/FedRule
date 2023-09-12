import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import networkx as nx
import dgl
from pathlib import Path
import glob
import re

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(pos_score.device)
    return F.binary_cross_entropy(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    scores = np.nan_to_num(scores, nan=0, posinf=0, neginf=0)
    #deal with overflow
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def compute_mr(predictions, test_g):
    _, indices = torch.sort(predictions, descending = True)
    hits = (test_g.edata['etype'].view(-1, 1) == indices).nonzero()[:,-1]
    return hits.float().mean()

def test_global_model(train_gs, train_pos_gs, train_neg_gs, test_pos_gs, test_neg_gs, global_model, global_pred):
    global_model.eval()
    global_pred.eval()
    total_loss = 0
    total_AUC = 0
    total_pos_MR = 0

    total_train_loss = 0
    total_train_AUC = 0
    total_train_pos_MR = 0
    with torch.no_grad():
        for user_index in test_pos_gs:
            train_g = train_gs[user_index]
            train_pos_g = train_pos_gs[user_index]
            train_neg_g = train_neg_gs[user_index]

            h = global_model(train_g, train_g.ndata['feat'])
            pos_score = global_pred(train_pos_g, h)[
                list(range(len(train_pos_g.edata['etype']))), train_pos_g.edata['etype']]
            neg_score = global_pred(train_neg_g, h)[
                list(range(len(train_neg_g.edata['etype']))), train_neg_g.edata['etype']]

            total_train_loss += compute_loss(pos_score, neg_score)
            total_train_AUC += compute_auc(pos_score, neg_score)
            total_train_pos_MR += compute_mr(global_pred(train_pos_g, h), train_pos_g)

            test_pos_g = test_pos_gs[user_index]
            test_neg_g = test_neg_gs[user_index]

            pos_score = global_pred(test_pos_g, h)[
                list(range(len(test_pos_g.edata['etype']))), test_pos_g.edata['etype']]

            neg_score = global_pred(test_neg_g, h)[
                list(range(len(test_neg_g.edata['etype']))), test_neg_g.edata['etype']]

            total_loss += compute_loss(pos_score, neg_score)
            total_pos_MR += compute_mr(global_pred(test_pos_g, h), test_pos_g)
            total_AUC += compute_auc(pos_score, neg_score)

    print('Global Test Loss', total_loss / len(test_pos_gs))
    print('Global Test AUC', total_AUC / len(test_pos_gs))
    print('Global Test Positive MR', float(total_pos_MR / len(test_pos_gs)))

    return float(total_train_loss / len(train_pos_gs)), total_train_AUC / len(train_pos_gs), float(
        total_train_pos_MR / len(train_pos_gs)), float(total_loss / len(test_pos_gs)), total_AUC / len(
        test_pos_gs), float(total_pos_MR / len(test_pos_gs))


def get_recommendation_result(G, model, pred, topk):
    Complete_G = nx.complete_graph(list(G.nodes()), nx.MultiDiGraph())
    Complete_G = dgl.from_networkx(Complete_G)

    model.eval()
    pred.eval()
    with torch.no_grad():
        h = model(G, G.ndata['feat'])
        # predictor, use node embeddings of source node and target node as input, predict the link probability of current edge
        # need a complete graph as input
        scores = pred(Complete_G, h)
    L = []
    edges = Complete_G.edges()
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            L.append([int(edges[0][i]), int(edges[1][i]), j, float(scores[i][j])])
    L = torch.tensor(sorted(L, key=lambda e: e[3], reverse=True))[:, :-1]

    return L[:topk]

def increment_dir(dir, comment=''):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    dirs = sorted(glob.glob(dir + '*'))  # directories
    if dirs:
        matches = [re.search(r"exp(\d+)", d) for d in dirs]
        idxs = [int(m.groups()[0]) for m in matches if m]
        if idxs:
            n = max(idxs) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')
import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from multiprocessing import Pool
from utils import add_nx_graph, precompute_dist_data
from queue import Queue
from copy import deepcopy
from networkx.algorithms.community import kernighan_lin_bisection
from torch_geometric.data import Data
import dgl
import torch.nn as nn

def check(dist_row, n, N):
    res = set()
    for i in range(N):
        for j in range(i+1, N):
            if dist_row[i] == dist_row[j]:
                res.add(N*i+j)
    # inds = inds[np.argwhere(dist_row[inds[:,0]] == dist_row[inds[:,1]]).flatten()]
    # res += set((N*inds[:,0]+inds[:,1]).tolist())
    print(f"finished {n}")
    return res

def compute_ntable(data):
    edges = data.edge_index
    edges = torch.cat((edges, torch.flip(edges,dims=(0,))), dim=1)
    dist_matrix = 1/data.dists - 1
    ntable = defaultdict(set)
    N = dist_matrix.shape[0]
    p = Pool(16)
    # inds = np.array([[i,j] for i in range(N) for j in range(N)])    
    # pargs = [(dist_matrix[i], inds, i, N) for i in range(N)]
    res = p.starmap(check, [(dist_matrix[i], i, N) for i in range(N)])
    # res = []
    # for parg in pargs:
    #     res.append(check(*parg))

    for i, rset in enumerate(res):
        ntable[i] = rset
    return ntable, edges

def cut(data, cut_size, max_iter=2):
    add_nx_graph(data, False)
    assert len(data.G) == len(data.x)
    q = Queue()
    q.put(data.G)
    cuts = []
    while not q.empty():
        cur = q.get()
        if len(cur) <= cut_size:
            cuts.append(cur)
        else:
            a, b = kernighan_lin_bisection(cur, max_iter=max_iter)
            print(f"bisected {len(cur)} into {len(a)} and {len(b)}")
            q.put(deepcopy(cur.subgraph(a)))
            q.put(deepcopy(cur.subgraph(b)))

    cuts_data = []
    for i, cut in enumerate(cuts):
        cut_mask = list(cut.nodes())
        reindex_map = dict(zip(cut_mask,range(len(cut_mask))))
        cut_set = set(cut_mask)
        cut_edges = data.edge_index.T[:0]
        for e in data.edge_index.T:
            e0, e1 = e[0].item(), e[1].item()
            if e0 in cut_set and e1 in cut_set:
                cut_edge = torch.tensor([[reindex_map[e0],reindex_map[e1]]])
                cut_edges = torch.cat((cut_edges,cut_edge),dim=0)
        cut_edges = cut_edges.T
        
        cut_dists = precompute_dist_data(cut_edges,len(cut_mask))
        cut_data = Data(x=data.x[cut_mask],edge_index=cut_edges.T,dists=cut_dists,cut_info=(i,cut_mask))
        cuts_data.append(cut_data)

    return cuts_data


def format(cut_size, data_list):
    new_data_list = []
    for data in data_list:
        if cut_size == -1:
            new_data_list.append(data)
        else:
            new_data_list += cut(data, cut_size)
    if cut_size > -1:
        data_list = new_data_list

    ntables, adjs, cuts = [], [], []
    for _, data in enumerate(data_list):
        ntable, edges = compute_ntable(data)
        data.ntable = ntable
        nr_degree = []
        for i in range(len(ntable)):
            nr_degree.append(len(ntable[i]))
        nr_degree = torch.tensor(nr_degree).reshape(-1,1)    
        data.nr_degree = nr_degree
            
        ntables.append(ntable)
        adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])),
                    shape=(len(ntable), len(ntable)),
                    dtype=np.float32)
        adjs.append(adj)
        if cut_size == -1:
            continue
        if data.cut_info[0] == 0:
            cuts.append([])
        cuts[-1].append(data.cut_info[-1])
    return ntables, adjs, cuts


def join_cuts(models, cuts_list):
    models.reverse()
    new_models = []
    for cuts in cuts_list:        
        new_model = deepcopy(models[-1]) # how to join models, weight space averaging?
        N = sum([len(c) for c in cuts])
        mask_weight = torch.empty(N,2)
        for c in cuts:
            mask_weight[c,:] = models[-1].mask.weight
            models.pop()
        new_model.mask = nn.Embedding.from_pretrained(mask_weight)
        new_models.append(new_model)

    return new_models


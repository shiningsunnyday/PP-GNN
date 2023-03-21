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
import os
import pickle

def check(dist_matrix, n, N):
    res = set()
    for i in range(N):
        for j in range(i+1, N):
            if dist_matrix[n][i] == dist_matrix[n][j]:
                res.add(N*i+j)
    # inds = inds[np.argwhere(dist_row[inds[:,0]] == dist_row[inds[:,1]]).flatten()]
    # res += set((N*inds[:,0]+inds[:,1]).tolist())
    return res

def compute_ntable(data):
    edges = data.edge_index
    edges = torch.cat((edges, torch.flip(edges,dims=(0,))), dim=1)
    dist_matrix = 1/data.dists - 1
    ntable = defaultdict(set)
    N = dist_matrix.shape[0]
    p = Pool(8)
    # inds = np.array([[i,j] for i in range(N) for j in range(N)])    
    # pargs = [(dist_matrix[i], inds, i, N) for i in range(N)]
    res = p.starmap(check, [(dist_matrix, i, N) for i in range(N)])
    # res = []
    # for parg in pargs:
    #     res.append(check(*parg))

    for i, rset in enumerate(res):
        ntable[i] = rset
    return ntable, edges

def cut(data, cut_size, max_iter=1):
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


def format(cut_size, data_list, cache_path=""):
    cuts_cache_path = cache_path.replace(".pkl","_cuts.pkl")
    if cut_size > -1 and os.path.exists(cuts_cache_path):
        num_cuts, new_data_list = pickle.load(open(cuts_cache_path,"rb"))
        print(f"loaded {num_cuts}/{len(data_list)} cuts for {cache_path}")
    else:
        num_cuts, new_data_list = 0, []
        if cut_size > -1:
            print(f"loaded 0/{len(data_list)} cuts for {cache_path}")

    for i,data in enumerate(data_list[num_cuts:]):
        if cut_size == -1:
            new_data_list.append(data)
        else:
            new_data_list += cut(data, cut_size)
            if cache_path:
                pickle.dump([i+1,new_data_list],open(cuts_cache_path,"wb+"))
                print(f"saved {i+1}/{len(data_list)} cuts")

    if cut_size > -1:
        data_list = new_data_list

    ntables, adjs, cuts = [], [], []    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path),exist_ok=True)
        if os.path.exists(cache_path):
            data = pickle.load(open(cache_path,"rb"))            
            print(f"loading from {cache_path}")
            ntables, adjs, cuts = data  
            print(f"loaded {cache_path}, {len(ntables)}/{len(data_list)} done")
        else:            
            pickle.dump([[],[],[]],open(cache_path,"wb+"))
            print(f"saved {cache_path}, 0/{len(data_list)} done") 

    num_cached = len(ntables)
    for _, data in enumerate(data_list[num_cached:]):
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
        if cache_path:
            pickle.dump([ntables,adjs,cuts],open(cache_path,"wb+"))
            print(f"saved {cache_path}, {len(ntables)}/{len(data_list)} done")            

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


from multiprocessing import Pool
import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from utils import add_nx_graph, precompute_dist_data
from queue import Queue
from copy import deepcopy
from networkx.algorithms.community import kernighan_lin_bisection
from torch_geometric.data import Data
import time
import torch.nn as nn
import os
import pickle
from tqdm import tqdm

def check(dist_row, n, N):
    res = set()
    # start = time.time()
    cts = [[] for _ in range(N+1)]
    for i in range(N):
        d = dist_row[i].int().item()
        cts[d].append(i)
    # t = time.time()-start
    for ct in cts:
        if not ct: continue
        for a in ct:
            for b in ct:
                if a >= b: continue
                res.add(N*a+b)
    # res_a = set()
    # t_a = time.time()
    # for i in range(N):
    #     for j in range(i+1, N):
    #         if dist_matrix[n][i] == dist_matrix[n][j]:
    #             res_a.add(N*i+j)
    # t_a = time.time()-t_a
    # print(f"t:{t} t_a:{t_a}")
    # assert res == res_a    
    return res

def compute_ntable(data):
    edges = data.edge_index
    edges = torch.cat((edges, torch.flip(edges,dims=(0,))), dim=1)
    dist_matrix = torch.as_tensor(1/data.dists - 1)
    N = dist_matrix.shape[0]
    dist_matrix[dist_matrix==np.inf] = N
    dist_matrix = torch.round(dist_matrix) # precision errors during division
    ntable = defaultdict(set)    
    p = Pool(8)
    # inds = np.array([[i,j] for i in range(N) for j in range(N)])    
    # pargs = [(dist_matrix[i], inds, i, N) for i in range(N)]
    res = p.starmap(check, tqdm([(dist_matrix[i], i, N) for i in range(N)]))
    # res = []
    # for parg in pargs:
    #     res.append(check(*parg))
    for i, rset in enumerate(res):
        ntable[i] = rset
    
    p.close()
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
        
        cut_dists = precompute_dist_data(cut_edges.T,len(cut_mask))
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
    num_cached = 0
    if cache_path:
        os.makedirs(os.path.dirname(cache_path),exist_ok=True)
        if os.path.exists(cache_path.replace('.pkl','.txt')):
            f = open(cache_path.replace('.pkl','.txt'),'r')
            num_cached = int(f.readlines()[0])
            # data = pickle.load(open(cache_path,"rb"))            
            print(f"{num_cached}/{len(data_list)} done")
            f.close()
        else:            
            # pickle.dump([[],[],[]],open(cache_path,"wb+"))
            f = open(cache_path.replace('.pkl','.txt'),'w+')            
            f.write(f"0\n")
            f.close()
            print(f"saved {cache_path}, 0/{len(data_list)} done") 
    for i, data in enumerate(data_list[num_cached:]):
        ntable, edges = compute_ntable(data)
        data.ntable = ntable
        nr_degree = []
        for j in range(len(ntable)):
            nr_degree.append(len(ntable[j]))
        nr_degree = torch.tensor(nr_degree).reshape(-1,1)    
        data.nr_degree = nr_degree
            
        adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])),
                    shape=(len(ntable), len(ntable)),
                    dtype=np.float32)
        
        if cache_path:
            cut_info_str = f"-{data.cut_info[0]}" if cut_size != -1 else ""
            cur_cache_path = cache_path.replace(".pkl",f"{cut_info_str}-{i+num_cached}.pkl")
            pickle.dump([ntable,adj,data.cut_info[-1] if cut_size != -1 else None],open(cur_cache_path,"wb+"))
            f = open(cache_path.replace(".pkl",".txt"),"w+")
            f.write(f"{i+1+num_cached}\n")
            f.close()
            print(f"saved to {cur_cache_path}, {i+num_cached+1}/{len(data_list)} done")
        else:
            ntables.append(ntable)
            adjs.append(adj)
            if cut_size != -1:
                if data.cut_info[0] == 0:
                    cuts.append([])
                cuts[-1].append(data.cut_info[-1])

    print("done with processing")
    if cache_path:
        for i in range(len(data_list)):
            data = data_list[i]
            cut_info_str = f"-{data.cut_info[0]}" if cut_size != -1 else ""
            cur_cache_path = cache_path.replace(".pkl",f"{cut_info_str}-{i}.pkl")
            ntable, adj, _ = pickle.load(open(cur_cache_path,"rb"))
            print(f"loaded {cur_cache_path}")
            ntables.append(ntable)
            adjs.append(adj)
            if cut_size != -1:
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


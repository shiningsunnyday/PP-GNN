import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from multiprocessing import Pool

def check(dist_matrix, n, N):
    res = set()
    for i in range(N):
        for j in range(i+1, N):
            if dist_matrix[n][i] == dist_matrix[n][j]:
                res.add(N*i+j)
    print(f"finished {n}")
    return res

def compute_ntable(data):
    edges = data.edge_index
    edges = torch.cat((edges, torch.flip(edges,dims=(0,))), dim=1)
    dist_matrix = 1/data.dists - 1
    ntable = defaultdict(set)
    N = dist_matrix.shape[0]
    p = Pool(8)
    res = p.starmap(check, [(dist_matrix, i, N) for i in range(N)])
    for i, rset in enumerate(res):
        ntable[i] = rset
    return ntable, edges


def format(data_list):
    ntables = []
    adjs = []
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
    return ntables, adjs

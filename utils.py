import torch
import networkx as nx
import numpy as np
import random
import hashlib
import json
import heapq
from tqdm import tqdm
from collections import defaultdict

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

def dict_hash(dictionary) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def hash_tensor(tensor):
    if isinstance(tensor,list):
        return hashlib.sha1(''.join([hash_tensor(t) for t in tensor]).encode('utf-8')).hexdigest()
    return hashlib.sha1(np.ascontiguousarray(tensor.numpy())).hexdigest()


# # approximate
def get_edge_mask_link_negative_approximate(mask_link_positive, num_nodes, num_negtive_edges):
    links_temp = np.zeros((num_nodes, num_nodes)) + np.identity(num_nodes)
    mask_link_positive = duplicate_edges(mask_link_positive)
    links_temp[mask_link_positive[0],mask_link_positive[1]] = 1
    # add random noise
    links_temp += np.random.rand(num_nodes,num_nodes)
    prob = num_negtive_edges / (num_nodes*num_nodes-mask_link_positive.shape[1])
    mask_link_negative = np.stack(np.nonzero(links_temp<prob))
    return mask_link_negative


# exact version, slower
def get_edge_mask_link_negative(mask_link_positive, num_nodes, num_negtive_edges):
    mask_link_positive_set = []
    for i in range(mask_link_positive.shape[1]):
        mask_link_positive_set.append(tuple(mask_link_positive[:,i]))
    mask_link_positive_set = set(mask_link_positive_set)

    mask_link_negative = np.zeros((2,num_negtive_edges), dtype=mask_link_positive.dtype)
    for i in range(num_negtive_edges):
        while True:
            mask_temp = tuple(np.random.choice(num_nodes,size=(2,),replace=False))
            if mask_temp not in mask_link_positive_set:
                mask_link_negative[:,i] = mask_temp
                break

    return mask_link_negative

def resample_edge_mask_link_negative(data):
    """
    sample negatives from undirected positive edges
    negative train edges exclude positive train edges
    negative val/test edges exclude all positive edges
    """
    data.mask_link_negative_train = get_edge_mask_link_negative(data.mask_link_positive_train, num_nodes=data.num_nodes,
                                                      num_negtive_edges=data.mask_link_positive_train.shape[1])
    data.mask_link_negative_val = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                      num_negtive_edges=data.mask_link_positive_val.shape[1])
    data.mask_link_negative_test = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                     num_negtive_edges=data.mask_link_positive_test.shape[1])


def deduplicate_edges(edges):
    edges_new = np.zeros((2,edges.shape[1]//2), dtype=int)
    # add none self edge
    j = 0
    skip_node = {} # node already put into result
    for i in range(edges.shape[1]):
        if edges[0,i]<edges[1,i]:
            edges_new[:,j] = edges[:,i]
            j += 1
        elif edges[0,i]==edges[1,i] and edges[0,i] not in skip_node:
            edges_new[:,j] = edges[:,i]
            skip_node.add(edges[0,i])
            j += 1

    return edges_new

def duplicate_edges(edges):
    return np.concatenate((edges, edges[::-1,:]), axis=-1)


# each node at least remain in the new graph
def split_edges(edges, remove_ratio, connected=False):
    e = edges.shape[1]
    edges = edges[:, np.random.permutation(e)]
    if connected:
        unique, counts = np.unique(edges, return_counts=True)
        node_count = dict(zip(unique, counts))

        index_train = []
        index_val = []
        for i in range(e):
            node1 = edges[0,i]
            node2 = edges[1,i]
            if node_count[node1]>1 and node_count[node2]>1: # if degree>1
                index_val.append(i)
                node_count[node1] -= 1
                node_count[node2] -= 1
                if len(index_val) == int(e * remove_ratio):
                    break
            else:
                index_train.append(i)
        index_train = index_train + list(range(i + 1, e))
        index_test = index_val[:len(index_val)//2]
        index_val = index_val[len(index_val)//2:]

        edges_train = edges[:, index_train]
        edges_val = edges[:, index_val]
        edges_test = edges[:, index_test]
    else:
        split1 = int((1-remove_ratio)*e)
        split2 = int((1-remove_ratio/2)*e)
        edges_train = edges[:,:split1]
        edges_val = edges[:,split1:split2]
        edges_test = edges[:,split2:]

    return edges_train, edges_val, edges_test




def edge_to_set(edges):
    edge_set = []
    for i in range(edges.shape[1]):
        edge_set.append(tuple(edges[:, i]))
    edge_set = set(edge_set)
    return edge_set


def get_link_mask(data, remove_ratio=0.2, resplit=True, infer_link_positive=True):
    if resplit:
        if infer_link_positive:
            data.mask_link_positive = deduplicate_edges(data.edge_index.numpy())
        data.mask_link_positive_train, data.mask_link_positive_val, data.mask_link_positive_test = \
            split_edges(data.mask_link_positive, remove_ratio)
    resample_edge_mask_link_negative(data)


def add_nx_graph(data, connected=True):
    G = nx.Graph()
    if not connected:
        G.add_nodes_from(range(len(data.x)))
    edge_numpy = data.edge_index.numpy()
    edge_list = []
    for i in range(data.num_edges):
        edge_list.append(tuple(edge_numpy[:, i]))
    G.add_edges_from(edge_list)
    data.G = G

def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in tqdm(node_range):
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes)<50:
        num_workers = int(num_workers/4)
    elif len(nodes)<400:
        num_workers = int(num_workers/2)
    import multiprocessing as mp
    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
            args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, num_nodes, approximate=0):
        '''
        Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
        :return:
        '''
        graph = nx.Graph()
        edge_list = edge_index.transpose(1,0).tolist()
        graph.add_edges_from(edge_list)

        n = num_nodes
        dists_array = np.zeros((n, n))
        # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
        # dists_dict = {c[0]: c[1] for c in dists_dict}
        dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff=approximate if approximate>0 else None,num_workers=8)
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist!=-1:
                    # dists_array[i, j] = 1 / (dist + 1)
                    dists_array[node_i, node_j] = 1 / (dist + 1)
        return dists_array


def get_candidates(N,filename):
    with open(filename,"r") as file:
        data = file.readlines()
    n = len(data)
    res = []
    for i in range(n-1,-1,-1):
        line = data[i].split('	')[3]
        line = line.replace('\'','').replace('u','').replace(',,',',')
        line = eval(line)
        for lin in line:
            res.append(lin[0])
            res.append(lin[1])
        res = list(set(res))
#         print(i,len(res))
        if len(res)>=N:
            break
    return res


def get_random_anchorset(n,c=0.5):
    m = int(np.log2(n))
    copy = int(c*m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n,size=anchor_size,replace=False))
    return anchorset_id

def get_edgeshell_anchors(n, filename="grid", subgraph_id = 0, c=0.5):
    # file = "/content/P-GNN-Google/P-GNN-master/data/"+filename+"/"+str(subgraph_id)+"_edge_shell.txt"
    file = f"./datasets/ShellOfEdgesInDatasets/{filename}_edge_shell.txt"
    m = int(np.log2(n))
    copy = int(c*m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        ''' change '''
        candidates = get_candidates(anchor_size,file)
        for j in range(copy):
            anchorset_id.append(np.random.choice(candidates,size=anchor_size,replace=True))
            # anchorset_id.append(candidates[0:anchor_size])
    return anchorset_id

def get_mvc_anchors(data):
    """
    GA-MPCA as in https://link.springer.com/chapter/10.1007/978-3-030-38819-5_9    
    """   
    N = data.num_nodes
    adj = np.zeros((N, N))
    V = set(range(N))
    A = set()
    anchors = []
    degs = defaultdict(int)
    for j in range(data.edge_index.shape[1]):
        a = data.edge_index[0,j].item()
        b = data.edge_index[1,j].item()
        degs[a] += 1
        degs[b] += 1
        adj[a][b] = 1
        adj[b][a] = 1
    

    while len(A) != N:
        max_v = -1
        max_deg = -1
        for v in V-A:
            if degs[v] > max_deg:
                max_deg = degs[v]
                max_v = v
        A.add(max_v)
        anchors.append([max_v])
        for j in range(N):
            if adj[max_v][j]:
                A.add(j)
    
    return anchors



def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0],len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0],len(anchorset_id))).long().to(device)
    for i in range(len(anchorset_id)):
        temp_id = torch.as_tensor(anchorset_id[i], dtype=torch.long)
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:,i] = dist_max_temp
        dist_argmax[:,i] = temp_id[dist_argmax_temp]
    return dist_max, dist_argmax


def preselect_anchor(args, data, layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu', dataset='grid'):

    # data.anchor_size_num = anchor_size_num
    # data.anchor_set = []
    # anchor_num_per_size = anchor_num//anchor_size_num
    # for i in range(anchor_size_num):
    #     anchor_size = 2**(i+1)-1
    #     anchors = np.random.choice(data.num_nodes, size=(layer_num,anchor_num_per_size,anchor_size), replace=True)
    #     data.anchor_set.append(anchors)
    # data.anchor_set_indicator = np.zeros((layer_num, anchor_num, data.num_nodes), dtype=int)

    if args.anchor_selection in ['p-gnn', 'pp-gnn']:
        anchorset_id = get_random_anchorset(data.num_nodes,c=1)
        data.anchorset_id = anchorset_id        
    elif args.anchor_selection == 'a-gnn':
        anchorset_id = get_mvc_anchors(data)
    elif args.anchor_selection == 'as-gnn':
        subgraph_id = 0        
        anchorset_id = get_edgeshell_anchors(data.num_nodes, dataset, subgraph_id, c=1)
        
    else:
        raise NotImplementedError
    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)

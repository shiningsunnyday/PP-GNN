from pretrain.GNNMDP.gnn_solver import pretrain_gnn_mdp
from pretrain.GNNMDP.models.gnn import GCN
from model import PGNN
import torch.nn.functional as F
from pretrain.utils import format, join_cuts
import pretrain.GNNMDP.repair_method as rm
from utils import hash_tensor
import torch
import dgl
import argparse

class PPGNN(PGNN):
    def __init__(self, model, mask, *pargs, **kwargs):
        super(PPGNN, self).__init__(*pargs, **kwargs)
        self.pretrained_model = model
        self.mask_ = mask
        if self.pretrained_model:
            for x, p in self.pretrained_model.named_parameters():
                p.requires_grad_(False)   
        

    def forward(self, G, data):
        if self.pretrained_model:
            data.x = torch.cat([data.x, data.nr_degree],dim=-1)
            data.x = self.pretrained_model(G, data.x)
        data.dists_argmax = torch.arange(len(data.x))[self.mask_==1][None].expand((len(data.x),-1))
        data.dists_max = 1/(1+data.dists[:, self.mask_==1])
        res = super(PPGNN, self).forward(data)
        return res
    

def pretrain(model, dataset, args, *pargs, **kwargs):
    """
    in: p-gnn model, list of [data]
    out: every data.dists_max, data.dists_argmax updated
    """
    path = 'datasets/cache/format/'    
    tensor_hash = hash_tensor([data.edge_index for data in dataset])
    hash_str = f"{args.approximate}-{args.cut_size}_{tensor_hash}"
    cache_path = path+f"{hash_str}.pkl" if args.cache else ""
    ntables, adjs, cuts = format(args.cut_size, dataset, cache_path) # convert to mdp format
    parser = argparse.ArgumentParser()
    mdp_args = parser.parse_args([])
    Gs = []
    models = []
    
    for ntable, adj in zip(ntables, adjs):
        for k, v in args.__dict__.items():
            if 'mdp_' in k:
                setattr(mdp_args, k[4:], v)
        (model, G), _, _, _, local_best_ind, _, _, _, _ = pretrain_gnn_mdp(mdp_args, 'gcn', adj, ntable) # if not gcn, need to worry about disconnected components
        model.local_best_ind = local_best_ind
        models.append(model)
        Gs.append(G)

    if args.cut_size == -1:
        ntable = ntables[0]
        model = models[0]
        G = Gs[0]
    else:
        ntable = ntables[0]
        models = join_cuts(models, cuts) # from cuts per dataset to one (model, G) per dataset    
        model = models[0] # for now assume only one pretraining dataset   
        _, adjs, _ = format(-1, dataset) # format the original dataset
        for adj, data in zip(adjs, dataset):
            edges = data.edge_index
            edges = torch.cat((edges, torch.flip(edges,dims=(0,))), dim=1)
            G = dgl.from_scipy(adj)
            G = dgl.add_self_loop(G)
            Gs.append(G)

        G = Gs[0] # for now assume one pretraining dataset

    
    mask = torch.zeros(model.mask.weight[:,0].shape)
    if len(local_best_ind):       
        r_set, dim = rm.repair_iter(args, local_best_ind, ntable, 100) 
    else:        
        r_set, dim = rm.repair_iter(args, set(ntable), ntable, 100)
    mask[list(r_set)] = 1.

    
    if args.anchors_only:
        model = PPGNN(None, mask, *pargs, **kwargs)
    else:
        model = PPGNN(model, mask, *pargs, **kwargs)
    return model, G
    
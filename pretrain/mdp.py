from pretrain.GNNMDP.gnn_solver import pretrain_gnn_mdp
from pretrain.GNNMDP.models.gnn import GCN
from model import PGNN
import torch.nn.functional as F
from pretrain.utils import format, join_cuts
import torch
import dgl
import argparse

class PPGNN(PGNN):
    def __init__(self, model, *pargs, **kwargs):
        super(PPGNN, self).__init__(*pargs, **kwargs)
        self.pretrained_model = model
        self.mask_ = F.gumbel_softmax(model.mask.weight,hard=True)[:, 0].T.clone().detach()
        for x, p in self.pretrained_model.named_parameters():
            p.requires_grad_(False)   
        

    def forward(self, G, data):
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
    ntables, adjs, cuts = format(args.cut_size, dataset) # convert to mdp format
    parser = argparse.ArgumentParser()
    mdp_args = parser.parse_args([])
    Gs = []
    models = []
    for ntable, adj in zip(ntables, adjs):
        for k, v in args.__dict__.items():
            if 'mdp_' in k:
                setattr(mdp_args, k[4:], v)

        model, G = pretrain_gnn_mdp(mdp_args, 'gcn', adj, ntable)[0] # if not gcn, need to worry about disconnected components
        models.append(model)
        Gs.append(G)

    if args.cut_size == -1:
        return models[0], Gs[0]
    else:
        Gs = []

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
    model = PPGNN(model, *pargs, **kwargs)
    return model, G
    
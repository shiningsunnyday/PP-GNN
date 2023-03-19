from pretrain.GNNMDP.gnn_solver import pretrain_gnn_mdp
from pretrain.GNNMDP.models.gnn import GCN
from model import PGNN
import torch.nn.functional as F
from pretrain.utils import format
import torch
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
    

def pretrain(model, dataset, *pargs, **kwargs):
    """
    in: p-gnn model, list of [data]
    out: every data.dists_max, data.dists_argmax updated
    """
    ntables, adjs = format(dataset) # convert to mdp format
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    Gs = []
    models = []
    for ntable, adj in zip(ntables, adjs):
        args.hidden = 32
        args.num_hidden_layers = 2
        args.mask_c = 3.0
        args.lr = 0.03
        args.weight_decay = 0.0
        args.epochs = 100
        args.batch_size = 1
        model, G = pretrain_gnn_mdp(args, 'gcn', adj, ntable)[0]
        mdp = (F.gumbel_softmax(model.mask.weight,hard=True)[:, :1].T).sum()
        print(f"{mdp}")
        models.append(model)
        Gs.append(G)

    model = models[0]   
    G = Gs[0]
    model = PPGNN(model, *pargs, **kwargs)
    return model, G
    
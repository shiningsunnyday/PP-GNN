from pretrain.GNNMDP.gnn_solver import pretrain_gnn_mdp
from pretrain.GNNMDP.models.gnn import GCN
from model import PGNN
from pretrain.utils import format
import argparse

class PPGNN(PGNN):
    def __init__(self, model, *pargs, **kwargs):
        super(PPGNN, self).__init__(*pargs, **kwargs)
        print("hi")
        return

    def forward(self, x):
        pass
    

def pretrain(model, dataset, *pargs, **kwargs):
    """
    in: p-gnn model, list of [data]
    out: every data.dists_max, data.dists_argmax updated
    """
    ntables, adjs = format(dataset) # convert to mdp format
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    for ntable, adj in zip(ntables, adjs):
        args.hidden = 32
        args.num_hidden_layers = 2
        args.mask_c = .5
        args.lr = 0.001
        args.weight_decay = 0.0
        args.epochs = 100
        args.batch_size = 1
        model = pretrain_gnn_mdp(args, 'gcn', adj, ntable)[0]
        
    model = PPGNN(model, *pargs, **kwargs)
    return model
    
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):

    DEFAULTS = {
        "hidden_dim" : 16,
        "depth" : 2,
        "dropout" : 0.5
    }

    def __init__(self, hparams):
        super(Net, self).__init__()

        self.__dict__ .update(self.DEFAULTS)
        self.__dict__.update(hparams)

        self.conv_start = GCNConv(3, self.hidden_dim)
        self.mid_conv = nn.ModuleList([GCNConv(self.hidden_dim, self.hidden_dim) for _ in range(self.depth - 1)])
        self.conv_end = GCNConv(self.hidden_dim, 1)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv_start(x, edge_index)
        x = self.act(x)
        
        for layer in self.mid_conv:
            x = self.drop(x)
            x = self.act(layer(x, edge_index))

        x = self.drop(x)
        x = self.conv_end(x, edge_index)

        return torch.sigmoid(x)
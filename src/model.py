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

def train(X, y, net, train_idx, val_idx, optimizer, epochs = 100, early_stopping = 20, verbose = True):
    criterion = nn.BCELoss()

    max_acc = 0
    epochs_without_increase = 0
    state = None

    for epoch in range(1, epochs + 1):

        net.train()
        optimizer.zero_grad()
        output = net(X)
        loss = criterion(output.flatten()[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        if verbose:
            print("Epoch " + str(epoch) + " finished. Train loss equal to " + str(loss.item()) + ".")

        net.eval()
        with torch.no_grad():
            output = net(X)
        max_index = torch.where(output.flatten() > 0.5, 1, 0)
        acc = (max_index == y)[val_idx].sum().item() / len(val_idx)

        if verbose:
            print("Validation accuracy equal to " + str(acc) + ".")
        
        if acc > max_acc:
            max_acc = acc
            epochs_without_increase = 0
            state = net.state_dict()
        else:
            epochs_without_increase += 1
            if epochs_without_increase == early_stopping:
                break
    
    net.load_state_dict(state)
    net.eval()

    return acc

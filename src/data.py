import numpy as np

from PIL import Image

import torch
from torch_geometric.data import Data

from src import utils

def read_image(image_path):
    im = Image.open(image_path)

    x = torch.Tensor(np.array(im))
    edge_index = utils.graph_of_image(x)

    return Data(x = x.view((-1, 3)), edge_index = edge_index.contiguous().type(torch.long))

def read_image_points(image_path, back = [255, 0, 0], not_back = [74, 207, 78]):
    im = Image.open(image_path)

    unk_sum = 255 * 3
    back_sum = sum(back)
    im_sum = sum(not_back)

    x = torch.Tensor(np.array(im)).view((-1, 3))
    x = torch.sum(x, dim = 1)
    
    idx = x != unk_sum

    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)

    return torch.where(x == back_sum, ones, zeros), idx
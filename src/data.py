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
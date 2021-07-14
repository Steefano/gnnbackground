import torch

def graph_of_image(im_tensor):
    rows, columns, _ = im_tensor.size()
    total = rows * columns

    return torch.cat(list([pixel_neighbors(i, rows, columns) for i in range(total)]), dim = 1)

def pixel_neighbors(pixel, rows, columns):
    total = rows * columns

    src = []
    tgt = []

    if pixel > columns:
        src.append(pixel)
        tgt.append(pixel - columns)
    if pixel < total - columns:
        src.append(pixel)
        tgt.append(pixel + columns)
    if pixel % columns != 0:
        src.append(pixel)
        tgt.append(pixel - 1)
    if (pixel + 1) % columns != 0:
        src.append(pixel)
        tgt.append(pixel + 1)

    return torch.Tensor([src, tgt]).type(torch.int)
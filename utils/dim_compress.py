import torch


def dim_compress(inp: torch.Tensor, dims):
    """
    Compress the given dims into a single dimension (order preserved).
    """
    if isinstance(dims, int):
        dims = [dims]
    dims = sorted([d % inp.ndim for d in dims])
    shape = list(inp.shape)
    new_shape = []
    i = 0
    while i < len(shape):
        if i in dims:
            size = 1
            while i < len(shape) and i in dims:
                size *= shape[i]
                i += 1
            new_shape.append(size)
        else:
            new_shape.append(shape[i])
            i += 1
    return inp.reshape(new_shape)

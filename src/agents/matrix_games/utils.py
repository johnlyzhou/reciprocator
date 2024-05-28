import torch


def state_to_idx(state: torch.Tensor):
    """
    Convert a state tensor to an index tensor.
    :param state: A tensor of shape (bsz, 2), all binary, or (-1, 0) for the first time step
    :return: A tensor of shape (bsz,) containing the index of the state
    """
    return (state[:, 0] + state[:, 1] * 2 + 1).int().flatten()

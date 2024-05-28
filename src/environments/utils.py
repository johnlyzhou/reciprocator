import torch


def rand_labels(bsz: int, grid_height: int, grid_width: int, num_positions: int):
    """
    Generates a batch of random position labels without replacement.
    :param bsz: Batch size.
    :param grid_height: Height of the square grid.
    :param grid_width: Width of the square grid.
    :param num_positions: Number of positions to generate per batch.
    """
    return torch.argsort(torch.rand(bsz, grid_height * grid_width), dim=-1)[:, :num_positions]


def label_to_coords(position_val: torch.Tensor, grid_width: int) -> torch.Tensor:
    """
    Converts a position value to a 2D grid coordinate pair.
    :param position_val: Tensor of shape (bsz, num_agents)
    :param grid_width: Width of the grid.
    :return: Tensor of shape (bsz, num_agents, 2).
    """
    return torch.stack([position_val // grid_width, position_val % grid_width], dim=-1)


def coords_to_label(coords: torch.Tensor, grid_width: int) -> torch.Tensor:
    """
    Converts a 2D grid coordinate pair to a single integer.
    :param coords: Tensor of shape (bsz, num_agents, 2).
    :param grid_width: Width of the grid.
    :return: Tensor of shape (bsz, num_agents)
    """
    return coords[..., 0] * grid_width + coords[..., 1]


def same_coords(x: torch.Tensor, y: torch.Tensor):
    """
    Checks if two positions are the same.
    :param x: Position tensor of shape (..., 2).
    :param y: Position tensor of shape (..., 2).
    :return: Boolean tensor of shape (..., 1).
    """
    return torch.all(x == y, dim=-1)


def state_to_label(state):
    """
    Converts a one-hot state tensor to a label tensor.
    :param state: Tensor of shape (bsz, num_agents * 2, grid_size, grid_size).
    :return: Tensor of shape (bsz, num_agents * 2), with values in range [0, grid_size * grid_size).
    """
    return torch.argmax(state.view(state.size(0), state.size(1), -1), dim=-1)


def label_to_state(label, grid_height: int, grid_width: int):
    """
    Converts a label tensor to a state tensor.
    :param label: Tensor of shape (bsz, n), with values in range [0, grid_height * grid_width).
    :param grid_height: The grid height of the environment.
    :param grid_width: The grid width of the environment.
    :return: Tensor of shape (bsz, n, grid_size, grid_size).
    """
    state = torch.zeros((label.size(0), label.size(1), grid_height * grid_width), dtype=torch.bool).to(label.device)
    state.scatter_(2, label.unsqueeze(dim=-1), 1)
    return state.view(label.size(0), label.size(1), grid_height, grid_width)


def coords_to_state(coords, grid_height: int, grid_width: int):
    """
    Converts a coordinate tensor to a state tensor.
    :param coords: Tensor of shape (bsz, n, 2).
    :param grid_height: The grid height of the environment.
    :param grid_width: The grid width of the environment.
    :return: Tensor of shape (bsz, n, grid_size, grid_size).
    """
    return label_to_state(coords_to_label(coords, grid_width), grid_height, grid_width)

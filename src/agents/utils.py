import torch
from torch import nn

from src.agents.memory import VoIMemory, Memory, BufferMemory


class PrintLayer(nn.Module):
    def __init__(self, name=None):
        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        if self.name is not None:
            print(self.name, x.shape)
        else:
            print(x.shape)
        return x


def compute_returns(rewards: list[torch.Tensor], gamma: float):
    """
    Compute the state values for the states in the memory. Rewards should be a list of length T with shape (bsz,)
    :return returns: Tensor of shape (T, bsz)
    """
    state_values = []
    # In this case, the stored rewards should be a list of (bsz, num_agents) of length T
    discounted_rewards_to_go = 0
    for reward in reversed(rewards):
        discounted_rewards_to_go = reward + (gamma * discounted_rewards_to_go)
        state_values.append(discounted_rewards_to_go)
    state_values = torch.stack(state_values[::-1])  # (T, bsz)

    return state_values

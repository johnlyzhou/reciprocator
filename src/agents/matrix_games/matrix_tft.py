import torch
from src.agents.agent import Agent


class IPDTFT(Agent):
    def __init__(self, bsz, device):
        super(IPDTFT, self).__init__()
        self.bsz = bsz
        self.device = device
        self.last_actions = torch.zeros(bsz, device=device)

    def act(self, state):
        return self.last_actions

    def observe(self, other_actions):
        self.last_actions = other_actions

    def update(self):
        pass

    def save(self, filename):
        pass

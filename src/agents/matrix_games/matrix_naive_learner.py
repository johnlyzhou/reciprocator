import torch
from torch import nn as nn
import torch.nn.functional as F

from src.agents.agent import Agent
from src.agents.matrix_games.matrix_ppo import MatrixPPO
from src.agents.memory import Memory


class MatrixNaiveLearner(Agent):
    def __init__(self, state_dim: int, n_latent_var: int, lr: float, gamma: float, K_epochs: int,
                 eps_clip: float, entropy_weight: float, device: torch.device, rnn: bool = False):
        super(MatrixNaiveLearner, self).__init__()
        self.rnn = rnn
        self.ppo = MatrixPPO(state_dim, n_latent_var, lr, gamma, K_epochs, eps_clip, entropy_weight, device, rnn=rnn)
        self.memory = Memory()
        self.step_count = 0
        self.episode_count = 0

    def act(self, state: torch.Tensor):
        return self.ppo.policy_old.act(state, self.memory).detach()

    def observe(self, transition: tuple):
        obs, rewards, done, info = transition
        self.memory.rewards.append(rewards.detach())
        self.step_count += 1

    def update(self):
        self.ppo.update(self.memory)
        self.memory.clear_memory()

    def reset(self):
        self.memory.clear_memory()
        self.step_count = 0
        self.episode_count += 1

        if self.rnn:
            self.ppo.policy_old.actor.reset()

    def save(self, path: str):
        self.ppo.save(path)

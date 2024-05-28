import torch

from src.agents.agent import Agent
from src.agents.memory import Memory
from src.agents.ppo import PPO


class NaiveLearner(Agent):
    def __init__(self, state_dim: tuple, num_actions: int, n_latent_var: int, lr: float, gamma: float,
                 K_epochs: int, eps_clip: float, entropy_weight: float, device: torch.device, rnn: bool = False,
                 eval_mode: bool = False):
        super(NaiveLearner, self).__init__()
        self.rnn = rnn
        self.ppo = PPO(state_dim, num_actions, n_latent_var, lr, gamma, K_epochs, eps_clip, entropy_weight, device,
                       rnn=rnn)
        self.eval_mode = eval_mode
        self.memory = Memory()
        self.step_count = 0
        self.episode_count = 0

    def act(self, state: torch.Tensor):
        """State, action and logprob are stored in the memory."""
        return self.ppo.policy_old.act(state, self.memory).detach()

    def observe(self, transition: tuple):
        obs, rewards, done, info = transition
        self.memory.rewards.append(rewards.detach())
        self.step_count += 1

    def update(self):
        if not self.eval_mode:
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

    def load(self, filename):
        self.ppo.load(filename)

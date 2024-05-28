"""Code adapted from https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py"""
import torch
import torch.nn as nn
from tqdm import tqdm

from src.agents.components import ConvActorCritic
from src.agents.memory import Memory, VoIMemory
from src.agents.utils import compute_returns


class PPO:
    def __init__(self, state_dim: tuple, num_actions: int, n_latent_var: int, lr: float, gamma: float, K_epochs: int,
                 eps_clip: float, entropy_weight: float, device: torch.device, rnn: bool = False):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_weight = entropy_weight
        self.K_epochs = K_epochs
        self.rnn = rnn

        self.policy = ConvActorCritic(state_dim, num_actions, n_latent_var, n_latent_var // 4, rnn).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ConvActorCritic(state_dim, num_actions, n_latent_var, n_latent_var // 4, rnn).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.critic_loss = nn.MSELoss()

    def update(self, memory: Memory or VoIMemory):
        """
        Update the policy using the PPO algorithm. Note that the memory should be cleared after calling this in the
        main game loop to stay on-policy.
        """
        # Compute discounted rewards-to-go (state values) to train value function
        state_values = compute_returns(memory.get_rewards(), self.gamma).flatten(end_dim=1)  # (T, bsz)
        # Normalize state values
        state_values = (state_values - state_values.mean()) / (state_values.std() + 1e-5)

        # Convert lists to Tensors
        old_states = torch.stack(memory.get_states()).detach()  # (T, bsz, *state_shape)
        old_actions = torch.stack(memory.get_actions()).detach()
        old_logprobs = torch.stack(memory.get_logprobs()).detach()

        if not self.rnn:
            old_states = old_states.flatten(end_dim=1)
        old_logprobs = old_logprobs.flatten(end_dim=1)

        # Optimize policy for K epochs:
        for _ in tqdm(range(self.K_epochs)):
            # Evaluate old actions and values:
            logprobs, state_values_hat, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # Find the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Compute the surrogate loss:
            advantages = state_values - state_values_hat.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Total loss is a weighted sum of the surrogate loss, value (critic) loss, and entropy (exploration) loss:
            loss = -torch.min(surr1, surr2) + 0.5 * self.critic_loss(state_values_hat, state_values) - self.entropy_weight * dist_entropy

            # Take a gradient step on the average loss across the batch of sampled trajectories:
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy, which will be used for acting:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, filename):
        torch.save(
            {
                "actor_critic": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=torch.device("cpu"))
        self.policy.load_state_dict(checkpoint["actor_critic"])
        self.policy_old.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

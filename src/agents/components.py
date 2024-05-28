import torch
from torch import nn as nn
from torch.distributions import Categorical

from src.agents.memory import Memory


class StateEncoder(nn.Module):
    def __init__(self, input_shape: tuple, n_latent_var: int, n_out_channels: int, kernel_size: int = None):
        super(StateEncoder, self).__init__()
        if kernel_size is None:
            kernel_size = min(input_shape[1], input_shape[2])
            if kernel_size % 2 == 0:
                kernel_size += 1
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], n_out_channels, kernel_size=kernel_size, stride=1, padding="same",
                      padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(n_out_channels, n_out_channels, kernel_size=kernel_size, stride=1, padding="same",
                      padding_mode="circular"),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(n_out_channels * input_shape[1] * input_shape[2], n_latent_var),
            nn.ReLU(),
        )

    def forward(self, state_bs: torch.Tensor):
        return self.encoder(state_bs)


class Actor(nn.Module):
    """Discrete Actor only."""
    def __init__(self, input_shape: tuple, action_dim: int, n_latent_var: int, n_out_channels: int):
        """
        :param input_shape: Shape of the state space.
        :param action_dim: Number of dimensions of the action space.
        :param n_latent_var: Number of hidden units in the hidden linear layer.
        :param n_out_channels: Number of output channels for the convolutional layers.
        """
        super(Actor, self).__init__()
        self.state_encoder = StateEncoder(input_shape, n_latent_var, n_out_channels)
        # Set up actor architecture
        self.actor = nn.Sequential(
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state_bs: torch.Tensor):
        state_encoding = self.state_encoder(state_bs)
        return self.actor(state_encoding)

    def act(self, state_bs: torch.Tensor, memory: Memory):
        """
        Produce an action based on the current state and policy, and store the states, logprobs, and value estimates.
        Note that this will be called with torch.no_grad(), so purely for forward computation and not optimization.
        :param state_bs: Batch of states, of shape (batch_size, *state_shape)
        :param memory: Memory object.
        :return: Batch of actions, of shape (batch_size, *action_shape).
        """
        action_probs_bs = self(state_bs)
        dist = Categorical(action_probs_bs)
        action_bs = dist.sample()

        memory.states.append(state_bs)
        memory.actions.append(action_bs)
        memory.logprobs.append(dist.log_prob(action_bs))

        return action_bs.squeeze(0)


class Critic(nn.Module):
    """Discrete Critic only."""
    def __init__(self, input_shape: tuple, n_latent_var: int, n_out_channels: int, output_dims: int = 1):
        """
        :param input_shape: Shape of the state space.
        :param n_latent_var: Number of hidden units in the hidden linear layer.
        :param n_out_channels: Number of output channels for the convolutional layers.
        """
        super(Critic, self).__init__()
        self.state_encoder = StateEncoder(input_shape, n_latent_var, n_out_channels)
        # Set up critic architecture
        self.critic = nn.Sequential(
            nn.Linear(n_latent_var, output_dims),
        )

    def forward(self, state_bs: torch.Tensor):
        state_encoding = self.state_encoder(state_bs)
        return self.critic(state_encoding).squeeze()


class RecurrentActor(nn.Module):
    def __init__(self, input_shape: tuple, num_actions: int, n_latent_var: int, n_out_channels: int):
        """
        :param input_shape: Shape of the state space.
        :param num_actions: Number of possible discrete actions.
        :param n_latent_var: Number of hidden units in the hidden linear layer.
        :param n_out_channels: Number of output channels for the convolutional layers.
        """
        super(RecurrentActor, self).__init__()
        self.state_encoder = StateEncoder(input_shape, n_latent_var, n_out_channels)
        # Set up actor architecture
        self.actor_rnn = nn.GRU(n_latent_var, n_latent_var, num_layers=1, batch_first=False)
        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_latent_var, num_actions),
            nn.Softmax(dim=-1),
        )
        self.actor_hidden_state = None

    def forward(self, state_bs: torch.Tensor):
        """
        :param state_bs: Batch of states, either of shape (batch_size, *state_shape) during action selection, or
        (T, batch_size, *state_shape) during training.
        """
        if state_bs.ndim == 5:
            T, bsz, _, _, _ = state_bs.shape
            state_bs = state_bs.flatten(end_dim=1)
            state_encoding = self.state_encoder(state_bs).view(T, bsz, -1)  # (T, bsz, n_latent_var)
        else:
            state_encoding = self.state_encoder(state_bs).unsqueeze(dim=0)  # (1, bsz, n_latent_var)

        if self.actor_hidden_state is not None:
            out, self.actor_hidden_state = self.actor_rnn(state_encoding, self.actor_hidden_state)
        else:
            out, self.actor_hidden_state = self.actor_rnn(state_encoding)
        return self.actor(out).squeeze()

    def act(self, state_bs: torch.Tensor, memory: Memory):
        """
        Produce an action based on the current state and policy, and store the states, logprobs, and value estimates.
        Note that this will be called with torch.no_grad(), so purely for forward computation and not optimization.
        :param state_bs: Batch of states, of shape (batch_size, *state_shape)
        :param memory: Memory object.
        :return: Batch of actions, of shape (batch_size, *action_shape).
        """
        action_probs_bs = self(state_bs)
        dist = Categorical(action_probs_bs)
        action_bs = dist.sample()

        memory.states.append(state_bs)
        memory.actions.append(action_bs)
        memory.logprobs.append(dist.log_prob(action_bs))

        return action_bs

    def reset(self):
        self.actor_hidden_state = None


class RecurrentCritic(nn.Module):
    def __init__(self, input_shape: tuple, n_latent_var: int, n_out_channels: int):
        """
        :param input_shape: Shape of the state space.
        :param n_latent_var: Number of hidden units in the hidden linear layer.
        :param n_out_channels: Number of output channels for the convolutional layers.
        """
        super(RecurrentCritic, self).__init__()
        self.state_encoder = StateEncoder(input_shape, n_latent_var, n_out_channels)
        # Set up critic architecture
        self.critic_rnn = nn.GRU(n_latent_var, n_latent_var, num_layers=1, batch_first=False)
        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_latent_var, 1),
        )

    def forward(self, state_bs: torch.Tensor):
        if state_bs.ndim == 5:
            T, bsz, _, _, _ = state_bs.shape
            state_bs = state_bs.flatten(end_dim=1)
            state_encoding = self.state_encoder(state_bs).view(T, bsz, -1)  # (T, bsz, n_latent_var)
        else:
            state_encoding = self.state_encoder(state_bs).unsqueeze(dim=0)  # (1, bsz, n_latent_var)
        out, _ = self.critic_rnn(state_encoding)
        return self.critic(out).squeeze()


class ConvActorCritic(nn.Module):
    """Discrete Actor Critic only."""
    def __init__(self, input_shape: tuple, action_dim: int, n_latent_var: int, n_out_channels: int, rnn: bool = False):
        """
        :param input_shape: Shape of the state space.
        :param action_dim: Number of dimensions of the action space.
        :param n_latent_var: Number of hidden units in the hidden linear layer.
        :param n_out_channels: Number of output channels for the convolutional layers.
        """
        super(ConvActorCritic, self).__init__()
        self.rnn = rnn
        if rnn:
            self.actor = RecurrentActor(input_shape, action_dim, n_latent_var, n_out_channels)
            self.critic = RecurrentCritic(input_shape, n_latent_var, n_out_channels)
        else:
            self.actor = Actor(input_shape, action_dim, n_latent_var, n_out_channels)
            self.critic = Critic(input_shape, n_latent_var, n_out_channels)

    def forward(self):
        raise NotImplementedError

    def act(self, state_bs: torch.Tensor, memory: Memory):
        """
        Produce an action based on the current state and policy, and store the states, logprobs, and value estimates.
        Note that this will be called with torch.no_grad(), so purely for forward computation and not optimization.
        :param state_bs: Batch of states, of shape (batch_size, *state_shape)
        :param memory: Memory object.
        :return: Batch of actions.
        """
        return self.actor.act(state_bs, memory)

    def evaluate(self, state_bs: torch.Tensor, action_bs: torch.Tensor):
        """
        Called in the PPO update step.
        :param state_bs: Batch of states, of shape (bsz, *state_shape) or (T, bsz, *state_shape) for RNN.
        :param action_bs: Batch of actions, of shape (bsz, *action_shape) or (T, bsz, *state_shape) for RNN.
        :return: Batch of logprobs, state values from critic, and entropy of the action distribution.
        """
        if self.rnn:
            self.actor.reset()

        action_probs_ba = self.actor(state_bs)

        action_bs = action_bs.flatten(end_dim=1)
        if self.rnn:
            action_probs_ba = action_probs_ba.flatten(end_dim=1)

        dist = Categorical(action_probs_ba)
        action_logprobs = dist.log_prob(action_bs)
        dist_entropy = dist.entropy()
        state_value = self.critic(state_bs).squeeze(-1)  # (T, bsz, 1) -> (T, bsz)

        if self.rnn:
            state_value = state_value.flatten(end_dim=1)  # (T * bsz)

        return action_logprobs, state_value, dist_entropy

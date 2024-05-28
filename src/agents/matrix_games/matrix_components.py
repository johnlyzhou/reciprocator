import torch
from torch import nn as nn
from torch.distributions import Bernoulli

from src.agents.memory import Memory


class MatrixStateEncoder(nn.Module):
    def __init__(self, input_dims: int, n_latent_var: int):
        super(MatrixStateEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
        )

    def forward(self, state_bs: torch.Tensor):
        return self.encoder(state_bs)


class BernoulliActor(nn.Module):
    def __init__(self, input_dims: int, n_latent_var: int, rnn: bool = False):
        """
        :param input_dims: Shape of the state space.
        :param n_latent_var: Number of hidden units in the hidden linear layer.
        """
        super(BernoulliActor, self).__init__()
        self.state_encoder = MatrixStateEncoder(input_dims, n_latent_var)
        self.rnn = rnn
        # Set up actor architecture
        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_latent_var, 1),
            nn.Sigmoid(),
        )

        if self.rnn:
            self.actor_rnn = nn.GRU(n_latent_var, n_latent_var, num_layers=1, batch_first=False)
            self.actor_hidden_state = None

    def forward(self, state_bs: torch.Tensor):
        """
        :param state_bs: Batch of states, either of shape (batch_size, *state_shape) during action selection, or
        (T, batch_size, *state_shape) during training.
        """
        state_encoding = self.state_encoder(state_bs)  # (1, bsz, n_latent_var)
        if self.rnn:
            if self.actor_hidden_state is not None:
                state_encoding, self.actor_hidden_state = self.actor_rnn(state_encoding, self.actor_hidden_state)
            else:
                state_encoding, self.actor_hidden_state = self.actor_rnn(state_encoding)
        return self.actor(state_encoding)

    def act(self, state_bs: torch.Tensor, memory: Memory):
        """
        Produce an action based on the current state and policy, and store the states, logprobs, and value estimates.
        Note that this will be called with torch.no_grad(), so purely for forward computation and not optimization.
        :param state_bs: Batch of states, of shape (batch_size, *state_shape)
        :param memory: Memory object.
        :return: Batch of actions, of shape (batch_size, *action_shape).
        """
        action_probs_bs = self(state_bs).squeeze()
        dist = Bernoulli(action_probs_bs)
        action_bs = dist.sample().squeeze()

        memory.states.append(state_bs)
        memory.actions.append(action_bs)
        memory.logprobs.append(dist.log_prob(action_bs).squeeze())

        return action_bs

    def reset(self):
        self.actor_hidden_state = None


class Critic(nn.Module):
    def __init__(self, input_dims: int, n_latent_var: int, rnn: bool = False):
        """
        :param input_dims: Shape of the state space.
        :param n_latent_var: Number of hidden units in the hidden linear layer.
        """
        super(Critic, self).__init__()
        self.rnn = rnn
        self.state_encoder = MatrixStateEncoder(input_dims, n_latent_var)
        # Set up critic architecture
        self.critic_rnn = nn.GRU(n_latent_var, n_latent_var, num_layers=1, batch_first=False)
        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_latent_var, 1),
        )

    def forward(self, state_bs: torch.Tensor):
        state_encoding = self.state_encoder(state_bs)  # (1, bsz, n_latent_var)
        if self.rnn:
            state_encoding, _ = self.critic_rnn(state_encoding)
        return self.critic(state_encoding)


class IPDActorCritic(nn.Module):
    """Discrete Actor Critic only."""
    def __init__(self, input_dims: int, n_latent_var: int, rnn: bool = False):
        """
        :param input_dims: Dimensions of the state space.
        :param n_latent_var: Number of hidden units in the hidden linear layer.
        """
        super(IPDActorCritic, self).__init__()
        self.rnn = rnn
        self.actor = BernoulliActor(input_dims, n_latent_var, rnn=rnn)
        self.critic = Critic(input_dims, n_latent_var, rnn=rnn)

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

        action_probs_ba = self.actor(state_bs).squeeze()

        if self.rnn:
            action_probs_ba = action_probs_ba.flatten(end_dim=1)

        dist = Bernoulli(action_probs_ba)
        action_logprobs = dist.log_prob(action_bs)
        dist_entropy = dist.entropy()
        state_value = self.critic(state_bs).squeeze()

        if self.rnn:
            state_value = state_value.flatten(end_dim=1)

        return action_logprobs, state_value, dist_entropy

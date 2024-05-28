from collections import deque, namedtuple

import torch
from torch.utils.data import TensorDataset

# All fields are tensors. State: (T, bsz, *state_dim), actions: (T, bsz, num_agents, action_dim),
# rewards: (T, bsz, num_agents), returns: (T, bsz, num_agents)
Transition = namedtuple('Transition', ('state', 'timestep', 'actions', 'reward', 'state_value'))


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

    def __len__(self):
        return len(self.actions)

    def get_actions(self):
        return self.actions

    def get_states(self):
        return self.states

    def get_logprobs(self):
        return self.logprobs

    def get_rewards(self):
        return self.rewards


class VoIMemory:
    """A duplicate of the PPO Memory class, with additional attributes for other agents' actions and states."""

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.other_states = []
        self.other_actions = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.other_states[:]
        del self.other_actions[:]

    def __len__(self):
        return len(self.actions)

    def get_actions(self):
        return self.actions

    def get_states(self):
        return self.states

    def get_logprobs(self):
        return self.logprobs

    def get_rewards(self):
        return self.rewards

    def get_other_states(self):
        return self.other_states

    def get_other_actions(self):
        return self.other_actions


class BufferMemory:
    """Memory implemented with deque."""

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.actions = deque([], maxlen=maxlen)
        self.states = deque([], maxlen=maxlen)
        self.logprobs = deque([], maxlen=maxlen)
        self.rewards = deque([], maxlen=maxlen)
        self.other_states = deque([], maxlen=maxlen)
        self.other_actions = deque([], maxlen=maxlen)

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.other_states[:]
        del self.other_actions[:]

    def __len__(self):
        return len(self.actions)

    def __add__(self, other):
        """Concatenate two VoIMemories."""
        new = BufferMemory(self.maxlen)
        new.actions = self.actions + other.actions
        new.states = self.states + other.states
        new.logprobs = self.logprobs + other.logprobs
        new.rewards = self.rewards + other.rewards
        new.other_states = self.other_states + other.other_states
        new.other_actions = self.other_actions + other.other_actions
        return new

    def __iadd__(self, other):
        """Concatenate another VoIMemory to self."""
        self.actions += other.actions
        self.states += other.states
        self.logprobs += other.logprobs
        self.rewards += other.rewards
        self.other_states += other.other_states
        self.other_actions += other.other_actions
        return self

    def get_actions(self):
        return list(self.actions)

    def get_states(self):
        return list(self.states)

    def get_logprobs(self):
        return list(self.logprobs)

    def get_rewards(self):
        return list(self.rewards)

    def get_other_states(self):
        return list(self.other_states)

    def get_other_actions(self):
        return list(self.other_actions)


class TargetMemory:
    def __init__(self, maxlen: int, device: torch.device = torch.device('cpu')):
        """
        Memory implemented with deque. Entries to the deques are blocks of samples as Tensors of shape
        (num_samples, ...) where num_samples == bsz. These will then be stacked together on retrieval -
        more efficient than casting to a list and appending an individual entry for every sample. Note that
        state values should be computed from the full episode's data before inserting samples from each individual
        timestep.
        :param maxlen: Maximum length of the memory.
        :param device: Device to store the memory on.
        """
        self.maxlen = maxlen
        self.states = deque([], maxlen=maxlen)
        self.actions = deque([], maxlen=maxlen)
        self.rewards = deque([], maxlen=maxlen)
        self.returns = deque([], maxlen=maxlen)
        self.device = device

    def clear_memory(self):
        del self.inputs[:]
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.returns[:]

    def __len__(self):
        return len(self.rewards)

    def push_states(self, states: torch.Tensor):
        self.states.append(states.to(self.device))

    def push_actions(self, actions: torch.Tensor):
        self.actions.append(actions.to(self.device))

    def push_rewards(self, rewards: torch.Tensor):
        self.rewards.append(rewards.to(self.device))

    def push_returns(self, returns: torch.Tensor):
        self.returns.append(returns.to(self.device))

    def get_states(self, device: torch.device = None):
        """Returns states as a Tensor of shape (T, bsz * buffer_size, *state_dim)."""
        if device is None:
            device = self.device
        return torch.cat(list(self.states), dim=1).to(device)

    def get_actions(self, device: torch.device = None):
        """Returns actions as a Tensor of shape (T, bsz * buffer_size, num_players)."""
        if device is None:
            device = self.device
        return torch.cat(list(self.actions), dim=1).to(device)

    def get_rewards(self, device: torch.device = None):
        """Returns rewards as a Tensor of shape (T, bsz * buffer_size, num_players)."""
        if device is None:
            device = self.device
        return torch.cat(list(self.rewards), dim=1).to(device)

    def get_returns(self, device: torch.device = None):
        """Returns returns as a Tensor of shape (T, bsz * buffer_size, num_players)."""
        if device is None:
            device = self.device
        return torch.cat(list(self.returns), dim=1).to(device)

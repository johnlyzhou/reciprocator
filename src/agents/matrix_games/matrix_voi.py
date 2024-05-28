from collections import deque

import torch

from src.agents.memory import Memory
from src.agents.matrix_games.utils import state_to_idx
from src.environments.ipd import REWARD_MATRIX


class MatrixVOI:
    # (p0 action index, p1 action index, p0/1 reward index)
    NUM_PLAYERS = 2

    def __init__(self, discount_factor: float, buffer_len: int, device: torch.device):
        self.gamma = discount_factor
        self.reward_matrix = REWARD_MATRIX.to(device)
        self.device = device
        self.episode_memory = Memory()  # Just to store both actions for the reciprocal reward
        self.state_buffer = deque(maxlen=buffer_len)
        self.action_buffers = [deque(maxlen=buffer_len) for _ in range(self.NUM_PLAYERS)]
        self.reward_buffer = deque(maxlen=buffer_len)
        self.reward_expectation = {}
        self._init_baseline()

    def update_actions(self, action_batch: torch.Tensor):
        """
        Update the action buffer.
        :param action_batch: A batch of binary actions of shape (bsz, 2) taken by the agents in the game
        (0: cooperate, 1: defect)
        """
        for i in range(self.NUM_PLAYERS):
            self.action_buffers[i].extend(list(action_batch[:, i].int().detach().cpu().numpy()))
        self.episode_memory.actions.append(action_batch)

    def update_states(self, state_batch: torch.Tensor):
        """
        Update the state buffer.
        :param state_batch: A batch of states of shape (bsz, 2)
        """
        self.episode_memory.states.append(state_batch)
        self.state_buffer.extend(list(state_batch.int().detach()))

    def update_rewards(self, reward_batch: torch.Tensor):
        """
        Update the reward buffer.
        :param reward_batch: A batch of rewards of shape (bsz, 2)
        """
        self.episode_memory.rewards.append(reward_batch)
        self.reward_buffer.extend(list(reward_batch.int().detach()))

    def update_baselines(self):
        """
        Update the baseline reward expectations for all pairs of agents.
        """
        for i in range(self.NUM_PLAYERS):
            for j in range(self.NUM_PLAYERS):
                self._update_baseline(i, j)

    def _init_baseline(self):
        for influencer_idx in range(2):
            for influenced_idx in range(2):
                reward_structure = self.reward_matrix[..., influenced_idx]  # (2, 2)
                p_defects = torch.ones(5, device=self.device) * 0.5
                reward_expectation = torch.zeros((5, 2), device=self.device)
                for i in range(5):
                    if influencer_idx == 0:
                        # We're marginalizing out the influencer, so we consider the action of the influenced
                        reward_expectation[i] = (reward_structure[0, :] * (1 - p_defects[i]) + reward_structure[1, :] *
                                                 p_defects[i])
                    else:
                        reward_expectation[i] = (reward_structure[:, 0] * (1 - p_defects[i]) + reward_structure[:, 1] *
                                                 p_defects[i])
                self.reward_expectation[(influencer_idx, influenced_idx)] = reward_expectation

    def _update_baseline(self, influencer_idx: int, influenced_idx: int):
        reward_structure = self.reward_matrix[..., influenced_idx]  # (2, 2)
        # Note that this assumes the other agent is Markov and selecting actions i.i.d. at each timestep
        buffer_state_idxs = state_to_idx(torch.stack(list(self.state_buffer))).to(self.device)
        influencer_buffer_actions = torch.tensor(self.action_buffers[influencer_idx])
        p_defects = torch.zeros(5, device=self.device)
        reward_expectation = torch.zeros((5, 2), device=self.device)
        for i in range(5):
            p_defects[i] = influencer_buffer_actions[torch.where(buffer_state_idxs == i)].float().mean()
            if influencer_idx == 0:
                # We're marginalizing out the influencer, so we consider the action of the influenced in the dim
                reward_expectation[i] = (reward_structure[0, :] * (1 - p_defects[i]) + reward_structure[1, :] *
                                         p_defects[i])
            else:
                reward_expectation[i] = (reward_structure[:, 0] * (1 - p_defects[i]) + reward_structure[:, 1] *
                                         p_defects[i])
        print(f"Cooperate probs {influencer_idx}:", 1 - p_defects)
        self.reward_expectation[(influencer_idx, influenced_idx)] = reward_expectation

    def estimate_baseline(self, influencer_idx: int, influenced_idx: int, states: torch.Tensor, actions: torch.tensor):
        """
        Estimate the counterfactual baseline value.
        :param influencer_idx: The index of the player whose actions are being marginalized
        :param influenced_idx: The index of the player whose rewards are being estimated
        :param states: A tensor of shape (bsz, 2) or (bsz, T, 2) containing the state (of dim 2)
        :param actions: A tensor of shape (bsz, 2) or (bsz, T, 2) containing the actions of each agent
        :return: A tensor of shape (bsz,) or (bsz, T) containing the expected reward for the influenced agent given the
        state and action of the influenced agent (marginalizing out the influencer)
        """
        cf_actions = actions[..., ~influencer_idx]  # Not influencer
        if (influencer_idx, influenced_idx) not in self.reward_expectation:
            self._update_baseline(influencer_idx, influenced_idx)
        reward_expectation = self.reward_expectation[(influencer_idx, influenced_idx)]

        if cf_actions.dim() == 2:
            bsz, T = cf_actions.shape
            flat_actions = cf_actions.view(-1)
            flat_states = states.view(bsz * T, 2)
            state_idxs = state_to_idx(flat_states)
            return reward_expectation[state_idxs, flat_actions].view(bsz, T)
        else:
            state_idxs = state_to_idx(states)
            return reward_expectation[state_idxs, cf_actions]

    def compute_voi(self, influencer_idx: int, influenced_idx: int, states: torch.tensor, actions: torch.tensor,
                    rewards: torch.tensor, return_stepwise: bool):
        """
        Compute the value of interaction for the given action or series of actions. Note that because the Markov game
        has a stationary state, the future state values should be equal and cancel out.
        :param influencer_idx: The index of the player to marginalize out for the baseline.
        :param influenced_idx: The index of the player whose reward is being influenced.
        :param states: A tensor of shape (bsz, 2) or (bsz, T, 2) containing the state
        :param actions: A tensor of shape (bsz, 2) or (bsz, T, 2) containing the action of both agents
        :param rewards: A tensor of shape (bsz, 2) or (bsz, T, 2) containing the rewards of both agents
        :param return_stepwise: Return the individual VoI at each timestep (not accumulated over time)
        :return: A tensor of shape (bsz,) containing the VoI for each agent at each timestep, or (bsz, T) containing
            the n-step VoI at each timestep, or (bsz, T) containing the one-step VoI at each timestep if return_stepwise
        """
        actions = actions.int()
        baseline = self.estimate_baseline(influencer_idx, influenced_idx, states, actions)
        if actions.dim() == 2 or return_stepwise:
            # Return influenced agent's rewards - baseline expectation marginalizing out the influencer's action
            return rewards[..., influenced_idx] - baseline, rewards, baseline
        else:
            bsz, T, _ = actions.shape
            n_step_voi = torch.zeros((bsz, T + 1), device=self.device)
            for t in range(1, T + 1):
                one_step_voi = rewards[:, t - 1, influenced_idx] - baseline[:, t - 1]
                n_step_voi[:, t] = self.gamma ** t * one_step_voi + n_step_voi[:, t - 1]
            return n_step_voi[:, 1:]

    def reset(self):
        self.episode_memory.clear_memory()

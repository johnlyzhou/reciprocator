import torch

from src.agents.agent import Agent
from src.agents.memory import Memory
from src.agents.matrix_games.matrix_ppo import MatrixPPO
from src.agents.matrix_games.matrix_voi import MatrixVOI


class MatrixReciprocator(Agent):
    def __init__(self, state_dim: int, n_latent_var: int, lr: float, gamma: float, K_epochs: int, eps_clip: float,
                 entropy_weight: float, player_idx: int, reciprocal_reward_weight: int, device: torch.device,
                 batch_size: int = 512, max_steps: int = 32, buffer_episodes: int = 5, target_update_period: int = 5):
        super(MatrixReciprocator, self).__init__()
        self.state_dim = state_dim
        self.num_actions = 2
        self.num_agents = 2
        self.episode_len = max_steps
        self.bsz = batch_size

        self.device = device
        self.n_latent_var = n_latent_var
        self.lr = lr
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.reciprocal_reward_weight = reciprocal_reward_weight

        self.step_count = 0
        self.episode_count = -1

        # Initialize basic policy components
        self.index = player_idx
        self.target_update_period = target_update_period
        self.ppo = MatrixPPO(self.state_dim, n_latent_var, self.lr, gamma, K_epochs, eps_clip, entropy_weight, device,
                             rnn=False)
        self.memory = Memory()
        self.influence_estimator = MatrixVOI(gamma, self.episode_len * self.bsz * buffer_episodes, device)

        self.episode_log = {}

    def act(self, state: torch.Tensor):
        """State, action and logprob are stored in the memory."""
        return self.ppo.policy_old.act(state, self.memory).detach()

    def observe(self, transition: tuple):
        """This should be a joint transition."""
        obs, actions, rewards, done, info = transition
        self.influence_estimator.update_states(obs)
        self.influence_estimator.update_actions(actions)
        self.influence_estimator.update_rewards(rewards)
        self.memory.rewards.append(rewards[:, self.index])
        self.step_count += 1

    def reset(self):
        """Behavior called at the beginning of every new episode."""
        self.memory.clear_memory()
        self.step_count = 0
        self.episode_count += 1
        self.influence_estimator.reset()
        self.ppo.policy_old.actor.reset()

    def compute_reciprocal_reward(self, agent_idx: int):
        """
        Compute the intrinsic reward for the agent and update reward in memory.
        :param agent_idx: Index of the agent for which to compute the intrinsic reward
        :return: Tensor (bsz, T) containing the reciprocal reward
        """
        states = torch.stack(self.influence_estimator.episode_memory.states, dim=1)  # (bsz, T, 2)
        memory_states = torch.stack(self.memory.states, dim=1)
        actions = torch.stack(self.influence_estimator.episode_memory.actions, dim=1)  # (bsz, T, 2)
        rewards = torch.stack(self.influence_estimator.episode_memory.rewards, dim=1)
        grudge, rewards, baseline = self.influence_estimator.compute_voi(1 - agent_idx, agent_idx, states, actions,
                                                                         rewards, return_stepwise=True)
        # We offset by 1 time step at the beginning, since we reward the current action for addressing the last
        #  timestep's grudge/debit (the current grudge/debit factors in other agents' current actions, which
        #  we wouldn't have had access to while choosing our current action
        grudge = grudge.roll(1, dims=1)
        grudge[:, 0] = 0
        # debit_voi, _, _ = self.influence_estimator.compute_voi(1 - agent_idx, 1 - agent_idx, states, actions,
        #                                                        rewards, return_stepwise=True)
        # # reciprocal_voi, _, _ = self.influence_estimator.compute_voi(agent_idx, 1 - agent_idx, states, actions,
        # #                                                             rewards, return_stepwise=True)
        # debit = debit_voi  # + reciprocal_voi
        # debit = debit.roll(1, dims=1)
        # debit[:, 0] = 0

        reciprocal_voi_stepwise, _, _ = self.influence_estimator.compute_voi(agent_idx, 1 - agent_idx, states, actions,
                                                                             rewards, return_stepwise=True)

        reciprocal_reward = grudge * reciprocal_voi_stepwise

        # ---- #
        # print("COMPARING")
        # for i in range(actions.size(1)):
        #     print("State:", states[0, i],
        #           "Memory state:", memory_states[0, i],
        #           "Action:", actions[0, i, 0].item(),
        #           "Reward:", rewards[0, i, 0].item(),
        #           "Expected:", baseline[0, i].item(),
        #           "Grudge:", grudge[0, i].item(), "Debit:",
        #           debit[0, i].item(), "Reciprocal VOI:", reciprocal_voi_stepwise[0, i].item(), "Reciprocal reward:",
        #           reciprocal_reward[0, i].item())
        #
        # # print("Actions: ", actions[0, :])
        # print(f"Reciprocal reward: {reciprocal_reward[0, :]}"
        #       f"\nGrudge - debit: {grudge[0, :] - debit[0, :]} "
        #       f"\nDebit: {debit[0, :]} "
        #       f"\nVoI: {reciprocal_voi_stepwise[0, :]}")
        return reciprocal_reward, grudge, reciprocal_voi_stepwise

    def get_episode_log(self):
        log = self.episode_log.copy()
        self.episode_log = {}
        return log

    def update(self):
        """Behavior called at the end of every new episode."""
        if self.episode_count % self.target_update_period == 0:
            self.influence_estimator.update_baselines()
        rr, grudge, stepwise_voi = self.compute_reciprocal_reward(self.index)
        self.episode_log[f"{self.index}_mean_reciprocal_reward_across_batches"] = rr.mean().item() * self.reciprocal_reward_weight
        self.episode_log[f"{self.index}_mean_end_grudge_across_batches"] = grudge[:, -1].mean().item()
        self.episode_log[f"{self.index}_mean_voi_across_batches"] = stepwise_voi.mean().item()
        # print("UPDATE")
        for t in range(len(self.memory.rewards)):
            # print(self.memory.rewards[t][0], rr[0, t] * self.reciprocal_reward_weight)
            self.memory.rewards[t] += rr[:, t] * self.reciprocal_reward_weight
            # print(self.memory.rewards[t][0])

        self.ppo.update(self.memory)
        self.memory.clear_memory()

    def save(self, filename: str):
        self.ppo.save(filename)

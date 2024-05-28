import omegaconf
import torch

from src.agents.agent import Agent
from src.agents.coins_voi import CoinsVoI
from src.agents.memory import Memory
from src.agents.ppo import PPO
from src.agents.voi import ValueOfInfluence


class Reciprocator(Agent):

    def __init__(self, state_dim: tuple, num_actions: int, n_latent_var: int, lr: float, gamma: float,
                 K_epochs: int, eps_clip: float, entropy_weight: float, num_agents: int, player_idx: int,
                 influence_estimator: ValueOfInfluence or CoinsVoI, reciprocal_reward_weight: int or tuple,
                 reciprocal_reward_type: str, device: torch.device, normalize_reciprocal_reward: bool,
                 rnn: bool = False):
        super(Reciprocator, self).__init__()
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.n_latent_var = n_latent_var
        self.lr = lr
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.device = device
        self.rnn = rnn

        self.step_count = 0
        self.episode_count = 0
        self.current_obs = None

        # Initialize basic policy components
        self.ppo = PPO(state_dim, num_actions, n_latent_var, lr, gamma, K_epochs, eps_clip, entropy_weight, device,
                       rnn=rnn)
        self.memory = Memory()

        # Attributes for the influence estimator
        self.index = player_idx
        self.influence_estimator = influence_estimator
        self.grow_reciprocal_reward = isinstance(reciprocal_reward_weight, omegaconf.listconfig.ListConfig)
        if self.grow_reciprocal_reward:
            if len(reciprocal_reward_weight) != 3:
                raise ValueError("Reciprocal reward weight tuple must have 3 elements: (start, end, rate).")
            self.reciprocal_reward_weight = reciprocal_reward_weight[0]
            self.reciprocal_weight_limit = reciprocal_reward_weight[1]
            self.reciprocal_weight_rate = reciprocal_reward_weight[2]
        else:
            self.reciprocal_reward_weight = reciprocal_reward_weight
        self.normalize_reciprocal_reward = normalize_reciprocal_reward

        reciprocal_reward_functions = {
            "petty": self.petty_rr,
            "signed_grudge": self.signed_grudge_rr,
            "petty_payoff": self.petty_payoff_rr,
            "signed_petty_payoff": self.signed_petty_payoff_rr,
            "signed_grudge_petty_payoff": self.signed_grudge_petty_payoff_rr,
            "grudge_minus_debit": self.grudge_minus_debit_rr,
            "signed_grudge_minus_debit": self.signed_grudge_minus_debit_rr
        }

        try:
            self.reciprocal_reward_fn = reciprocal_reward_functions[reciprocal_reward_type]
        except KeyError:
            raise ValueError(f"Invalid reciprocal reward type: {reciprocal_reward_type}, "
                             f"choose from {reciprocal_reward_functions.keys()}.")

        self.episode_log = {}

    def act(self, state: torch.Tensor):
        """State, action and logprob are stored in the memory."""
        return self.ppo.policy_old.act(state, self.memory).detach()

    def observe(self, transition: tuple):
        """This should be a joint transition."""
        obs, rewards, done, info = transition
        self.memory.rewards.append(rewards)
        self.step_count += 1

    def reset(self):
        """Behavior called at the beginning of every new episode."""
        self.memory.clear_memory()
        self.step_count = 0
        self.episode_count += 1

        if self.rnn:
            self.ppo.policy_old.actor.reset()

    def petty_rr(self, states, actions, other_index):
        T, bsz, *_ = states.shape
        self_voi_on_other = self.influence_estimator(states, actions, self.index, other_index).reshape(T, bsz)
        other_voi_on_self = self.influence_estimator(states, actions, other_index, self.index).reshape(T, bsz)

        grudge = [torch.zeros(bsz, device=self.device)]  # (bsz)

        for t in range(T):
            current_grudge = self.gamma * grudge[-1] + other_voi_on_self[t]
            grudge.append(current_grudge)

        grudge = torch.stack(grudge[:-1])  # (T, bsz)
        self.episode_log[f"{self.index}_mean_end_grudge_vs_{other_index}"] = grudge[-1].mean().item()
        self.episode_log[f"{self.index}_mean_voi_on_{other_index}"] = self_voi_on_other.mean().item()
        self.episode_log[f"mean_{other_index}_voi_on_{self.index}"] = other_voi_on_self.mean().item()
        return grudge * self_voi_on_other

    def signed_grudge_rr(self, states, actions, other_index):
        T, bsz, *_ = states.shape
        self_voi_on_other = self.influence_estimator(states, actions, self.index, other_index).reshape(T, bsz)
        other_voi_on_self = self.influence_estimator(states, actions, other_index, self.index).reshape(T, bsz)

        grudge = [torch.zeros(bsz, device=self.device)]  # (bsz)

        for t in range(T):
            current_grudge = self.gamma * grudge[-1] + other_voi_on_self[t]
            grudge.append(current_grudge)

        grudge = torch.stack(grudge[:-1])  # (T, bsz)
        self.episode_log[f"{self.index}_mean_end_grudge_vs_{other_index}"] = grudge[-1].mean().item()
        self.episode_log[f"{self.index}_mean_voi_on_{other_index}"] = self_voi_on_other.mean().item()
        self.episode_log[f"mean_{other_index}_voi_on_{self.index}"] = other_voi_on_self.mean().item()
        return torch.sign(grudge) * self_voi_on_other

    def petty_payoff_rr(self, states, actions, other_index):
        T, bsz, *_ = states.shape
        self_voi_on_other = self.influence_estimator(states, actions, self.index, other_index).reshape(T, bsz)
        other_voi_on_self = self.influence_estimator(states, actions, other_index, self.index).reshape(T, bsz)

        grudge = [torch.zeros(bsz, device=self.device)]  # (bsz)

        for t in range(T):
            current_grudge = self.gamma * grudge[-1] + other_voi_on_self[t] - self_voi_on_other[t]
            grudge.append(current_grudge)

        grudge = torch.stack(grudge[:-1])  # (T, bsz)
        self.episode_log[f"{self.index}_mean_end_grudge_vs_{other_index}"] = grudge[-1].mean().item()
        self.episode_log[f"{self.index}_mean_voi_on_{other_index}"] = self_voi_on_other.mean().item()
        self.episode_log[f"mean_{other_index}_voi_on_{self.index}"] = other_voi_on_self.mean().item()
        return grudge * self_voi_on_other

    def signed_petty_payoff_rr(self, states, actions, other_index):
        T, bsz, *_ = states.shape
        self_voi_on_other = self.influence_estimator(states, actions, self.index, other_index).reshape(T, bsz)
        other_voi_on_self = self.influence_estimator(states, actions, other_index, self.index).reshape(T, bsz)

        grudge = [torch.zeros(bsz, device=self.device)]  # (bsz)

        for t in range(T):
            current_grudge = self.gamma * grudge[-1] + other_voi_on_self[t] - self_voi_on_other[t]
            grudge.append(current_grudge)

        grudge = torch.stack(grudge[:-1])  # (T, bsz)
        self.episode_log[f"{self.index}_mean_end_grudge_vs_{other_index}"] = grudge[-1].mean().item()
        self.episode_log[f"{self.index}_mean_voi_on_{other_index}"] = self_voi_on_other.mean().item()
        self.episode_log[f"mean_{other_index}_voi_on_{self.index}"] = other_voi_on_self.mean().item()
        return torch.sign(grudge) * self_voi_on_other

    def signed_grudge_petty_payoff_rr(self, states, actions, other_index):
        T, bsz, *_ = states.shape
        self_voi_on_other = self.influence_estimator(states, actions, self.index, other_index).reshape(T, bsz)
        other_voi_on_self = self.influence_estimator(states, actions, other_index, self.index).reshape(T, bsz)

        grudge = [torch.zeros(bsz, device=self.device)]  # (bsz)

        for t in range(T):
            current_grudge = self.gamma * grudge[-1] + other_voi_on_self[t] - self_voi_on_other[t]
            grudge.append(current_grudge)

        grudge = torch.stack(grudge[:-1])  # (T, bsz)
        self.episode_log[f"{self.index}_mean_end_grudge_vs_{other_index}"] = grudge[-1].mean().item()
        self.episode_log[f"{self.index}_mean_voi_on_{other_index}"] = self_voi_on_other.mean().item()
        self.episode_log[f"mean_{other_index}_voi_on_{self.index}"] = other_voi_on_self.mean().item()
        return grudge * torch.sign(self_voi_on_other)

    def grudge_minus_debit_rr(self, states, actions, other_index):
        T, bsz, *_ = states.shape
        self_voi_on_other = self.influence_estimator(states, actions, self.index, other_index).reshape(T, bsz)
        other_voi_on_self = self.influence_estimator(states, actions, other_index, self.index).reshape(T, bsz)
        other_voi_on_other = self.influence_estimator(states, actions, other_index, other_index).reshape(T, bsz)

        grudge = [torch.zeros(bsz, device=self.device)]  # (bsz)
        debit = [torch.zeros(bsz, device=self.device)]  # (bsz)

        for t in range(T):
            current_grudge = self.gamma * grudge[-1] + other_voi_on_self[t]
            current_debit = self.gamma * debit[-1] + self_voi_on_other[t] + other_voi_on_other[t]
            grudge.append(current_grudge)
            debit.append(current_debit)

        grudge = torch.stack(grudge[:-1])  # (T, bsz)
        debit = torch.stack(debit[:-1])
        self.episode_log[f"{self.index}_mean_end_grudge_vs_{other_index}"] = grudge[-1].mean().item()
        self.episode_log[f"{self.index}_mean_voi_on_{other_index}"] = self_voi_on_other.mean().item()
        self.episode_log[f"mean_{other_index}_voi_on_{self.index}"] = other_voi_on_self.mean().item()
        self.episode_log[f"mean_end_debit_{other_index}_payable_by_{self.index}"] = debit[-1].mean().item()
        return (grudge - debit) * self_voi_on_other

    def signed_grudge_minus_debit_rr(self, states, actions, other_index):
        T, bsz, *_ = states.shape
        self_voi_on_other = self.influence_estimator(states, actions, self.index, other_index).reshape(T, bsz)
        other_voi_on_self = self.influence_estimator(states, actions, other_index, self.index).reshape(T, bsz)
        other_voi_on_other = self.influence_estimator(states, actions, other_index, other_index).reshape(T, bsz)

        grudge = [torch.zeros(bsz, device=self.device)]  # (bsz)
        debit = [torch.zeros(bsz, device=self.device)]  # (bsz)

        for t in range(T):
            current_grudge = self.gamma * grudge[-1] + other_voi_on_self[t]
            current_debit = self.gamma * debit[-1] + self_voi_on_other[t] + other_voi_on_other[t]
            grudge.append(current_grudge)
            debit.append(current_debit)

        grudge = torch.stack(grudge[:-1])  # (T, bsz)
        debit = torch.stack(debit[:-1])
        self.episode_log[f"{self.index}_mean_end_grudge_vs_{other_index}"] = grudge[-1].mean().item()
        self.episode_log[f"{self.index}_mean_voi_on_{other_index}"] = self_voi_on_other.mean().item()
        self.episode_log[f"mean_{other_index}_voi_on_{self.index}"] = other_voi_on_self.mean().item()
        self.episode_log[f"mean_end_debit_{other_index}_payable_by_{self.index}"] = debit[-1].mean().item()
        return torch.sign(grudge - debit) * self_voi_on_other

    def update_reciprocal_reward(self):
        T = len(self.memory.rewards)
        bsz = self.memory.rewards[0].shape[0]
        old_states = torch.stack(self.influence_estimator.joint_memory.states, dim=0)  # (T, bsz, *state_dim)
        old_actions = torch.stack(self.influence_estimator.joint_memory.actions, dim=0)
        reciprocal_reward = torch.zeros(T, bsz, device=self.device)

        for i in range(self.num_agents):
            if i == self.index:
                continue
            agent_rr = self.reciprocal_reward_fn(old_states, old_actions, i)
            self.episode_log[f"{self.index}_cumulative_reciprocal_reward_vs_{i}"] = agent_rr.sum(dim=0).mean().item()
            reciprocal_reward += agent_rr

        reciprocal_reward /= self.num_agents - 1

        if self.normalize_reciprocal_reward:
            reciprocal_reward = (reciprocal_reward - reciprocal_reward.mean()) / (reciprocal_reward.std() + 1e-7)

        for t in range(T):
            self.memory.rewards[t] += self.reciprocal_reward_weight * reciprocal_reward[t]

        print(f"Max reciprocal reward: {reciprocal_reward.max().item()}\n"
              f"Min reciprocal reward: {reciprocal_reward.min().item()}\n"
              f"Mean reciprocal reward: {reciprocal_reward.mean().item()}\n"
              f"Std reciprocal reward: {reciprocal_reward.std().item()}\n")

        self.episode_log["mean_cumulative_reciprocal_reward"] = reciprocal_reward.sum(dim=0).mean().item()
        self.episode_log["std_cumulative_reciprocal_reward"] = reciprocal_reward.sum(dim=0).std().item()

        return reciprocal_reward

    def get_episode_log(self):
        log = self.episode_log.copy()
        self.episode_log = {}
        return log

    def update(self):
        """Behavior called at the end of every new episode."""
        self.update_reciprocal_reward()
        self.ppo.update(self.memory)
        self.memory.clear_memory()
        if self.grow_reciprocal_reward:
            self.reciprocal_reward_weight = min(self.reciprocal_weight_limit,
                                                self.reciprocal_reward_weight + self.reciprocal_weight_rate)

    def save(self, filename):
        self.ppo.save(filename)

    def load(self, filename):
        self.ppo.load(filename)

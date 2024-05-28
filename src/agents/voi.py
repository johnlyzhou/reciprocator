from _operator import mul
from functools import reduce
import threading
from time import sleep

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.agents.memory import Memory, TargetMemory, Transition
from src.agents.target_functions import ScalarPredictor, RecurrentScalarPredictor
from src.agents.utils import compute_returns


class ValueOfInfluence(nn.Module):
    """
    Shared components for estimating decision-theoretic influence between agents to save computational resources.
    """

    def __init__(self, num_players: int, state_dim: tuple, gamma: float, device: torch.device,
                 rnn: bool = True, n_latent_var: int = 64, lr: float = 1e-2, tau: float = 1.0,
                 target_period: int = 1, target_epochs: int = 10, target_buffer_size: int = 5,
                 target_batch_size: int = 512, num_train_batches: int = 64, parallel_trainers: int = 1):
        super(ValueOfInfluence, self).__init__()

        self.device = device
        self.num_players = num_players
        self.state_dim = state_dim
        self.gamma = gamma
        self.rnn = rnn
        self.n_latent_var = n_latent_var
        self.lr = lr
        self.tau = tau
        self.target_period = target_period
        self.target_epochs = target_epochs
        self.target_buffer_size = target_buffer_size
        self.target_batch_size = target_batch_size
        self.parallel_trainers = parallel_trainers
        self.num_train_batches = num_train_batches

        # Initialize network architecture parameters
        self.input_dims = num_players + reduce(mul, state_dim) + 1  # Actions, states, and time remaining
        self.cf_input_dims = num_players - 1 + reduce(mul, state_dim) + 1
        self.scalar_output_dims = num_players  # Either immediate rewards or Q-values

        # Initialize target networks
        self.net_type = RecurrentScalarPredictor if rnn else ScalarPredictor
        self.reward_estimator = self.net_type(self.input_dims, self.scalar_output_dims, n_latent_var).to(self.device)
        self.cf_reward_estimators = nn.ModuleList([self.net_type(self.cf_input_dims, self.scalar_output_dims,
                                                                 n_latent_var).to(self.device)
                                                   for _ in range(num_players)])
        self.q_estimator = self.net_type(self.input_dims, self.scalar_output_dims, n_latent_var).to(self.device)
        self.cf_q_estimators = nn.ModuleList([self.net_type(self.cf_input_dims, self.scalar_output_dims,
                                                            n_latent_var).to(self.device)
                                              for _ in range(num_players)])

        # If tau < 1.0, make copies of the target networks for a soft update
        if self.tau < 1.0:
            self.target_reward_estimator = self.net_type(self.input_dims, self.scalar_output_dims, n_latent_var).to(
                self.device)
            self.target_cf_reward_estimators = nn.ModuleList([self.net_type(self.cf_input_dims, self.scalar_output_dims,
                                                                            n_latent_var).to(self.device)
                                                              for _ in range(num_players)])
            self.target_q_estimator = self.net_type(self.input_dims, self.scalar_output_dims, n_latent_var).to(
                self.device)
            self.target_cf_q_estimators = nn.ModuleList([self.net_type(self.cf_input_dims, self.scalar_output_dims,
                                                                       n_latent_var).to(self.device)
                                                         for _ in range(num_players)])
            self.target_reward_estimator.load_state_dict(self.reward_estimator.state_dict())
            self.target_q_estimator.load_state_dict(self.q_estimator.state_dict())
            for i in range(num_players):
                self.target_cf_reward_estimators[i].load_state_dict(self.cf_reward_estimators[i].state_dict())
                self.target_cf_q_estimators[i].load_state_dict(self.cf_q_estimators[i].state_dict())

        # Initialize target network training optimizers and losses
        self.scalar_loss = nn.MSELoss()
        self.reward_optimizer = torch.optim.Adam(self.reward_estimator.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(self.q_estimator.parameters(), lr=lr)
        self.cf_reward_optimizers = [torch.optim.Adam(net.parameters(), lr=lr) for net in self.cf_reward_estimators]
        self.cf_q_optimizers = [torch.optim.Adam(net.parameters(), lr=lr) for net in self.cf_q_estimators]

        # Initialize memory data structures
        self.joint_memory = Memory()  # Stores joint experience (all actions and rewards from all agents)
        self.target_buffer = TargetMemory(target_buffer_size, device=torch.device('cpu'))

    def _soft_copy_weights(self, source_net, target_net, tau: float):
        source_net_dict = source_net.state_dict()
        target_net_dict = target_net.state_dict()
        for key in source_net_dict.keys():
            target_net_dict[key] = tau * source_net_dict[key] + (1 - tau) * target_net_dict[key]
        target_net.load_state_dict(target_net_dict)

    def soft_copy_weights(self, tau: float):
        self._soft_copy_weights(self.reward_estimator, self.target_reward_estimator, tau)
        self._soft_copy_weights(self.q_estimator, self.target_q_estimator, tau)
        for i in range(self.num_players):
            self._soft_copy_weights(self.cf_reward_estimators[i], self.target_cf_reward_estimators[i], tau)
            self._soft_copy_weights(self.cf_q_estimators[i], self.target_cf_q_estimators[i], tau)

    def observe(self, transition: Transition):
        """
        Observe a transition and store it in the joint memory - actions and rewards should be from all agents
        :param transition: A Transition namedtuple containing the state, action, reward, and done signal for each agent
        """
        obs, actions, rewards, _, _ = transition
        self.joint_memory.states.append(obs)
        self.joint_memory.actions.append(actions)
        self.joint_memory.rewards.append(rewards)

    def preprocess_inputs(self, states: torch.Tensor, joint_actions: torch.Tensor):
        """
        Preprocess states and actions to form input to target networks.
        :param states: Tensor of shape (T, bsz, *state_dim)
        :param joint_actions: Tensor of shape (T, bsz, num_players)
        :return: Concatenated input tensor of shape (T, bsz, input_dims)
        """
        T, bsz, _ = joint_actions.size()
        time_remaining = torch.arange(T - 1, -1, -1).repeat(bsz, 1).T.to(self.device)  # (T, bsz)

        # Concatenate joint actions, states, and timestamps to form input to target networks
        inputs = torch.cat([joint_actions.float(),
                            states.flatten(start_dim=2),
                            time_remaining.unsqueeze(-1).float()], dim=-1)  # (T, bsz, *)
        return inputs

    def store(self):
        """
        Preprocess and store a full episode of experience for training target functions. Call at the end of an episode.
        We prepend the joint_action to the input to make indexing for masking easier.
        """
        states = torch.stack(self.joint_memory.get_states(), dim=0)  # (T, bsz, *state_dim)
        joint_actions = torch.stack(self.joint_memory.get_actions(), dim=0)  # (T, bsz, num_players)
        joint_rewards = self.joint_memory.get_rewards()  # List of length T of (bsz, num_players)
        joint_returns = compute_returns(self.joint_memory.get_rewards(), self.gamma)  # (T, bsz, num_players)

        joint_rewards = torch.stack(joint_rewards, dim=0)  # (T, bsz, num_players)
        T, bsz, _ = joint_rewards.size()

        self.target_buffer.push_states(states)
        self.target_buffer.push_actions(joint_actions)
        self.target_buffer.push_rewards(joint_rewards)
        self.target_buffer.push_returns(joint_returns)

    def make_dataset(self, states, joint_actions, labels, marginalize_idx: int = None, num_samples: int = None):
        """
        Make a dataset for training a target function.
        Args:
            states: Tensor of shape (T, bsz, *state_dim)
            joint_actions: Tensor of shape (T, bsz, num_players)
            labels: Tensor of shape (T, bsz, *)
            marginalize_idx: Index of the player to marginalize out of the joint actions
            num_samples: Number of samples to take from the dataset

        Returns:
            TensorDataset of the input and labels
        """
        if marginalize_idx is not None:
            joint_actions = torch.cat([joint_actions[..., :marginalize_idx], joint_actions[..., marginalize_idx + 1:]],
                                      dim=-1)
        inputs = self.preprocess_inputs(states, joint_actions).flatten(end_dim=1)

        if num_samples is None or num_samples > inputs.size(0):
            num_samples = inputs.size(0)
        sampled_indices = torch.randperm(inputs.size(0))[:num_samples]
        return TensorDataset(inputs[sampled_indices], labels.flatten(end_dim=1)[sampled_indices])

    def fit(self, net: nn.Module, dataloader: DataLoader, optimizer, loss_fn: nn.Module):
        """
        Fit a network to the data in the dataloader.
        :param net: Network to train
        :param dataloader: Dataloader of the data to train on
        :param optimizer: Optimizer to use for training
        :param loss_fn: Loss function to use for training
        """
        for _ in tqdm(range(self.target_epochs)):
            net.train()
            for batch in dataloader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            total_loss = 0
            num_batches = 0
            net.eval()
            for batch in dataloader:
                inputs, targets = batch
                num_batches += inputs.shape[0]
                outputs = net(inputs)
                total_loss += loss_fn(outputs, targets).item()
                num_batches += 1
            print(f"Loss: {total_loss / num_batches}")

    @staticmethod
    def random_sample(inputs, labels, num_samples):
        """
        Randomly sample from the input and labels.
        Args:
            inputs: Tensor of shape (N, *)
            labels: Tensor of shape (N, *)
            num_samples: Number of samples to take

        Returns:
            Randomly sampled inputs and labels
        """
        if num_samples > inputs.size(0):
            num_samples = inputs.size(0)
        indices = torch.randperm(inputs.size(0))[:num_samples]
        return inputs[indices], labels[indices]

    def training_queue_generator(self):
        """
        Yields parameters for a training job (net, dataloader, optimizer, loss_fn, flatten) for each target function.
        """
        n_samples = self.num_train_batches * self.target_batch_size
        states = self.target_buffer.get_states(device=self.device)  # (T, buffer_size * bsz, 4, 3, 3)
        joint_actions = self.target_buffer.get_actions(device=self.device)  # (T, buffer_size * bsz, num_players)
        joint_rewards = self.target_buffer.get_rewards(device=self.device)  # (T, buffer_size * bsz, num_players)

        reward_dataloader = DataLoader(self.make_dataset(states, joint_actions, joint_rewards, num_samples=n_samples),
                                       batch_size=self.target_batch_size, shuffle=True)
        yield self.reward_estimator, reward_dataloader, self.reward_optimizer, self.scalar_loss
        del reward_dataloader

        for i in range(self.num_players):
            cf_reward_dataloader = DataLoader(
                self.make_dataset(states, joint_actions, joint_rewards, marginalize_idx=i, num_samples=n_samples),
                batch_size=self.target_batch_size, shuffle=True)
            yield self.cf_reward_estimators[i], cf_reward_dataloader, self.cf_reward_optimizers[i], self.scalar_loss
            del cf_reward_dataloader

        joint_returns = self.target_buffer.get_returns(device=self.device)  # (T, buffer_size * bsz, num_players)
        joint_q_values = joint_returns - joint_rewards  # Note this just the future component of the Q-value
        del joint_returns, joint_rewards
        q_dataloader = DataLoader(self.make_dataset(states, joint_actions, joint_q_values, num_samples=n_samples),
                                  batch_size=self.target_batch_size, shuffle=True)

        yield self.q_estimator, q_dataloader, self.q_optimizer, self.scalar_loss
        del q_dataloader

        for i in range(self.num_players):
            cf_q_dataloader = DataLoader(
                self.make_dataset(states, joint_actions, joint_q_values, marginalize_idx=i, num_samples=n_samples),
                batch_size=self.target_batch_size, shuffle=True)
            yield self.cf_q_estimators[i], cf_q_dataloader, self.cf_q_optimizers[i], self.scalar_loss
            del cf_q_dataloader

    def update(self):
        training_queue = self.training_queue_generator()

        if self.parallel_trainers == 1:
            for job_args in training_queue:
                self.fit(*job_args)
        else:
            threads = [None for _ in range(self.parallel_trainers)]

            for job_args in training_queue:
                while True:
                    try:
                        # Check for free processes
                        free_thread_idx = threads.index(None)
                        break
                    except ValueError:
                        # Check for finished processes and free them up
                        for i, p in enumerate(threads):
                            if not p.is_alive():
                                threads[i] = None
                        sleep(1)

                threads[free_thread_idx] = threading.Thread(target=self.fit, args=job_args)
                threads[free_thread_idx].start()

    def forward(self, states: torch.Tensor, joint_actions: torch.Tensor, influencer_idx: int, influenced_idx: int):
        """
        Forward pass through the target networks.
        Args:
            states: Tensor of shape (T, bsz, *state_dim)
            joint_actions: Tensor of shape (T, bsz, num_players)
            influencer_idx: Index of the player to marginalize out of the joint actions
            influenced_idx: Index of the player to estimate the value of

        Returns: VoI estimates of the marginalized player on the other player
        """
        joint_inputs = self.preprocess_inputs(states, joint_actions).flatten(end_dim=1)
        with torch.no_grad():
            if self.tau < 1.0:
                reward_hat = self.target_reward_estimator(joint_inputs)[:, influenced_idx]
                q_hat = self.target_q_estimator(joint_inputs)[:, influenced_idx]
            else:
                reward_hat = self.reward_estimator(joint_inputs)[:, influenced_idx]
                q_hat = self.q_estimator(joint_inputs)[:, influenced_idx]

        marginalized_actions = torch.cat(
            [joint_actions[..., :influencer_idx], joint_actions[..., influencer_idx + 1:]],
            dim=-1)
        marginalized_inputs = self.preprocess_inputs(states, marginalized_actions).flatten(end_dim=1)
        with torch.no_grad():
            if self.tau < 1.0:
                cf_reward_hat = self.target_cf_reward_estimators[influencer_idx](marginalized_inputs)[:, influenced_idx]
                cf_q_hat = self.target_cf_q_estimators[influencer_idx](marginalized_inputs)[:, influenced_idx]
            else:
                cf_reward_hat = self.cf_reward_estimators[influencer_idx](marginalized_inputs)[:, influenced_idx]
                cf_q_hat = self.cf_q_estimators[influencer_idx](marginalized_inputs)[:, influenced_idx]

        immediate_rew_influence = reward_hat - cf_reward_hat
        future_rew_influence = q_hat - cf_q_hat

        voi = immediate_rew_influence + self.gamma * future_rew_influence
        print(f"Max immediate reward influence: {torch.max(immediate_rew_influence).item():.4f}\n"
              f"Min immediate reward influence: {torch.min(immediate_rew_influence).item():.4f}\n"
              f"Max future reward influence: {torch.max(future_rew_influence).item():.4f}\n"
              f"Min future reward influence: {torch.min(future_rew_influence).item():.4f}\n")
        return voi

    def episode_reset(self, tau: float = None):
        self.joint_memory.clear_memory()

        if self.rnn:
            self.reward_estimator.reset()
            self.q_estimator.reset()
            for net in self.cf_reward_estimators:
                net.reset()
            for net in self.cf_q_estimators:
                net.reset()

        if self.tau == 1.0:
            pass
        elif self.tau < 1.0 and tau is None:
            self.soft_copy_weights(self.tau)
        else:
            self.soft_copy_weights(tau)

    def save(self, filename):
        if self.tau < 1.0:
            torch.save(
                {
                    "reward": self.reward_estimator.state_dict(),
                    "q": self.q_estimator.state_dict(),
                    "cf_rewards": [net.state_dict() for net in self.cf_reward_estimators],
                    "cf_q": [net.state_dict() for net in self.cf_q_estimators],
                    "target_reward": self.target_reward_estimator.state_dict(),
                    "target_q": self.target_q_estimator.state_dict(),
                    "target_cf_rewards": [net.state_dict() for net in self.target_cf_reward_estimators],
                    "target_cf_q": [net.state_dict() for net in self.target_cf_q_estimators],
                },
                filename,
            )
        else:
            torch.save(
                {
                    "reward": self.reward_estimator.state_dict(),
                    "q": self.q_estimator.state_dict(),
                    "cf_rewards": [net.state_dict() for net in self.cf_reward_estimators],
                    "cf_q": [net.state_dict() for net in self.cf_q_estimators],
                },
                filename,
            )

import time

import torch
import pygame

from src.environments.utils import same_coords
from src.environments.render_utils import draw_circle_alpha
from src.environments.vectorized_env import VectorizedEnv


class CoinGameStats:
    def __init__(self, num_players: int, device: torch.device):
        self.episode_count = 0
        self.num_players = num_players
        self.device = device
        self.batch_size = None
        self.running_rewards_per_batch = torch.zeros(num_players, device=device)
        self.own_coin_count = torch.zeros(num_players, device=device)
        self.other_coin_count = torch.zeros(num_players, device=device)
        self.total_coin_count = torch.zeros(num_players, device=device)
        self.running_info = torch.zeros((num_players, num_players), device=device)
        self.logs = []

    def update(self, rewards: torch.Tensor, info: dict):
        """
        Update the running statistics of the environment.
        :param rewards: Tensor of shape (batch_size, num_players).
        :param info: Dictionary containing the total coin collection counts across the batch, where info[i, j] is the
        number of agent j's coins collected by agent i.
        """
        if self.batch_size is None:
            self.batch_size = rewards.size(0)

        self.running_info += info

        for agent_idx in range(self.num_players):
            self.own_coin_count[agent_idx] += info[agent_idx, agent_idx].item()
            self.other_coin_count[agent_idx] += (info[agent_idx, 0:agent_idx].sum() +
                                                 info[agent_idx, agent_idx + 1:].sum())
            self.total_coin_count[agent_idx] += info[agent_idx, :].sum()
            self.running_rewards_per_batch[agent_idx] += rewards[:, agent_idx].detach().mean()

    def _log_episode_agent(self, idx):
        agent_logs = {f"coins_taken_by_{i}": self.running_info[i, idx].item() / self.batch_size for i in range(self.num_players)}
        agent_logs.update({f"reward": self.running_rewards_per_batch[idx].item(),
                           "own_coin_count": self.own_coin_count[idx].item() / self.batch_size,
                           "other_coin_count": self.other_coin_count[idx].item() / self.batch_size,
                           "total_coin_count": self.total_coin_count[idx].item() / self.batch_size, })
        return agent_logs

    def log_episode(self, extra_logs: list or dict = None, verbose: bool = False):
        log = {f"player_{idx}": self._log_episode_agent(idx) for idx in range(self.num_players)}
        if extra_logs is not None:
            if isinstance(extra_logs, list):
                for idx, extra_log in enumerate(extra_logs):
                    log[f"player_{idx}"].update(extra_log)
            elif isinstance(extra_logs, dict):
                log.update(extra_logs)
            else:
                raise ValueError("Extra logs must be a list or a dictionary.")
        self.logs.append(log)
        print(f"EPISODE {self.episode_count}:", self.logs[-1]) if verbose else None
        self._reset()
        self.episode_count += 1

    def _reset(self):
        self.running_rewards_per_batch = torch.zeros(self.num_players, device=self.device)
        self.own_coin_count = torch.zeros(self.num_players, device=self.device)
        self.other_coin_count = torch.zeros(self.num_players, device=self.device)
        self.total_coin_count = torch.zeros(self.num_players, device=self.device)
        self.running_info = torch.zeros((self.num_players, self.num_players), device=self.device)


class CoinGame(VectorizedEnv):
    """Generalized and vectorized implementation of the Coin Game environment for arbitrary numbers of agents."""
    MOVE_NAMES = ["RIGHT", "LEFT", "DOWN", "UP", "STAY"]
    MOVES = torch.stack(
        [
            torch.LongTensor([0, 1]),
            torch.LongTensor([0, -1]),
            torch.LongTensor([1, 0]),
            torch.LongTensor([-1, 0]),
            torch.LongTensor([0, 0]),
        ],
        dim=0,
    )
    COIN_REWARD = 1.
    # COIN_MISMATCH_PUNISHMENT = -2

    agent_colors = [(255, 0, 0, 128),
                    (0, 0, 255, 128),
                    (0, 255, 0, 128),
                    (255, 255, 0, 128),
                    (255, 0, 255, 128),
                    (0, 255, 255, 128)]

    def __init__(self, batch_size: int, max_steps: int, device: torch.device, num_agents: int = 2, grid_size: int = 3,

                 enable_render: bool = False):
        super().__init__(batch_size, max_steps, device, num_agents, enable_render)
        self.grid_size = grid_size
        self.COIN_MISMATCH_PUNISHMENT = - num_agents / (num_agents - 1)
        self.agent_positions = torch.zeros((self.batch_size, self.num_agents, 2), dtype=torch.long, device=self.device)
        # Only one coin per agent at any given timestep
        self.coin_positions = torch.zeros((self.batch_size, self.num_agents, 2), dtype=torch.long, device=self.device)
        self.moves = self.MOVES.to(self.device)
        self.reset()

        if enable_render:
            pygame.init()
            self.screen_size = (500, 500)
            self.grid_width = self.screen_size[0] // self.grid_size
            self.grid_height = self.screen_size[1] // self.grid_size
            self.coin_radius = self.grid_width // 4
            self.screen = pygame.display.set_mode(self.screen_size)
            self.font = pygame.font.Font('freesansbold.ttf', 24)
            if self.num_agents > len(self.agent_colors):
                raise ValueError("Too many agents for the number of colors available, add more colors!")

    def reset(self):
        """Set nonoverlapping initial positions of agents and coins."""
        self.step_count = 0

        flat_pos = torch.randint(self.grid_size ** 2, size=(self.batch_size, self.num_agents * 2)).to(self.device)
        self.agent_positions = self._get_coords(flat_pos[:, :self.num_agents]).to(self.device)
        self.coin_positions = self._get_coords(flat_pos[:, self.num_agents:]).to(self.device)
        return self._generate_observations()

    def _get_coords(self, position_val: torch.Tensor) -> torch.Tensor:
        """
        Converts a position value to a 2D grid coordinate pair.
        :param position_val: Tensor of shape (bsz, num_agents)
        :return: Tensor of shape (bsz, num_agents, 2).
        """
        return torch.stack([position_val // self.grid_size, position_val % self.grid_size], dim=-1)

    def _flatten_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Converts a 2D grid coordinate pair to a single integer.
        :param coords: Tensor of shape (bsz, num_agents, 2).
        :return: Tensor of shape (bsz, num_agents)
        """
        return (coords[..., 0] * self.grid_size + coords[..., 1]).to(self.device)

    def _generate_coins(self, collected_coins: torch.Tensor):
        """
        Spawn new coins for agents who have had their coin collected.
        :param collected_coins: Boolean Tensor of shape (bsz, num_agents). True if agent's coin was collected.
        """
        for coin_idx in range(self.num_agents):
            mask = collected_coins[:, coin_idx]
            new_coin_pos_flat = torch.randint(self.grid_size ** 2, size=(self.batch_size,)).to(self.device)[mask]
            new_coin_pos = self._get_coords(new_coin_pos_flat)
            self.coin_positions[mask, coin_idx, :] = new_coin_pos

    def _generate_observations(self):
        """Fully observable environment so all agent observations are the same (for now)."""
        # IMPORTANT! scatter_ operation doesn't work on mps, returns a tensor of all 0s
        if self.device == torch.device('mps'):
            device = 'cpu'
        else:
            device = self.device

        state = torch.zeros((self.batch_size, self.num_agents * 2, self.grid_size * self.grid_size), dtype=torch.float,
                            device=device)

        flat_agent_positions = self._flatten_coords(self.agent_positions).to(device)
        flat_coin_positions = self._flatten_coords(self.coin_positions).to(device)

        for i in range(self.num_agents):
            state[:, i].scatter_(1, flat_agent_positions[:, i:i + 1], 1)
            state[:, i + self.num_agents].scatter_(1, flat_coin_positions[:, i:i + 1], 1)

        state = state.view(self.batch_size, self.num_agents * 2, self.grid_size, self.grid_size).to(self.device)
        return [state.roll(-i * 2, dims=1) for i in range(self.num_agents)]

    def step(self, actions: torch.Tensor):
        """
        Take a step in the environment.
        :param actions: Tensor of shape (batch_size, num_agents). Each entry is an index of the action space (MOVES).
        :return: Tuple of (observations, rewards, done, coin_counts). coin_counts is a Tensor of shape
        (num_agents, num_agents). Info is a matrix containing containing the total coin collection counts across the
        batch, where info[i, j] is the number of agent j's coins collected by agent i.
        """
        collected_coins = torch.zeros((self.batch_size, self.num_agents), dtype=torch.bool, device=self.device)
        total_coin_counts = torch.zeros((self.num_agents, self.num_agents), dtype=torch.long, device=self.device)
        rewards = torch.zeros((self.batch_size, self.num_agents), dtype=torch.float, device=self.device)
        moves = torch.index_select(self.moves, 0, actions.view(-1)).view(self.batch_size, self.num_agents, 2)
        # self.agent_positions = (self.agent_positions + moves).clamp(0, self.grid_size - 1)
        self.agent_positions = (self.agent_positions + moves) % self.grid_size

        # Check coin collections and compute rewards
        for coin_idx in range(self.num_agents):
            coin_pos = self.coin_positions[:, coin_idx, :]
            collected_coins[:, coin_idx] = same_coords(
                self.agent_positions.transpose(0, 1), coin_pos).sum(dim=0) > 0
            coin_count_per_batch = same_coords(self.agent_positions.transpose(0, 1), coin_pos).T
            total_coin_counts[:, coin_idx] = coin_count_per_batch.sum(dim=0)  # Sum across batches
            rewards += coin_count_per_batch * self.COIN_REWARD

            mismatched_coin_count = (coin_count_per_batch[:, 0:coin_idx].sum(dim=1) +
                                     coin_count_per_batch[:, coin_idx + 1:self.num_agents].sum(dim=1))
            rewards[:, coin_idx] += mismatched_coin_count * self.COIN_MISMATCH_PUNISHMENT

        self._generate_coins(collected_coins)
        # observations = [self._generate_observations(agent_idx) for agent_idx in range(self.num_agents)]
        observations = self._generate_observations()
        self.step_count += 1
        done = (self.step_count >= self.max_steps) * torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        return observations, rewards, done, total_coin_counts

    def render(self, batch_idx, delay: float = 1.0):
        if not self.enable_render:
            raise RuntimeError(
                "Cannot render environment that has not been initialized with enable_render=True.")
        agent_positions = self.agent_positions[batch_idx]
        coin_positions = self.coin_positions[batch_idx]
        self._draw_grid(agent_positions, coin_positions)
        self._update_screen()
        time.sleep(delay)

    def _draw_grid(self, agent_positions, coin_positions):
        self.screen.fill((0, 0, 0))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                coords = torch.LongTensor([i, j]).to(self.device)
                coin_matches = torch.nonzero(same_coords(coin_positions, coords))
                agent_matches = torch.nonzero(same_coords(agent_positions, coords))

                grid_pos = (j * self.grid_width, i * self.grid_height)
                rect = pygame.Rect(*grid_pos, self.grid_width, self.grid_height)
                if coin_matches.size(0) > 0:
                    draw_circle_alpha(self.screen, self.agent_colors[coin_matches[0]],
                                      (grid_pos[0] + self.grid_width // 2,
                                       grid_pos[1] + self.grid_height // 2),
                                      self.coin_radius)
                else:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, width=2)

                if agent_matches.size(0) > 0:
                    pygame.draw.rect(self.screen, self.agent_colors[agent_matches[0]], rect, width=2)
                else:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, width=2)

    @staticmethod
    def _update_screen():
        pygame.display.flip()


if __name__ == '__main__':
    device = torch.device("cpu")
    num_players = 3
    grid_size = 4
    bsz = 1
    max_steps = 100

    test_env = CoinGame(bsz, max_steps, device, num_agents=num_players, grid_size=grid_size, enable_render=True)
    total_rews = 0
    for _ in range(max_steps):
        actions = torch.randint(0, test_env.MOVES.shape[0], (bsz, num_players), dtype=torch.long, device=device)
        actions[0, :1] = 4
        # print(test_env.MOVE_NAMES[actions[1, 0].item()])
        obs, rew, _, counts = test_env.step(actions)
        # total_rews += rew[1].item()
        # print("Total rewards:", total_rews)
        test_env.render(0, delay=1.0)
        print(rew[0])
        print(counts)
        pygame.event.pump()

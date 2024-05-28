import os
import json

from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from src.agents.coins_voi import CoinsVoI
from src.agents.voi import ValueOfInfluence
from src.agents.naive_learner import NaiveLearner
from src.agents.reciprocator import Reciprocator
from src.environments.coin_game import CoinGame, CoinGameStats


class CoinsTrainer:
    def __init__(self, config: dict, device: torch.device, expt_dir):
        self.expt_dir = expt_dir
        self.random_seed = config['random_seed']
        self.name = config['name']
        self.config = config
        self.env_config = config['environment']
        self.agents_config = config['agents']
        self.log_config = config['logs']
        self.device = device

        self.env = CoinGame(**self.env_config, device=self.device, enable_render=False)
        self.agent_types = self.agents_config['types']
        self.agents_config.pop('types')

        self.reciprocate = any([agent_type == 'reciprocator' for agent_type in self.agent_types])
        if self.reciprocate:
            self.num_init_eps = self.config['influence'].pop('num_initialization_episodes', 0)
            self.reciprocator_config = self.config['reciprocator']
            self.influence_kwargs = {'num_players': self.env.num_agents, 'state_dim': (self.env.num_agents * 2,
                                                                                       self.env.grid_size,
                                                                                       self.env.grid_size),
                                     'gamma': self.agents_config['gamma'], 'device': self.device}
            self.influence_kwargs.update(self.config['influence'])
            if self.env.grid_size == 3 and self.env.num_agents == 2:
                self.influence_estimator = CoinsVoI(**self.influence_kwargs)
            else:
                self.influence_estimator = ValueOfInfluence(**self.influence_kwargs)

        self.agents_kwargs = {'state_dim': (self.env.num_agents * 2, self.env.grid_size, self.env.grid_size),
                              'num_actions': self.env.MOVES.shape[0], 'device': self.device}
        self.agents_kwargs.update(self.agents_config)
        self.agents = []
        self._init_agents()

        self.logs = CoinGameStats(self.env.num_agents, self.device)
        self.episode_count = 0

        self.last_obs = self.env.reset()

    def _init_agents(self):
        for i in range(self.env.num_agents):
            if self.agent_types[i] == 'naive_learner':
                self.agents.append(NaiveLearner(**self.agents_kwargs))
            elif self.agent_types[i] == 'reciprocator':
                self.reciprocator_kwargs = {'influence_estimator': self.influence_estimator,
                                            'num_agents': self.env.num_agents, 'player_idx': i}
                self.reciprocator_kwargs.update(self.agents_kwargs)
                self.reciprocator_kwargs.update(self.reciprocator_config)
                print(self.reciprocator_kwargs)
                self.agents.append(Reciprocator(**self.reciprocator_kwargs))
            elif self.agent_types[i] == 'mfos':
                raise NotImplementedError("MFOS agent not yet implemented.")
            else:
                raise ValueError(f"Agent type {self.agents_config['type']} not supported.")

    def step(self):
        """Run a single step of the environment. Return True if the episode is done."""
        actions = torch.stack([agent.act(self.last_obs[i].float()) for i, agent in enumerate(self.agents)], dim=1)
        obs, rewards, done, info = self.env.step(actions)

        for i, agent in enumerate(self.agents):
            agent.observe((None, rewards[:, i], None, None))

        if self.reciprocate:
            self.influence_estimator.observe((self.last_obs[0], actions, rewards, done, None))

        self.last_obs = obs
        self.logs.update(rewards, info)

        return done.any()

    def episode_start(self):
        """Behavior on episode start."""
        self.episode_count += 1
        for agent in self.agents:
            agent.reset()
        self.last_obs = self.env.reset()

    def episode_end(self):
        """Behavior on episode end."""
        if self.reciprocate:
            self.influence_estimator.store()
            if self.episode_count <= self.num_init_eps or self.episode_count % self.influence_estimator.target_period == 0:
                self.influence_estimator.update()

        if self.reciprocate and self.episode_count <= self.num_init_eps:
            pass
        else:
            for agent in self.agents:
                agent.update()

        extra_logs = []
        for agent in self.agents:
            if isinstance(agent, Reciprocator):
                extra_logs.append(agent.get_episode_log())
            else:
                extra_logs.append({})
        self.logs.log_episode(extra_logs=extra_logs, verbose=True)

        if self.reciprocate:
            if self.influence_estimator.tau and self.episode_count <= self.num_init_eps:
                self.influence_estimator.episode_reset(tau=1.0)
            else:
                self.influence_estimator.episode_reset()

    def train(self, num_episodes: int):
        """Train the agents."""
        for episode in tqdm(range(1, num_episodes + 1)):
            self.episode_start()
            for _ in tqdm(range(self.env.max_steps)):
                done = self.step()
                if done.any():
                    break
            self.episode_end()

            if episode % self.log_config['checkpoint_interval'] == 0:
                self.save()

        self.save()

    def save(self):
        print(f"Saving episode {self.episode_count}!")
        try:
            for agent_idx, agent in enumerate(self.agents):
                agent.save(os.path.join(self.expt_dir, f"{self.episode_count}_{agent_idx}.pth"))
        except NotImplementedError:
            print("Agent save method not implemented. Only saving logs.")

        if self.reciprocate:
            self.influence_estimator.save(os.path.join(self.expt_dir, f"influence_{self.episode_count}.pth"))

        with open(os.path.join(self.expt_dir, f"out_{self.episode_count}.json"), "w") as f:
            json.dump(self.logs.logs, f)


if __name__ == '__main__':
    print("Testing CoinsTrainer...")
    config_path = "../../configs/coins.yaml"
    config = dict(OmegaConf.load(config_path))
    # print(config)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    trainer = CoinsTrainer(config, device, ".")
    trainer.train(10)

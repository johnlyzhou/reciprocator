import os
import json

from omegaconf import OmegaConf
import torch
from src.agents.matrix_games.matrix_tft import IPDTFT
from tqdm import tqdm

from src.agents.matrix_games.matrix_reciprocator import MatrixReciprocator
from src.environments.ipd import IteratedPrisonersDilemma, IPDStats
from src.agents.matrix_games.matrix_naive_learner import MatrixNaiveLearner


class IPDTrainer:
    def __init__(self, config: dict, device: torch.device, expt_dir):
        self.expt_dir = expt_dir
        self.random_seed = config['random_seed']
        self.name = config['name']
        self.config = config
        self.env_config = config['environment']
        self.agents_config = config['agents'].copy()
        self.log_config = config['logs']
        self.device = device

        self.env_kwargs = {'device': self.device}
        self.env_kwargs.update(self.env_config)
        self.env = IteratedPrisonersDilemma(**self.env_kwargs)

        agent_types = self.agents_config['types']
        self.agents_config.pop('types')
        self.agents_kwargs = {'device': self.device}
        self.agents_kwargs.update(self.agents_config)

        self.agents = []
        for i in range(self.env.num_agents):
            if agent_types[i] == 'reciprocator':
                self.agents.append(self._init_reciprocator(i))
            elif agent_types[i] == 'tft':
                self.agents.append(IPDTFT(self.env_config['batch_size'], device))
            elif agent_types[i] == 'naive_learner':
                self.agents.append(MatrixNaiveLearner(**self.agents_kwargs))
            else:
                raise ValueError(f"Agent type {agent_types[i]} not recognized.")
        # self.agents = [self._init_reciprocator(0), self._init_reciprocator(1)]
        # self.agents = [self._init_reciprocator(0), MatrixNaiveLearner(**self.agents_kwargs)]
        # self.agents = [self._init_reciprocator(0), IPDTFT(self.env_config['batch_size'], device)]
        # self.agents = [IPDTFT(self.env_config['batch_size'], device), MatrixNaiveLearner(**self.agents_kwargs)]
        # self.agents = [MatrixNaiveLearner(**self.agents_kwargs), MatrixNaiveLearner(**self.agents_kwargs)]

        self.logs = IPDStats(self.device)
        self.episode_count = 0

        self.last_obs = self.env.reset()

    def _init_reciprocator(self, player_idx):
        reciprocator_kwargs = {'player_idx': player_idx, 'device': self.device}
        reciprocator_kwargs.update(self.agents_config)
        reciprocator_kwargs.update(self.env_config)
        reciprocator_kwargs.update(self.config['reciprocator'])
        reciprocator_kwargs.pop('rnn')
        return MatrixReciprocator(**reciprocator_kwargs)

    def step(self):
        """Run a single step of the environment. Return True if the episode is done."""
        actions = torch.stack([agent.act(self.last_obs.float()) for agent in self.agents], dim=1)
        obs, rewards, done, info = self.env.step(actions)
        for i, agent in enumerate(self.agents):
            if isinstance(agent, MatrixReciprocator):
                agent.observe((self.last_obs, actions, rewards, None, None))
            elif isinstance(agent, IPDTFT):
                agent.observe(actions[:, 1 - i])
            else:
                agent.observe((None, rewards[:, i], None, None))

        self.last_obs = obs
        self.logs.update(actions)

        return done

    def episode_start(self):
        """Behavior on episode start."""
        self.episode_count += 1
        for agent in self.agents:
            agent.reset()
        self.last_obs = self.env.reset()

    def episode_end(self):
        """Behavior on episode end."""
        extra_logs = []
        for agent in self.agents:
            if isinstance(agent, MatrixReciprocator):
                extra_logs.append(agent.get_episode_log())
            else:
                extra_logs.append({})
        self.logs.log_episode(extra_logs=extra_logs, verbose=True)

        for agent in self.agents:
            agent.update()

    def train(self, num_episodes: int):
        """Train the agents."""
        for episode in tqdm(range(1, num_episodes + 1)):
            self.episode_start()
            for _ in tqdm(range(self.env_kwargs['max_steps'])):
                done = self.step()
                if done:
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

        with open(os.path.join(self.expt_dir, f"out_{self.episode_count}.json"), "w") as f:
            json.dump(self.logs.logs, f)


if __name__ == '__main__':
    print("Testing IPDTrainer...")
    config_path = "../../configs/ipd.yaml"
    config = dict(OmegaConf.load(config_path))
    # print(config)
    device = torch.device("cpu")
    trainer = IPDTrainer(config, device, "ipd_test")
    trainer.train(1000)

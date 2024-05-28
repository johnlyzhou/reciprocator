import torch

from omegaconf import OmegaConf

from src.agents.naive_learner import NaiveLearner
from src.environments.coin_game import CoinGame, CoinGameStats
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--expt-path", type=str, required=True)
parser.add_argument("-d", "--device", type=str, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    expt_path = args.expt_path
    # Set device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    with open(f"{args.expt_path}/config.yaml", "r") as f:
        config = dict(OmegaConf.load(f))
    env_config = config["environment"]
    env_config['batch_size'] = 1
    env = CoinGame(**env_config, device=device, enable_render=True)
    agents_config = config["agents"]
    agents_config.pop("types")
    agents_kwargs = {'state_dim': (env.num_agents * 2, env.grid_size, env.grid_size),
                     'num_actions': env.MOVES.shape[0], 'device': device}
    agents_kwargs.update(agents_config)

    # Load experiment
    agent_0 = NaiveLearner(**agents_kwargs)
    agent_0.ppo.load(f"{expt_path}/275_0.pth")
    agent_1 = NaiveLearner(**agents_kwargs)
    agent_1.ppo.load(f"{expt_path}/275_1.pth")
    agents = [agent_0, agent_1]

    # Set up logging
    logs = CoinGameStats(env.num_agents, device)

    # Training loop
    last_obs = env.reset()
    for t in range(env.max_steps):
        # Running policy_old:
        with torch.no_grad():
            actions = torch.Tensor([agent.act(last_obs[0].float()).squeeze() for i, agent in enumerate(agents)]).int()
        last_obs, rewards, done, info = env.step(actions)
        logs.update(rewards, info)
        print("STEP:", t)
        for agent_idx in range(env.num_agents):
            if info[agent_idx, agent_idx]:
                print(f"    Agent {agent_idx} collected 1 of its own coins!")
            if info[agent_idx, 1 - agent_idx]:
                print(f"    Agent {agent_idx} collected 1 of the other agent's coins!")

        env.render(0, delay=1.0)

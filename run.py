import argparse
import json
import os
import pathlib
from time import sleep
import multiprocessing as mp

from omegaconf import OmegaConf
import torch

from src.training.coins_trainer import CoinsTrainer
from src.training.cleanup_trainer import CleanUpTrainer
from src.training.ipd_trainer import IPDTrainer

CUDA_LIST = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]


def run_trainer(trainer_class, config: dict, device: torch.device, max_episodes: int, results_dir: str = None):
    """Train function for use with multiprocessing."""
    trainer = trainer_class(config, device, results_dir)
    trainer.train(max_episodes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="test")
    parser.add_argument("-g", "--game", type=str, required=True)
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-e", "--episodes", type=int, required=True)
    parser.add_argument("-d", "--device", type=str, default=None)
    parser.add_argument("-r", "--replicate", type=int, default=1)
    parser.add_argument("-dd", "--results_dir", type=str, default="results")
    args = parser.parse_args()

    game_name = args.game.lower()
    if game_name == "cleanup":
        trainer_class = CleanUpTrainer
    elif game_name == "coins":
        trainer_class = CoinsTrainer
    elif game_name == "ipd":
        trainer_class = IPDTrainer
    else:
        raise ValueError(f"Game {game_name} not recognized.")

    if args.device == "all":
        device_list = CUDA_LIST
        print("Using all available devices:", device_list)
    elif args.device is not None:
        device_list = [torch.device(args.device)]
        print("Using:", device_list[0])
    else:
        device_list = [torch.device("cpu")]
        print("No device specified, using CPU")

    config = OmegaConf.load(args.config)

    base_name = args.name
    config.name = base_name
    max_episodes = args.episodes
    results_dir = os.path.join(args.results_dir, base_name)

    processes = [None for _ in range(len(device_list))]

    for i in range(args.replicate):
        if args.replicate > 1:
            config.name = f"{base_name}_{i}"
        expt_dir = os.path.join(results_dir, config.name)
        if not os.path.isdir(expt_dir):
            pathlib.Path(expt_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(expt_dir, "commandline_args.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=2)
            with open(os.path.join(expt_dir, "config.yaml"), "w") as f:
                OmegaConf.save(config, f)

        while True:
            try:
                device_idx = processes.index(None)
                device = device_list[device_idx]
                break
            except ValueError:
                for i, p in enumerate(processes):
                    if not p.is_alive():
                        processes[i] = None
                sleep(1)

        process_args = (trainer_class, config, device, max_episodes, expt_dir)

        processes[device_idx] = mp.Process(target=run_trainer, args=process_args)
        processes[device_idx].start()

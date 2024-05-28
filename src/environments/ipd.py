import torch

REWARD_MATRIX = torch.tensor([[[-1.0, -1.0], [-3.0, -0.0]], [[-0.0, -3.0], [-2.0, -2.0]]])
# REWARD_MATRIX = torch.tensor([[[0.0, 0.0], [-1.0, 1.0]], [[1.0, -1.0], [-100.0, -100.0]]])


class IPDStats:
    def __init__(self, device: torch.device):
        self.device = device
        self.reward_matrix = REWARD_MATRIX.to(device)
        self.num_players = 2
        self.episode_count = 0
        self.step_count = 0
        self.rewards_per_batch = torch.zeros(self.num_players, device=device)
        self.p_cooperate_per_batch = [[] for _ in range(self.num_players)]  # Mean p(cooperate) per batch
        self.logs = []

    def update(self, actions: torch.Tensor):
        """
        Update the action history.
        :param actions: A batch of binary actions of shape (bsz, 2) taken by the agents in the game
        (0: cooperate, 1: defect)
        """
        for i in range(self.num_players):
            self.p_cooperate_per_batch[i].append(1 - actions[:, i].mean().item())
        rewards = self.reward_matrix[actions[:, 0].type(torch.int32), actions[:, 1].type(torch.int32), :]
        self.rewards_per_batch += rewards.mean(dim=0)
        self.step_count += 1

    def _log_episode_agent(self, idx):
        return {"reward": self.rewards_per_batch[idx].item() / self.step_count,
                "p(cooperate)": sum(self.p_cooperate_per_batch[idx]) / len(self.p_cooperate_per_batch[idx])}

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
        self.rewards_per_batch = torch.zeros(self.num_players, device=self.device)
        self.p_cooperate_per_batch = [[] for _ in range(self.num_players)]
        self.step_count = 0


class IteratedPrisonersDilemma:
    MOVE_NAMES = ['Cooperate (Stay Silent)', 'Defect (Confess)']  # For clarity
    MOVES = torch.tensor([0, 1])
    num_agents = 2

    def __init__(self, batch_size: int, max_steps: int, device: torch.device):
        self.bsz = batch_size
        self.max_steps = max_steps
        self.device = device
        self.reward_matrix = REWARD_MATRIX.to(device)
        self.step_count = 0

    def step(self, actions: torch.Tensor):
        """
        Takes a step in the environment.
        :param actions: A tensor of shape (batch_size, num_players) containing the actions of each player.
        :return: A tensor of shape (batch_size, num_players).
        """
        self.step_count += 1
        rewards = self.reward_matrix[actions[:, 0].type(torch.int32), actions[:, 1].type(torch.int32), :]
        return actions, rewards, self.step_count >= self.max_steps, None

    def reset(self):
        self.step_count = 0
        return torch.tensor([[-1, 0]], device=self.device).repeat(self.bsz, 1)


if __name__ == '__main__':
    device = torch.device('cpu')
    ipd = IteratedPrisonersDilemma(512, 32, device)
    rews = ipd.step(torch.tensor([[0, 1], [1, 0], [1, 1], [0, 0]]).to(device))
    # print(rews)
    print(REWARD_MATRIX[0, 1, 0])
    print(REWARD_MATRIX[:, :, 1])
    print(REWARD_MATRIX[:, 0, 0])

    # print(rews)

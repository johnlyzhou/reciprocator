from abc import ABC, abstractmethod
import torch


class VectorizedEnv(ABC):
    def __init__(self, batch_size: int, max_steps: int, device: torch.device, num_agents: int,
                 enable_render: bool = False, **kwargs):
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.device = device
        self.num_agents = num_agents
        self.enable_render = enable_render
        self.step_count = 0

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions: torch.Tensor):
        pass

    @abstractmethod
    def render(self, batch_idx, delay: float = 1.0):
        pass

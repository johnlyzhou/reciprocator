from abc import ABC, abstractmethod

import torch


class Agent(ABC):
    def __init__(self):
        pass

    def reset(self):
        pass

    @abstractmethod
    def act(self, state: torch.Tensor):
        pass

    @abstractmethod
    def observe(self, transition: tuple):
        pass

    @abstractmethod
    def update(self):
        pass

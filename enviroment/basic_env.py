from abc import ABC, abstractmethod

class BasicEnviroment(ABC):
    def __init__(self, type, name):
        self.type = type
        self.name = name

    @abstractmethod
    def get_env_params(self, space):
        ...
    @abstractmethod
    def parse_env_space(self, obs_space, action_space):
        ...
    @abstractmethod
    def random_action(self):
        pass
    @abstractmethod
    def step(self, action):
        pass
    @abstractmethod
    def reset(self):
        pass
    @abstractmethod
    def close(self):
        pass

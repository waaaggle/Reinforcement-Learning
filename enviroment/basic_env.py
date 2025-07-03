from abc import ABC, abstractmethod

class BasicEnviroment(ABC):
    EnviromentList = {
        #离散动作空间：Acrobot-v1，CartPole-v1，LunarLander-v3，MountainCar-v0
        #连续动作空间：MountainCarContinuous-v0，BipedalWalker-v3，BipedalWalkerHardcore-v3
        'GYM': ('Acrobot-v1', 'CartPole-v1', 'LunarLander-v3', 'MountainCar-v0', 'MountainCarContinuous-v0',
                'BipedalWalker-v3', 'BipedalWalkerHardcore-v3',),
        'NETWORK': ('RouteV1',),
        'VIRTUAL': ('discrete_action_v1','not_discrete_action_v1')
    }
    def __init__(self, type, name):
        self.type = type
        self.name = name
        self.check_env()
    def check_env(self):
        if self.type not in BasicEnviroment.EnviromentList:
            raise Exception('invalid env type: {}'.format(self.type))
        if self.name not in BasicEnviroment.EnviromentList[self.type]:
            raise Exception('invalid env name: {}'.format(self.name))

    @abstractmethod
    def parse_env_space(self, space):
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

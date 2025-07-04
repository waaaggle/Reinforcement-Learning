from enviroment.virtual_env import start_virtual_env
from enviroment.gym_env import start_gym_env
from models.DDQN import ddqn_get_trained_model
from models.DDPG import ddpg_get_trained_model
from utils.logger import my_logger


class RegistryInfo(object):
    env_list = {
        'Acrobot-v1': {'origin':'GYM','action_type':'discrete'},
        'CartPole-v1': {'origin': 'GYM', 'action_type':'discrete'},
        'LunarLander-v3': {'origin': 'GYM', 'action_type':'discrete'},
        'MountainCar-v0': {'origin': 'GYM', 'action_type':'discrete'},
        'MountainCarContinuous-v0': {'origin': 'GYM', 'action_type':'continous'},
        'BipedalWalker-v3': {'origin': 'GYM', 'action_type':'continous'},
        'BipedalWalkerHardcore-v3': {'origin': 'GYM', 'action_type':'continous'},
        'Route-v1': {'origin': 'NETWORK', 'action_type':'discrete'},
        'Route-v2': {'origin': 'NETWORK', 'action_type':'continous'},
        'VIRTUAL-v1': {'origin': 'VIRTUAL', 'action_type':'discrete'},
        'VIRTUAL-v2': {'origin': 'VIRTUAL', 'action_type':'continous'},
    }
    model_list = {
        'DDQN':{'action_type':['discrete'], 'start_func':ddqn_get_trained_model},
        'DDPG':{'action_type':['continous'], 'start_func':ddpg_get_trained_model},
        # 'PPO':{'action_type':[], 'start_func':None}
    }
    env_start_registry = {
        'GYM': start_gym_env,
        'VIRTUAL': start_virtual_env,
        # 'NETWORK':start_network_env
    }

    def __init__(self):
        pass

    @classmethod
    def check_model_support_env(cls, model_name, env_name):
        if env_name not in cls.env_list or model_name not in cls.model_list:
            raise Exception('No supported, env name: {}, model name: {}'.format(env_name, model_name))
        action_type = cls.env_list[env_name]['action_type']
        if action_type in cls.model_list[model_name]['action_type']:
            return True
        my_logger.error('No supported, env name: {}, model name: {}'.format(env_name, model_name))
        return  False

    @classmethod
    def get_train_env(cls, env_name):
        env_origin = cls.env_list[env_name]['origin']
        if env_origin not in cls.env_start_registry:
            raise Exception('No find start registry env name: {}'.format(env_name))
        start_func = cls.env_start_registry[env_origin]
        return start_func(env_name)

    @classmethod
    def get_train_model(cls, model_name, env_inst):
        env_params = env_inst.get_env_params()
        start_func = cls.model_list[model_name]['start_func']
        return  start_func(env_params)
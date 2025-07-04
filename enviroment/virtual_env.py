import random
import numpy as np
from enviroment.basic_env import BasicEnviroment
from utils.logger import my_logger

class VirtualEnviroment(BasicEnviroment):
    def __init__(self, type, name, state_dim, state_count, action_dim, action_count):
        super().__init__(type, name)
        self.type = type
        self.name = name
        self.state_dim = state_dim
        self.state_count = state_count
        self.action_dim = action_dim
        self.action_count = action_count
        self.obs_space_info, self.action_space_info = self.parse_env_space((name, state_dim, state_count), (name, action_dim, action_count))
        self.reset()

    def parse_env_space(self, obs_space, action_space):
        obs_space_info = {}
        action_space_info = {}
        #解析state空间
        if obs_space[0] == 'VIRTUAL-v1':
            obs_space_info['discrete_space'] = True
            obs_space_info['discrete_count'] = obs_space[2]
        else:
            obs_space_info['discrete_space'] = False
            obs_space_info['nodiscrete_shape'] = (obs_space[1], )
            obs_space_info['nodiscrete_low'] = 1.0
            obs_space_info['nodiscrete_high'] = 10.0

        #解析action空间
        if action_space[0] == 'VIRTUAL-v1':
            action_space_info['discrete_space'] = True
            action_space_info['discrete_count'] = action_space[2]
        else:
            action_space_info['discrete_space'] = False
            action_space_info['nodiscrete_shape'] = (action_space[1], )
            action_space_info['nodiscrete_low'] = 1.0
            action_space_info['nodiscrete_high'] = 10.0
        return (obs_space_info, action_space_info)

    def get_env_params(self):
        if self.obs_space_info['discrete_space']:
            state_count = self.obs_space_info['discrete_count']
            state_dim = 1
        else:
            state_dim = self.obs_space_info['nodiscrete_shape'][0]
            state_count = None

        if self.action_space_info['discrete_space']:  # 连续空间维度为1，离散才是非1
            action_count = self.action_space_info['discrete_count']
            action_dim = 1
        else:
            action_dim = self.action_space_info['nodiscrete_shape'][0]
            action_count = None
        my_logger.info(
            'state dim: {}, state count: {}, action dim:{}, action count:{}'.format(state_dim, state_count, action_dim,
                                                                                    action_count))
        return state_dim, state_count, action_dim, action_count

    def random_action(self):
        if self.action_space_info['discrete_space']:     #离散动作，得到action编号，action_dim代表action个数，不是action tensor维度
            action = random.randint(0, self.action_space_info['discrete_count'] -1)
            action = np.int64(action)
        else:     #连续动作，得到action值
            max_value = self.action_space_info['nodiscrete_high']
            min_value = self.action_space_info['nodiscrete_low']
            action = [random.uniform(min_value, max_value) for _ in range(self.action_space_info['nodiscrete_shape'][0])]
            action = np.array(action, dtype=np.float32)
        return action

    def random_obs(self):
        if self.obs_space_info['discrete_space']:     #离散obs
            #离散obs输入一维张量，长度也为1
            obs = [random.randint(0, self.obs_space_info['discrete_count'] -1)]
            obs = np.array(obs, dtype=np.float32)
        else:     #连续动作，得到action值
            max_value = self.obs_space_info['nodiscrete_high']
            min_value = self.obs_space_info['nodiscrete_low']
            obs = [random.uniform(min_value*10, max_value*10) for _ in range(self.obs_space_info['nodiscrete_shape'][0])]
            obs = np.array(obs, dtype=np.float32)
        return obs

    def step(self, action):
        self.counts += 1
        next_obs = self.random_obs()
        info = {}
        info['step'] = self.counts
        if self.counts %50 ==0:
            reward = 100.0
            done = True
        else:
            reward = 0.0
            done = False
        self.reward = reward
        self.done = done
        self.info = info
        self.obs = next_obs
        return next_obs, self.reward, self.done, self.info

    def reset(self):
        self.obs = self.random_obs()
        self.counts = 1
        self.info = '{}th steps'.format(self.counts)
        self.Done = False

    def close(self):
        self.reset()

def start_virtual_env(name):
    state_dim = 10
    state_count = 50
    action_dim = 5
    action_count = 20
    env_inst = VirtualEnviroment('VIRTUAL', name, state_dim, state_count, action_dim, action_count)
    return env_inst

if __name__ == '__main__':
    env_inst = start_virtual_env('VIRTUAL-v1')  #state dim :1(count:50), action dim:1(count:20)
    # env_inst = start_virtual_env('not_discrete_action_v1')  #state dim :10, action dim:5
    done = False
    action = env_inst.random_action()
    print('action type:', type(action), 'action:', action)  # <class 'numpy.int64'> or  <class 'numpy.ndarray'>
    # print('obs type:', type(env_inst.obs), 'obs:', env_inst.obs)  # <class 'numpy.ndarray'>
    next_obs, reward, done, info = env_inst.step(action)
    print('next_obs type:', type(next_obs), 'next_obs:', next_obs)
    print('reward type:', type(reward), 'reward:', reward)  #<class 'float'>  or <class 'numpy.float64'>
    print('done type:', type(done), 'done:', done)         #<class 'bool'>
    print('info type:', type(info), 'info:', info)     #<class 'dict'>
    while not done:
        action = env_inst.random_action()  # 实际训练时用agent输出
        next_obs, reward, done, info = env_inst.step(action)
    env_inst.close()

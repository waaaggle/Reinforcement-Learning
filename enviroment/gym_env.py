from enviroment.basic_env import BasicEnviroment
from utils.logger import my_logger
import gymnasium as gym

class GymEnviroment(BasicEnviroment):
    def __init__(self, type, name, render_mode=None):
        super().__init__(type, name)
        self.type = type
        self.name = name
        if render_mode:
            self.env = gym.make(self.name, render_mode=render_mode)
        else:
            self.env = gym.make(self.name)
        self.obs_space_info, self.action_space_info = self.parse_env_space(self.env.observation_space, self.env.action_space)
        self.reset()

    def parse_env_space(self, obs_space, action_space):
        obs_space_info = {}
        action_space_info = {}
        if isinstance(obs_space, gym.spaces.Discrete):  #空间是离散的
            obs_space_info['discrete_space'] = True
            obs_space_info['discrete_count'] = obs_space.n
        elif isinstance(obs_space, gym.spaces.Box):  #空间是连续的
            obs_space_info['discrete_space'] = False
            obs_space_info['nodiscrete_shape'] = obs_space.shape
            obs_space_info['nodiscrete_low'] = obs_space.low
            obs_space_info['nodiscrete_high'] = obs_space.high
        else:
            raise Exception('not support state space: {}'.format(type(obs_space)))

        if isinstance(action_space, gym.spaces.Discrete):  #空间是离散的
            action_space_info['discrete_space'] = True
            action_space_info['discrete_count'] = action_space.n
        elif isinstance(action_space, gym.spaces.Box):  #空间是连续的
            action_space_info['discrete_space'] = False
            action_space_info['nodiscrete_shape'] = action_space.shape
            action_space_info['nodiscrete_low'] = action_space.low
            action_space_info['nodiscrete_high'] = action_space.high
        else:
            raise Exception('not support action space: {}'.format(type(action_space)))
        return obs_space_info, action_space_info

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
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        next_obs, self.reward, terminated, truncated, self.info = self.env.step(action)
        self.done = terminated or truncated
        self.obs = next_obs
        return next_obs, self.reward, self.done, self.info

    def reset(self):
        self.obs, self.info = self.env.reset()
        self.done = False

    def close(self):
        self.reset()
        self.env.close()

def start_gym_env(name):
    env_inst = GymEnviroment('GYM', name, render_mode='human')
    return env_inst

if __name__ == '__main__':
    env_inst = start_gym_env('CartPole-v1')  #state dim :4, action dim:1（count2）
    # env_inst = start_gym_env('BipedalWalker-v3')  #state dim :24, action dim:4
    done = False
    action = env_inst.random_action()
    # print('obs type:', type(obs), 'obs:', obs)  # <class 'numpy.ndarray'>
    print('action type:', type(action), 'action:', action)  # <class 'numpy.int64'> or  <class 'numpy.ndarray'>
    next_obs, reward, done, info = env_inst.step(action)
    print('next_obs type:', type(next_obs), 'next_obs:', next_obs)
    print('reward type:', type(reward), 'reward:', reward)  #<class 'float'>  or <class 'numpy.float64'>
    print('done type:', type(done), 'done:', done)         #<class 'bool'>
    print('info type:', type(info), 'info:', info)     #<class 'dict'>
    while not done:
        action = env_inst.random_action()  # 实际训练时用agent输出
        next_obs, reward, done, info = env_inst.step(action)
    env_inst.close()
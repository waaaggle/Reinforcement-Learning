from enviroment.basic_env import BasicEnviroment
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
        self.obs_space_info = self.parse_env_space(self.env.observation_space)
        self.action_space_info = self.parse_env_space(self.env.action_space)
        self.reset()

    def parse_env_space(self, space):
        space_info = {}
        if isinstance(space, gym.spaces.Discrete):  #空间是离散的
            space_info['discrete_space'] = True
            space_info['discrete_count'] = space.n
        elif isinstance(space, gym.spaces.Box):  #空间是连续的
            space_info['discrete_space'] = False
            space_info['nodiscrete_shape'] = space.shape
            space_info['nodiscrete_low'] = space.low
            space_info['nodiscrete_high'] = space.high
        else:
            raise Exception('not support action space: {}'.format(type(space)))
        return space_info

    def random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        self.obs, self.reward, terminated, truncated, self.info = self.env.step(action)
        self.done = terminated or truncated
        return self.obs, self.reward, self.done, self.info

    def reset(self):
        self.obs, self.info = self.env.reset()
        self.done = False

    def close(self):
        self.reset()
        self.env.close()

def start_gym_env(name):
    env_inst = GymEnviroment('GYM', name, render_mode='human')
    state_dim = None
    state_count = None
    action_dim = None
    action_count = None
    if env_inst.obs_space_info['discrete_space']:
        state_count = env_inst.obs_space_info['discrete_count']
        state_dim = 1
    else:
        state_dim = env_inst.obs_space_info['nodiscrete_shape'][0]
        state_count = None

    if env_inst.action_space_info['discrete_space']:  #连续空间维度为1，离散才是非1
        action_count = env_inst.action_space_info['discrete_count']
        action_dim = 1
    else:
        action_dim = env_inst.action_space_info['nodiscrete_shape'][0]
        action_count = None
    print('state dim: {}, state count: {}, action dim:{}, action count:{}'.format(state_dim, state_count, action_dim, action_count))
    return env_inst, state_dim, state_count, action_dim, action_count

if __name__ == '__main__':
    env_inst, state_dim, state_count, action_dim, action_count = start_gym_env('CartPole-v1')  #state dim :4, action dim:1（count2）
    # env_inst, state_dim, state_count, action_dim, action_count = start_gym_env('BipedalWalker-v3')  #state dim :24, action dim:4
    done = False
    action = env_inst.random_action()
    obs, reward, next_obs, done, info = env_inst.step(action)
    print('obs type:', type(obs), 'obs:', obs)  #<class 'numpy.ndarray'>
    print('action type:', type(action), 'action:', action)  #<class 'numpy.int64'> or  <class 'numpy.ndarray'>
    print('next_obs type:', type(next_obs), 'next_obs:', next_obs)
    print('reward type:', type(reward), 'reward:', reward)  #<class 'float'>  or <class 'numpy.float64'>
    print('done type:', type(done), 'done:', done)         #<class 'bool'>
    print('info type:', type(info), 'info:', info)     #<class 'dict'>
    while not done:
        action = env_inst.random_action()  # 实际训练时用agent输出
        obs, reward, next_obs, done, info = env_inst.step(action)
    env_inst.close()
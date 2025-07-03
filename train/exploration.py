import torch
import numpy as np
from samples.replay_buffer import ReplayMemory
from train.logger import my_logger

#使用agent在环境中探索count次
def expolre(agent, env_inst, count):
    samples_pool = ReplayMemory(count)
    done = False
    while count >0:
        count -= 1
        state_tensor = torch.tensor(env_inst.obs, dtype=torch.float32)
        # print('state_tensor', state_tensor)
        action_tensor = agent.take_action(state_tensor)  # 实际训练时用agent输出
        my_logger.debug('take action, obs:{}, action:{}'.format(state_tensor, action_tensor))
        # print('action_tensor', action_tensor, action_tensor.dim())
        #action需要的类型为#<class 'numpy.int64'> or  <class 'numpy.ndarray'>
        if 0 == action_tensor.dim():  #类型为#<class 'numpy.int64'>
            action = np.int64(action_tensor.item())
        elif 1 == action_tensor.dim():  #类型为<class 'numpy.ndarray'>
            action = action_tensor.detach().numpy()
        else:
            raise Exception('invalid action tensor:{}, dim:{}'.format(action_tensor, action_tensor.dim()))
        obs, reward, next_obs, done, info = env_inst.step(action)
        my_logger.debug('sample info, obs:{}, reward:{}, next obs:{}, done:{}, info:{}'.format(obs, reward, next_obs, done, info))
        samples_pool.push((obs, action, reward, next_obs, done, info))
        if done:
            env_inst.reset()

    return samples_pool

if __name__ == '__main__':
    te = torch.tensor([1,2,3], dtype=torch.float)
    print(te.shape, type(te.shape))   #torch.Size([3])
    print(te.size())  #torch.Size([3])
    print(te.dim())   #1
    te = torch.tensor(112, dtype=torch.float)
    print(te.shape)   #torch.Size([])
    print(te.size())  #torch.Size([])
    print(te.dim())   #0

    n1 = np.array([1,2,3])
    n1 = torch.tensor(n1)
    print(n1.shape, n1.dim())
    n1 = np.int64(123)
    n1 = torch.tensor(n1)
    print(n1.shape, n1.dim())
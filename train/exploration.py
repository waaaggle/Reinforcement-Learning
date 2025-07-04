import torch
import numpy as np
from utils.logger import my_logger

#使用agent在环境中探索num_episodes次,没探索一次会产生多条样本放入样本池，每次从样本池随机选一个batch训练
def on_policy_expolre(samples_pool, agent, env_inst, num_episodes, batch_size):
    episodes_count = 0
    while episodes_count < num_episodes:
        done = False
        obs = env_inst.obs
        samples_pool.clear()
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32)
            action_tensor = agent.take_action(state_tensor)  # 实际训练时用agent输出
            my_logger.debug('take action, obs:{}, action:{}'.format(state_tensor, action_tensor))
            #action需要的类型为#<class 'numpy.int64'> or  <class 'numpy.ndarray'>
            if 0 == action_tensor.dim():  #类型为#<class 'numpy.int64'>
                # action = np.int64(action_tensor.item())
                action = action_tensor.detach().numpy()
            elif 1 == action_tensor.dim():  #类型为<class 'numpy.ndarray'>
                action = action_tensor.detach().numpy()
            else:
                raise Exception('invalid action tensor:{}, dim:{}'.format(action_tensor, action_tensor.dim()))
            next_obs, reward, done, info = env_inst.step(action)
            my_logger.debug('sample info, obs:{}, reward:{}, next obs:{}, done:{}, info:{}'.format(obs, reward, next_obs, done, info))
            samples_pool.push((obs, action, reward, next_obs, done, info))
            obs = next_obs
        episodes_count += 1
        env_inst.reset()
        if samples_pool.size() >= batch_size:
            batch_samples = samples_pool.samples_k(batch_size)
        else:
            batch_samples = samples_pool.samples_k(samples_pool.size())
        agent.train(batch_samples)

#使用agent在环境中探索num_episodes次,没探索一次会产生多条样本放入样本池，每次从样本池随机选一个batch训练
def off_policy_expolre(samples_pool, agent, env_inst, num_episodes, num_episodes_per_train, batch_size):
    episodes_count = 0
    while episodes_count < num_episodes:
        done = False
        obs = env_inst.obs
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32)
            action_tensor = agent.take_action(state_tensor)  # 实际训练时用agent输出
            my_logger.debug('take action, obs:{}, action:{}'.format(state_tensor, action_tensor))
            #action需要的类型为#<class 'numpy.int64'> or  <class 'numpy.ndarray'>
            if 0 == action_tensor.dim():  #类型为#<class 'numpy.int64'>
                # action = np.int64(action_tensor.item())
                # action = action_tensor.detach().numpy()
                action = action_tensor.item()
            elif 1 == action_tensor.dim():  #类型为<class 'numpy.ndarray'>
                action = action_tensor.detach().numpy()
            else:
                raise Exception('invalid action tensor:{}, dim:{}'.format(action_tensor, action_tensor.dim()))
            next_obs, reward, done, info = env_inst.step(action)
            my_logger.debug('sample info, obs:{}, reward:{}, next obs:{}, done:{}, info:{}'.format(obs, reward, next_obs, done, info))
            samples_pool.push((obs, action, reward, next_obs, done, info))
            obs = next_obs
        episodes_count += 1
        env_inst.reset()
        if episodes_count %num_episodes_per_train ==0 and samples_pool.size() >= batch_size:
            batch_samples = samples_pool.samples_k(batch_size)
            agent.train(batch_samples)

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
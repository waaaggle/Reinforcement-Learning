import torch
import numpy as np
from utils.logger import my_logger
from tqdm import tqdm

def explore_one_step(agent, env_inst, obs, is_evaluate):
    state_tensor = torch.tensor(obs, dtype=torch.float32)
    action_tensor = agent.take_action(state_tensor, is_evaluate)  # 实际训练时用agent输出
    my_logger.debug('take action, obs:{}, action:{}'.format(state_tensor, action_tensor))
    if 0 == action_tensor.dim():  # 类型为#<class 'numpy.int64'>
        action = action_tensor.item()  # action为标量值
    elif 1 == action_tensor.dim():  # 类型为<class 'numpy.ndarray'>
        action = action_tensor.detach().numpy()  # action为np数组
    else:
        raise Exception('invalid action tensor:{}, dim:{}'.format(action_tensor, action_tensor.dim()))

    next_obs, reward, done, info = env_inst.step(action)
    my_logger.debug(
        'sample info, obs:{}, reward:{}, next obs:{}, done:{}, info:{}'.format(obs, reward, next_obs, done, info))

    return action, next_obs, reward, done, info

#使用agent在环境中探索num_episodes次,没探索一次会产生多条样本放入样本池，每次从样本池随机选一个batch训练
def on_policy_expolre(samples_pool, agent, env_inst, num_episodes):
    episodes_count = 0
    with tqdm(total=num_episodes, desc='训练进度') as pbar:
        while episodes_count < num_episodes:
            done = False
            obs = env_inst.reset()
            samples_pool.clear()
            episode_reward = 0
            while not done:
                action, next_obs, reward, done, info = explore_one_step(agent, env_inst, obs, False)
                episode_reward += reward
                samples_pool.push((obs, action, reward, next_obs, done, info))
                obs = next_obs
            agent.update_episode_rewards(episode_reward)
            #按顺序从头到尾打包sample
            batch_samples = samples_pool.samples_all_in_order()
            agent.train(batch_samples)
            episodes_count += 1
            #进度条前进1，自动根据总episode映射
            pbar.set_postfix({'episode': '%d' % (episodes_count), 'return': '%.3f' % np.mean(agent.episode_rewards[-10:])})
            pbar.update(1)
    print('trained total steps is:{}'.format(agent.total_steps))

#使用agent在环境中探索num_episodes次,没探索一次会产生多条样本放入样本池，每次从样本池随机选一个batch训练
def off_policy_expolre(samples_pool, agent, env_inst, num_episodes, warmup_steps, num_episodes_per_train, batch_size):
    episodes_count = 0
    with tqdm(total=num_episodes, desc='训练进度') as pbar:
        #正式采集num_episodes轮
        while episodes_count < num_episodes:
            done = False
            obs = env_inst.reset()
            episode_reward = 0
            while not done:
                action, next_obs, reward, done, info = explore_one_step(agent, env_inst, obs, False)
                episode_reward += reward
                samples_pool.push((obs, action, reward, next_obs, done, info))
                obs = next_obs
            # 每多少个episode训练一次，且要buffer个数大于warmup数量
            # and samples_pool.size() > warmup_steps        预热暂未生效
            if episodes_count % num_episodes_per_train == 0  and samples_pool.size() >= batch_size:
                batch_samples = samples_pool.samples_k(batch_size)
                agent.train(batch_samples)
            agent.update_episode_rewards(episode_reward)
            episodes_count += 1
            #进度条前进1，自动根据总episode映射
            pbar.set_postfix({'episode': '%d' % (episodes_count), 'return': '%.3f' % np.mean(agent.episode_rewards[-10:])})
            pbar.update(1)
    print('trained total steps is:{}'.format(agent.total_steps))

#使用agent在环境中探索num_episodes次,没探索一次会产生多条样本放入样本池，每次从样本池随机选一个batch训练
def evaluate_expolre(agent, env_inst, evaluate_num_episodes):
    episodes_count = 0
    with tqdm(total=evaluate_num_episodes, desc='评估进度') as pbar:
        #正式采集num_episodes轮
        while episodes_count < evaluate_num_episodes:
            done = False
            obs = env_inst.reset()
            episode_reward = 0
            while not done:
                action, next_obs, reward, done, info = explore_one_step(agent, env_inst, obs, True)
                episode_reward += reward
                obs = next_obs
            agent.update_episode_rewards(episode_reward)
            episodes_count += 1
            #进度条前进1，自动根据总episode映射
            pbar.set_postfix({'episode': '%d' % (episodes_count), 'return': '%.3f' % np.mean(agent.episode_rewards[-10:])})
            pbar.update(1)
    print('evaluated total steps is:{}'.format(agent.total_steps))
    avg_reward = sum(agent.episode_rewards) / evaluate_num_episodes
    print(f"评估集平均奖励: {avg_reward:.2f}")
    print(f"最大回报: {max(agent.episode_rewards):.2f}")
    print(f"最小回报: {min(agent.episode_rewards):.2f}")

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
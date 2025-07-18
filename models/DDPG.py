import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic import BasicModel
from utils.display import show_train_procedure
from utils.logger import my_logger
from utils.parse_config import load_training_config

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.f1 = nn.Linear(state_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, inputs):
        x = F.relu(self.f1(inputs))
        x = F.relu(self.f2(x))
        x = F.tanh(self.f3(x))
        return x*self.action_bound          #必须是具体action的值，值的范围是[-bound,bound]之间
#状态价值模型
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.f1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.f1(x))
        return self.f2(x)         #得到的是估计Q值，不是V值

class DDPG(BasicModel):
    def __init__(self, actor_learning_rate, critic_learning_rate, gamma, noise_std, tau, action_bound, hidden_dim, state_dim, action_dim):
        super().__init__()
        self.actor_loss = []
        self.critic_loss = []
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.noise_std = noise_std
        self.tau = tau
        #actor网络
        self.actor_net = ActorNetwork(state_dim, hidden_dim, action_dim, action_bound)
        self.target_actor_net = ActorNetwork(state_dim, hidden_dim, action_dim, action_bound)
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_learning_rate)  # target网络拷贝actor_net网络参数

        # critic网络
        self.critic_net = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.target_critic_net = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_learning_rate)  #target网络拷贝critic_net网络参数

    # state只可能是一条样本，不能是batch
    def take_action(self, states_tensor, is_evaluate=False)->torch.Tensor:
        self.total_steps += 1
        actions = self.actor_net(states_tensor)
        if not is_evaluate:
            select_action = self.take_action_with_noise(actions, self.noise_std)
        else:
            select_action = actions
        select_action = torch.clamp(select_action, -self.actor_net.action_bound, self.actor_net.action_bound)
        return select_action  #tensor,得到的是action值

    def load_state_dict_eval(self):
        if not os.path.exists('ddppg_actor_net.pth'):
            raise FileNotFoundError(f"模型文件未找到：ddppg_actor_net.pth")
        self.actor_net.load_state_dict(torch.load('ddppg_actor_net.pth'))

    def save_state_dict(self):
        torch.save(self.actor_net.state_dict(), '../evaluate/ddppg_actor_net.pth')

    def update_episode_rewards(self, episode_reward):
        self.episode_rewards.append(episode_reward)

    def update_target_model(self):
        super().update_model_params(self.target_actor_net, self.actor_net, True, self.tau)
        super().update_model_params(self.target_critic_net, self.critic_net, True, self.tau)

    def show_train_result(self):
        my_logger.info("train ddqn end, actor loss count:{}, critic loss count:{}".format(len(self.actor_loss),len(self.critic_loss)))
        show_train_procedure(actor_loss = self.actor_loss, critic_loss= self.critic_loss, episode_rewards = self.episode_rewards)

    def train(self, samples):
        states_tensor = torch.tensor(samples['states'], dtype=torch.float32)
        actions_tensor = torch.tensor(samples['actions'], dtype=torch.float32)
        if actions_tensor.dim() == 1:
            actions_tensor = actions_tensor.unsqueeze(1)
        rewards_tensor = torch.tensor(samples['rewards'], dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(samples['next_states'], dtype=torch.float32)
        dones_tensor = torch.tensor(samples['dones'], dtype=torch.float32).unsqueeze(1)

        #使用target完了计算next_Q_S_A
        next_actions = self.target_actor_net(next_states_tensor)
        next_Q_S_A = self.target_critic_net(next_states_tensor, next_actions)

        # 计算td target
        td_target = rewards_tensor + self.gamma* next_Q_S_A*(1-dones_tensor.float())

        #使用训练网络计算Q(S,A)并拟合td target，critic网络越准越好
        Q_S_A = self.critic_net(states_tensor, actions_tensor)
        critic_loss = (Q_S_A - td_target.detach()).pow(2).mean()
        # my_logger.debug('ddpg critic loss:{}'.format(critic_loss.item()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_loss.append(critic_loss.item())
        self.critic_optimizer.step()

        #此处需要重新计算Q_S_A,否则计算图会出错，计算actor loss，actor网络产生对应action的Q价值越大越好
        actions = self.actor_net(states_tensor)
        Q_S_A = self.critic_net(states_tensor, actions)
        actor_loss = -torch.mean(Q_S_A)
        # my_logger.debug('ddpg actor loss:', actor_loss.item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_loss.append(actor_loss.item())
        self.actor_optimizer.step()

        #更新target网络
        self.update_target_model()

def ddpg_get_trained_model(env_params):
    state_dim, state_count, action_dim, action_count = env_params
    #解析训练参数
    params = load_training_config('DDPG')
    my_logger.info("train ddpg begin.")
    train_model = DDPG(params['actor_learning_rate'],
                        params['critic_learning_rate'],
                        params['gamma'],
                        params['noise_std'],
                        params['tau'],
                        params['action_bound'],
                        params['hidden_dim'],
                        state_dim,
                        action_dim)
    return train_model

if __name__ == '__main__':
    batch_samples = {
        'states':([1,2,3,4,5], [2,3,4,5,6], [11,12,13,14,15], [21,22,23,24,25], [201,202,230,204,205]),
        'actions':([0, 1], [1, 1], [0, 2], [2, 1], [3, 1]),
        'rewards':(11, 21, 22, 0, 30),
        'next_states':([2,3,4,5,6], [11,12,13,14,15], [1,2,3,4,5], [201,202,230,204,205], [21,22,23,24,25]),
        'dones':(1, 0, 0, 1, 0),
        'infos':('test 1', 'test 2', 'test 3', 'test 4', 'test 5', ),
    }
    train_model = DDPG(1e-4, 1e-3,0.99, 0.1,  0.05, 2, 128, 5, 2)
    for _ in range(100):
        train_model.train(batch_samples)
    train_model.show_train_result()

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic import BasicModel
from utils.display import show_train_procedure
from utils.logger import my_logger
from utils.parse_config import load_training_config

#离散动作的actor模型
class ActorNetworkDiscrete(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_count):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_count)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)      #action个数，得到action的概率分布

#连续动作的actor模型
class ActorNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.action_bound * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std            #输出均值于方差

#状态价值模型
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)       #评估的是状态价值V


class PPOBase(BasicModel):
    def __init__(self, critic_learning_rate, hidden_dim, state_dim):
        super().__init__()
        self.actor_loss = []
        self.critic_loss = []
        self.critic_learning_rate = critic_learning_rate
        # critic网络
        self.critic_net = CriticNetwork(state_dim, hidden_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(),
                                                 lr=critic_learning_rate)  # target网络拷贝critic_net网络参数
        # 记录总reward
        self.epsode_rewards = []

    def update_epsode_rewards(self, epsode_reward):
        self.epsode_rewards.append(epsode_reward)

    def update_target_model(self):
        raise Exception('ppo no target net.')

    def show_procedure(self):
        my_logger.info("train ddqn end, actor loss count:{}, critic loss count:{}".format(len(self.actor_loss),
                                                                                          len(self.critic_loss)))
        show_train_procedure(actor_loss=self.actor_loss, critic_loss=self.critic_loss,
                             epsode_rewards=self.epsode_rewards)
#离散模型
class PPOv1(PPOBase):
    def __init__(self, epochs, actor_learning_rate, critic_learning_rate, eps, gamma, lamda, hidden_dim, state_dim, action_count):
        super().__init__(critic_learning_rate, hidden_dim, state_dim)
        self.epochs = epochs
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.eps = eps
        self.gamma = gamma
        self.lamda = lamda
        #actor网络
        self.actor_net = ActorNetworkDiscrete(state_dim, hidden_dim, action_count)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_learning_rate)  # target网络拷贝actor_net网络参数
        #记录总reward
        self.epsode_rewards = []

    # state只可能是一条样本，不能是batch
    def take_action(self, states_tensor)->torch.Tensor:
        actions = self.actor_net(states_tensor)        #输出的已经是概率分布
        select_action = self.take_action_probility(actions)
        return select_action.item()  #tensor,得到的是action值

    def train(self, samples):
        states_tensor = torch.tensor(samples['states'], dtype=torch.float32)
        actions_tensor = torch.tensor(samples['actions'], dtype=torch.float32)
        if actions_tensor.dim() == 1:
            actions_tensor = actions_tensor.unsqueeze(1)
        rewards_tensor = torch.tensor(samples['rewards'], dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(samples['next_states'], dtype=torch.float32)
        dones_tensor = torch.tensor(samples['dones'], dtype=torch.float32).unsqueeze(1)

        #使用target完了计算next_V_S,使用的是状态价值V,不是状态动作价值Q
        next_V_S = self.critic_net(next_states_tensor)

        # 计算td target
        td_target = rewards_tensor + self.gamma* next_V_S*(1-dones_tensor.float())
        td_delta = td_target - self.critic_net(states_tensor)

        #计算优势
        advantage = self.compute_advantage(self.gamma, self.lamda, td_delta)
        old_log_probs = torch.log(self.actor_net(states_tensor).gather(1, actions_tensor)).detach()
        for _ in range(self.epochs):
            new_log_probs = torch.log(self.actor_net(states_tensor).gather(1, actions_tensor))
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic_net(states_tensor), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_loss.append(actor_loss.item())
            critic_loss.backward()
            self.actor_loss.append(actor_loss.item())
            self.actor_optimizer.step()
            self.critic_optimizer.step()

#连续模型
class PPOv2(PPOBase):
    def __init__(self, epochs, actor_learning_rate, critic_learning_rate, eps, gamma, lamda, action_bound, hidden_dim, state_dim, action_dim):
        super().__init__(critic_learning_rate, hidden_dim, state_dim)
        self.epochs = epochs
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.eps = eps
        self.gamma = gamma
        self.lamda = lamda
        self.action_bound = action_bound
        #actor网络
        self.actor_net = ActorNetworkDiscrete(state_dim, hidden_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_learning_rate)  # target网络拷贝actor_net网络参数
        #记录总reward
        self.epsode_rewards = []

    # state只可能是一条样本，不能是batch,按照正太分布密度函数采样
    def take_action(self, states_tensor)->torch.Tensor:
        mu, sigma = self.actor_net(states_tensor)
        select_action = self.take_action_with_distribution(mu, sigma)
        return select_action.item()  #tensor,得到的是action值

    def train(self, samples):
        states_tensor = torch.tensor(samples['states'], dtype=torch.float32)
        actions_tensor = torch.tensor(samples['actions'], dtype=torch.float32)
        if actions_tensor.dim() == 1:
            actions_tensor = actions_tensor.unsqueeze(1)
        rewards_tensor = torch.tensor(samples['rewards'], dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(samples['next_states'], dtype=torch.float32)
        dones_tensor = torch.tensor(samples['dones'], dtype=torch.float32).unsqueeze(1)

        #使用target完了计算next_V_S,使用的是状态价值V,不是状态动作价值Q
        next_V_S = self.critic_net(next_states_tensor)

        # 计算td target
        td_target = rewards_tensor + self.gamma* next_V_S*(1-dones_tensor.float())
        td_delta = td_target - self.critic_net(states_tensor)

        #计算优势
        advantage = self.compute_advantage(self.gamma, self.lamda, td_delta)
        mu, std = self.actor_net(states_tensor)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions_tensor)
        for _ in range(self.epochs):
            mu, std = self.actor_net(states_tensor)
            action_dists = torch.distributions.Normal(mu.detach(), std.detach())
            new_log_probs = action_dists.log_prob(actions_tensor)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic_net(states_tensor), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_loss.append(actor_loss.item())
            critic_loss.backward()
            self.actor_loss.append(actor_loss.item())
            self.actor_optimizer.step()
            self.critic_optimizer.step()

#离散模型
def ppov1_get_trained_model(env_params):
    state_dim, state_count, action_dim, action_count = env_params
    #解析训练参数
    params = load_training_config('PPO')
    my_logger.info("train ppov1 begin.")
    train_model = PPOv1(params['epochs'],
                        params['actor_learning_rate'],
                        params['critic_learning_rate'],
                        params['eps'],
                        params['gamma'],
                        params['lamda'],
                        params['hidden_dim'],
                        state_dim,
                        action_count)
    return train_model
#连续模型
def ppov2_get_trained_model(env_params):
    state_dim, state_count, action_dim, action_count = env_params
    #解析训练参数
    params = load_training_config('PPO')
    my_logger.info("train ppov2 begin.")
    train_model = PPOv2(params['epochs'],
                        params['actor_learning_rate'],
                        params['critic_learning_rate'],
                        params['eps'],
                        params['gamma'],
                        params['lamda'],
                        params['action_bound'],
                        params['hidden_dim'],
                        state_dim,
                        action_dim)
    return train_model

if __name__ == '__main__':
    batch_samples = {
        'states':([1,2,3,4,5], [2,3,4,5,6], [11,12,13,14,15], [21,22,23,24,25], [201,202,230,204,205]),
        'actions':(0, 1, 2, 1, 2),
        'rewards':(11, 21, 22, 0, 30),
        'next_states':([2,3,4,5,6], [11,12,13,14,15], [1,2,3,4,5], [201,202,230,204,205], [21,22,23,24,25]),
        'dones':(1, 0, 0, 1, 0),
        'infos':('test 1', 'test 2', 'test 3', 'test 4', 'test 5', ),
    }
    train_model = PPOv1(5, 1e-4, 1e-3, 0.2, 0.99, 0.95, 128, 5, 3)
    for _ in range(100):
        train_model.train(batch_samples)
    train_model.show_procedure()

    batch_samples = {
        'states':([1,2,3,4,5], [2,3,4,5,6], [11,12,13,14,15], [21,22,23,24,25], [201,202,230,204,205]),
        'actions':([0, 1, 3], [1, 1, 3], [0, 3, 2], [2, 3, 1], [3, 1, 3]),
        'rewards':(11, 21, 22, 0, 30),
        'next_states':([2,3,4,5,6], [11,12,13,14,15], [1,2,3,4,5], [201,202,230,204,205], [21,22,23,24,25]),
        'dones':(1, 0, 0, 1, 0),
        'infos':('test 1', 'test 2', 'test 3', 'test 4', 'test 5', ),
    }
    train_model = PPOv2(5, 1e-4, 1e-3, 0.2, 0.99, 0.95, 3, 128, 5, 3)
    for _ in range(100):
        train_model.train(batch_samples)
    train_model.show_procedure()
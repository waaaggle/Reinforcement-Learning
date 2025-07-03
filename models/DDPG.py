import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic import BasicModel
from show.loss_display import show_loss
from train.logger import my_logger

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, bound):
        super().__init__()
        self.f1 = nn.Linear(state_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, action_dim)
        self.bound = bound

    def forward(self, inputs):
        x = F.relu(self.f1(inputs))
        x = F.tanh(self.f2(x))
        return x*self.bound          #必须是具体action的值，值的范围是[-bound,bound]之间

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
    def __init__(self, learning_rate, gamma, noise_std, tau, bound, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.actor_loss = []
        self.critic_loss = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.noise_std = noise_std
        self.tau = tau
        #actor网络
        self.actor_net = ActorNetwork(state_dim, hidden_dim, action_dim, bound)
        self.target_actor_net = ActorNetwork(state_dim, hidden_dim, action_dim, bound)
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=learning_rate)  # target网络拷贝actor_net网络参数

        # critic网络
        self.critic_net = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.target_critic_net = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=learning_rate)  #target网络拷贝critic_net网络参数

    # state只可能是一条样本，不能是batch
    def take_action(self, states)->torch.Tensor:
        actions = self.actor_net(states)
        select_action = super().take_action_with_noise(actions, self.noise_std)
        return select_action  #tensor,得到的是action值

    def update_target_model(self):
        super().update_model_params(self.target_actor_net, self.actor_net, True, self.tau)
        super().update_model_params(self.target_critic_net, self.critic_net, True, self.tau)

    def show_loss_procedure(self):
        show_loss(actor_loss = self.actor_loss, critic_loss= self.critic_loss)

    def train(self, samples):
        states_tensor = torch.tensor(samples['states'], dtype=torch.float32)
        actions_tensor = torch.tensor(samples['actions'], dtype=torch.float32)
        rewards_tensor = torch.tensor(samples['rewards'], dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(samples['next_states'], dtype=torch.float32)
        dones_tensor = torch.tensor(samples['dones'], dtype=torch.float32).unsqueeze(1)

        #使用target完了计算next_Q_S_A
        next_actions = self.target_actor_net(next_states_tensor)
        next_Q_S_A = self.target_critic_net(next_states_tensor, next_actions)

        # 计算td target
        td_target = rewards_tensor + self.gamma* next_Q_S_A*(1-dones_tensor)

        #使用训练网络计算Q(S,A)并拟合td target，critic网络越准越好
        Q_S_A = self.critic_net(states_tensor, actions_tensor)
        critic_loss = (Q_S_A - td_target.detach()).pow(2).mean()
        my_logger.debug('ddpg critic loss:', critic_loss)
        self.critic_optimizer.zero_grad()
        self.critic_loss.append(critic_loss.detach())
        critic_loss.backward()
        self.critic_optimizer.step()

        #此处需要重新计算Q_S_A,否则计算图会出错，计算actor loss，actor网络产生对应action的Q价值越大越好
        actions = self.actor_net(states_tensor)
        Q_S_A = self.critic_net(states_tensor, actions)
        actor_loss = -torch.mean(Q_S_A)
        my_logger.debug('ddpg actor loss:', actor_loss)
        self.actor_loss.append(actor_loss.detach())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #更新target网络
        self.update_target_model()

if __name__ == '__main__':
    batch_samples = {
        'states':([1,2,3,4,5], [2,3,4,5,6], [11,12,13,14,15], [21,22,23,24,25], [201,202,230,204,205]),
        'actions':(0, 1, 2, 1, 2),
        'rewards':(11, 21, 22, 0, 30),
        'next_states':([2,3,4,5,6], [11,12,13,14,15], [1,2,3,4,5], [201,202,230,204,205], [21,22,23,24,25]),
        'dones':(1, 0, 0, 1, 0),
        'infos':('test 1', 'test 2', 'test 3', 'test 4', 'test 5', ),
    }
    train_model = DDPG(1e-4, 0.99, 0.1,  0.05, 2, 5, 1, 128)
    for _ in range(100):
        train_model.train(batch_samples)
    train_model.show_loss_procedure()

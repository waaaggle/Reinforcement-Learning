import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic import BasicModel
from utils.display import show_train_procedure
from utils.logger import my_logger
from utils.parse_config import load_training_config

#状态动作价值模型
class DDQN_QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_count):
        super().__init__()
        self.f1 = nn.Linear(state_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, action_count)         #输出是离散的，action_count为action取值的个数

    def forward(self, inputs):
        x = F.relu(self.f1(inputs))
        x = F.relu(self.f2(x))
        return self.f3(x)         #必须要relu，需要输出每个action的输出Q值，不是概率

class DDQN(BasicModel):
    def __init__(self, learning_rate, epsilon_decay_steps, epsilon_start, epsilon_end, gamma, tau, hidden_dim, state_dim, action_count):
        super().__init__()
        self.loss = list()
        self.learning_rate = learning_rate
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.tau = tau
        self.q_net = DDQN_QNetwork(state_dim, hidden_dim, action_count)
        self.target_q_net = DDQN_QNetwork(state_dim, hidden_dim, action_count)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)  #target网络拷贝q_net网络参数

    #state只可能是一条样本，不能是batch
    def take_action(self, states_tensor, is_evaluate=False)->torch.Tensor:
        self.total_steps += 1
        actions = self.q_net(states_tensor)
        if not is_evaluate:
            epsilon = max(self.epsilon_end, self.epsilon_start - self.total_steps / self.epsilon_decay_steps * (self.epsilon_start - self.epsilon_end))
            select_action = self.take_action_epsilon_greedy(epsilon, actions)
        else:
            select_action = torch.argmax(actions, dim=-1)

        return select_action    #得到的是action的编号，不是action值

    def load_state_dict_eval(self):
        if not os.path.exists('ddqn_q_net.pth'):
            raise FileNotFoundError(f"模型文件未找到：ddqn_q_net.pth")
        self.q_net.load_state_dict(torch.load('ddqn_q_net.pth'))

    def save_state_dict(self):
        torch.save(self.q_net.state_dict(), '../evaluate/ddqn_q_net.pth')

    def update_episode_rewards(self, episode_reward):
        self.episode_rewards.append(episode_reward)

    def update_target_model(self):
        if self.total_steps % 1000 == 0:
            super().update_model_params(self.target_q_net, self.q_net)

    def show_train_result(self):
        my_logger.info("train ddqn end, loss count:{}".format(len(self.loss)))
        show_train_procedure(ddqn_loss = self.loss, episode_rewards = self.episode_rewards)

    def train(self, samples):
        states_tensor = torch.tensor(samples['states'], dtype=torch.float32)
        actions_tensor = torch.tensor(samples['actions'], dtype=torch.int64).unsqueeze(1)
        rewards_tensor = torch.tensor(samples['rewards'], dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(samples['next_states'], dtype=torch.float32)
        dones_tensor = torch.tensor(samples['dones'], dtype=torch.float32).unsqueeze(1)

        # print('states_tensor:', states_tensor)
        # print('actions_tensor:', actions_tensor)
        # print('rewards_tensor:', rewards_tensor)
        # print('next_states_tensor:', next_states_tensor)
        # print('dones_tensor:', dones_tensor)

        #使用train网络获得当前当前action价值以及下个state最大价值action
        current_state_value = self.q_net(states_tensor)
        next_state_value = self.q_net(next_states_tensor)
        #使用target网络计算下个state的最大action的价值
        target_next_state_value = self.target_q_net(next_states_tensor)
        #计算Q(S,A)
        Q_S_A = current_state_value.gather(1, actions_tensor)
        #计算下个state最大action，采用训练网络
        next_state_max_action = torch.argmax(next_state_value, dim=1).unsqueeze(1)
        # print(next_state_value)
        #计算下个state最大action价值，采用target网络
        # print(next_state_max_action)
        next_Q_S_A_max = target_next_state_value.gather(1, next_state_max_action)
        # 拟合目标为：Q(S,A)<---R + gamma*Q(S_next,A_next_max)*(1-dones)
        td_target = rewards_tensor + self.gamma*next_Q_S_A_max*(1-dones_tensor.float())

        #均方误差损失，网络越准越好
        loss = (Q_S_A - td_target.detach()).pow(2).mean()
        # my_logger.debug('ddqn loss:', loss.item())
        self.loss.append(loss.item())
        print(loss.item())

        #反向传播更新网络
        self.q_net_optimizer.zero_grad()
        loss.backward()
        self.q_net_optimizer.step()

        #更新target网络
        self.update_target_model()

def ddqn_get_trained_model(env_params):
    state_dim, state_count, action_dim, action_count = env_params
    #解析训练参数
    params = load_training_config('DDQN')
    my_logger.info("train ddqn begin.")
    train_model = DDQN(params['learning_rate'],
                        params['epsilon_decay_steps'],
                        params['epsilon_start'],
                        params['epsilon_end'],
                        params['gamma'],
                        params['tau'],
                        params['hidden_dim'],
                        state_dim,
                        action_count)
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
    train_model = DDQN(1e-3, 50000, 1.0, 0.1, 0.99, 0.1, 128, 5, 3)
    for _ in range(100):
        train_model.train(batch_samples)
    train_model.show_train_result()

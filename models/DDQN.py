import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic import BasicModel
from show.loss_display import show_loss
from train.logger import my_logger

class DDQN_QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super().__init__()
        self.f1 = nn.Linear(state_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, output_dim)         #output_dim为action取值的个数

    def forward(self, inputs):
        x = F.relu(self.f1(inputs))
        return self.f2(x)         #必须要relu，需要输出每个action的输出Q值，不是概率

class DDQN(BasicModel):
    def __init__(self, learning_rate, epsilon, gamma, tau, state_dim, hidden_dim, output_dim):
        super().__init__()
        self.loss = list()
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.q_net = DDQN_QNetwork(state_dim, hidden_dim, output_dim)
        self.target_q_net = DDQN_QNetwork(state_dim, hidden_dim, output_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)  #target网络拷贝q_net网络参数

    #state只可能是一条样本，不能是batch
    def take_action(self, state)->torch.Tensor:
        actions = self.q_net(state)
        select_action = super().take_action_epsilon_greedy(self.epsilon, actions)
        return select_action    #得到的是action的编号，不是action值

    def update_target_model(self):
        super().update_model_params(self.target_q_net, self.q_net)

    def show_loss_procedure(self):
        show_loss(ddqn_loss = self.loss)

    def train(self, samples):
        states_tensor = torch.tensor(samples['states'], dtype=torch.float32)
        actions_tensor = torch.tensor(samples['actions']).unsqueeze(1)
        rewards_tensor = torch.tensor(samples['rewards'], dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(samples['next_states'], dtype=torch.float32)
        dones_tensor = torch.tensor(samples['dones'], dtype=torch.float32).unsqueeze(1)

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
        # 拟合目标为：Q(S,A)<---R + gamma*Q(S',A'_max)*(1-dones)
        td_target = rewards_tensor + self.gamma*next_Q_S_A_max*(1-dones_tensor.float())

        #均方误差损失，网络越准越好
        loss = (Q_S_A - td_target.detach()).pow(2).mean()
        my_logger.debug('ddqn loss:', loss)
        self.loss.append(loss.detach())

        #反向传播更新网络
        self.q_net_optimizer.zero_grad()
        loss.backward()
        self.q_net_optimizer.step()

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
    train_model = DDQN(1e-3, 0.1, 0.99, 0.1, 5, 128, 3)
    for _ in range(100):
        train_model.train(batch_samples)
    train_model.show_loss_procedure()

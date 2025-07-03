from abc import ABC, abstractmethod
import torch
import random

class BasicModel(ABC):
    @abstractmethod
    def take_action(self, states):
        pass
    @abstractmethod
    def update_target_model(self):
        ...     #于pass等价
    @abstractmethod
    def show_loss_procedure(self):
        pass
    @abstractmethod
    def train(self, samples):
        ...
    #根据训练网络参数更新target网络参数
    def update_model_params(self, train_model, target_model, soft_update=False, tau=0.0):
        # target_model = (1.0-tau)*target_model + tau*train_model
        if soft_update:
            # 按比例组合参数，param.data为tensor
            for target_param, q_param in zip(target_model.parameters(), train_model.parameters()):
                target_param.data.copy_((1.0 - tau) * target_param.data + tau * q_param.data)
        else:
            # 训练网络更新target网络
            target_model.load_state_dict(train_model.state_dict())

    #GAE函数，广义优势估计
    #lamada用于控制优势受多少step步数影响
    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:        #每一步的delta近似于每一步的advantage，把所有step的delta累加起来就是每一步step的优势
            #每个state的优势是反向通过delta近似累加计算出来的，detla为通过rt + gamma*v(st+1)-V(st)近似替代At=Q(st,at)-V(st)
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        #返回的是每个 state 对应的 advantage，每个 advantage 是以该 state 为起点，未来 TD 残差的加权和（指数衰减），不是所有 state 的 advantage 的累加
        return torch.tensor(advantage_list, dtype=torch.float32)

    #以概率ε随机探索，以1-ε概率选择当前最优动作
    #与策略无关，一般是off-policy算法采用
    def take_action_epsilon_greedy(self, epsilon, actions):
        if actions.dim() == 1:  # actions是按batch输入，并且得到的是action的概率分布
            if random.random() < epsilon:  # 随机探索
                indices = torch.randint(0, len(actions), ())
            else:  # 最优选择
                indices = torch.argmax(actions)
        else:
            raise Exception('invalid action tensors:{}, dim:{}'.format(actions, actions.dim()))
        return indices

    #按照概率分布采样，Policy-gradient算法采用
    def take_action_probility(self, actions):
        # epsilon代表探索概率
        # actions = torch.softmax(actions, dim=1)
        t = actions.softmax(dim=1)
        dist = torch.distributions.Categorical(actions)
        return dist.sample()    #得到tensor，返回的是采样索引

    #确定性动作增加正太分布噪声，主要是DDPG
    def take_action_with_noise(self, actions, noise_std):
        noise = torch.rand_like(actions) * noise_std  #rand_like生成正太分布
        return actions+noise

if __name__ == "__main__":
    test_tensor = torch.tensor([[1,2,3,4,5],[3,4,5,6,7]], dtype=torch.float)
    t = take_action_probility(test_tensor)
    print(t)

    t = take_action_epsilon_greedy(1, test_tensor)
    print(t)

    t = take_action_with_noise(test_tensor, 0.1)
    print(t)
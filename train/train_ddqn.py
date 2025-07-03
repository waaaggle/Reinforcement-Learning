from enviroment.virtual_env import start_virtual_env
from enviroment.gym_env import start_gym_env
import models.DDQN as DDQN
from train.exploration import  off_policy_expolre, on_policy_expolre
from utils.logger import my_logger
from utils.parse_config import load_training_config
from utils.sample_pool import SamplePool

def train():
    #解析训练参数
    params = load_training_config()

    #准备验证环境
    # env_inst, state_dim, state_count, action_dim, action_count = start_virtual_env('discrete_action_v1')  #产生随机样本，必须是输出离散动作，维度为action个税
    env_inst, state_dim, state_count, action_dim, action_count = start_gym_env('LunarLander-v3')

    #准备模型
    my_logger.info("train ddqn begin.")
    train_model = DDQN.DDQN(params['learning_rate'],
                            params['epsilon'],
                            params['gamma'],
                            params['tau'],
                            state_dim,
                            params['hidden_dim'],
                            action_count)

    #准备样本池
    samp_pool = SamplePool(params['sample_pool_size'])

    # 探索学习
    off_policy_expolre(samp_pool, train_model, env_inst, params['num_episodes'], params['num_episodes_per_train'], params['batch_size'])
    # on_policy_expolre(samp_pool, train_model, env_inst, params['num_episodes'], params['batch_size'])

    #关闭采样环境
    env_inst.close()
    my_logger.info("train ddqn end, loss count:{}".format(len(train_model.loss)))

    #显示损失
    train_model.show_loss_procedure()


if __name__ == '__main__':
    train()
from enviroment.virtual_env import start_virtual_env
from enviroment.gym_env import start_gym_env
import models.DDPG as DDPG
from train.exploration import  off_policy_expolre, on_policy_expolre
from utils.logger import my_logger
from utils.parse_config import load_training_config
from utils.sample_pool import SamplePool

def train():
    #解析训练参数
    params = load_training_config()

    #准备验证环境
    # env_inst, state_dim, state_count, action_dim, action_count = start_virtual_env('not_discrete_action_v1')  #产生随机样本，必须是输出连续动作，维度为1
    env_inst, state_dim, state_count, action_dim, action_count = start_gym_env('BipedalWalker-v3')
    #准备模型
    my_logger.info("train ddpg begin.")
    train_model = DDPG.DDPG(params['learning_rate'],
                            params['gamma'],
                            params['noise_std'],
                            params['tau'],
                            params['action_bound'],
                            state_dim,
                            action_dim,
                            params['hidden_dim'])
    #准备样本池
    samp_pool = SamplePool(params['sample_pool_size'])

    #探索学习
    off_policy_expolre(samp_pool, train_model, env_inst, params['num_episodes'], params['num_episodes_per_train'], params['batch_size'])
    # on_policy_expolre(samp_pool, train_model, env_inst, params['num_episodes'], params['batch_size'])

    # 关闭采样环境
    env_inst.close()
    my_logger.info("train ddqn end, actor loss count:{}, critic loss count:{}".format(len(train_model.actor_loss), len(train_model.critic_loss)))

    #显示损失
    train_model.show_loss_procedure()

if __name__ == '__main__':
    train()
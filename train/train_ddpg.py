from samples.virtual_env import start_virtual_env
from samples.gym_env import start_gym_env
import models.DDPG as DDPG
from train.exploration import  expolre
from train.logger import my_logger

def train(epochs):
    #产生随机样本，必须是输出连续动作，维度为1
    # env_inst, state_dim, state_count, action_dim, action_count = start_virtual_env('not_discrete_action_v1')
    env_inst, state_dim, state_count, action_dim, action_count = start_gym_env('BipedalWalker-v3')
    #准备模型
    my_logger.info("train ddpg begin.")
    train_model = DDPG.DDPG(1e-4, 0.99, 0.1, 0.05, 1, state_dim, action_dim, 128)
    #训练
    for _ in range(epochs):
        samples_pool = expolre(train_model, env_inst, 100)
        batch_samples = samples_pool.samples(10)
        train_model.train(batch_samples)

    # 关闭采样环境
    env_inst.close()
    my_logger.info("train ddpg end.")

    #显示损失
    train_model.show_loss_procedure()

if __name__ == '__main__':
    train(100)
from samples.virtual_env import start_virtual_env
from samples.gym_env import start_gym_env
import models.DDQN as DDQN
from train.exploration import  expolre
from train.logger import my_logger

def train(epochs):
    #产生随机样本，必须是输出离散动作，维度为action个税
    # env_inst, state_dim, state_count, action_dim, action_count = start_virtual_env('discrete_action_v1')
    env_inst, state_dim, state_count, action_dim, action_count = start_gym_env('LunarLander-v3')

    #准备模型
    my_logger.info("train ddqn begin.")
    train_model = DDQN.DDQN(1e-4, 0.1, 0.99, 0.05, state_dim, 128, action_count)

    #准备模型
    for _ in range(epochs):
        samples_pool = expolre(train_model, env_inst, 100)
        batch_samples = samples_pool.samples(10)
        train_model.train(batch_samples)

    #关闭采样环境
    env_inst.close()
    my_logger.info("train ddqn end.")

    #显示损失
    train_model.show_loss_procedure()


if __name__ == '__main__':
    train(100)
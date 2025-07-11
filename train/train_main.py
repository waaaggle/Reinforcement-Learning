
from train.exploration import  off_policy_expolre, on_policy_expolre
from utils.logger import my_logger
from utils.parse_config import load_training_config
from utils.sample_pool import SamplePool
from registry import RegistryInfo

def train(model_name, env_name):
    #检查动作类型是否匹配
    if not RegistryInfo.check_model_support_env(model_name, env_name):
        return
    #解析训练参数
    params = load_training_config('TRAINING')

    #准备验证环境
    env_inst = RegistryInfo.get_train_env(env_name)

    #准备模型
    train_model = RegistryInfo.get_train_model(model_name, env_inst)

    #准备样本池
    samp_pool = SamplePool(params['sample_pool_size'])

    #探索学习
    off_policy_expolre(samp_pool, train_model, env_inst, params['num_episodes'], params['warmup_steps'], params['num_episodes_per_train'], params['batch_size'])
    # on_policy_expolre(samp_pool, train_model, env_inst, params['num_episodes'], params['batch_size'])

    # 关闭采样环境
    env_inst.close()

    #显示损失
    train_model.show_procedure()

if __name__ == '__main__':
    # train('DDQN', 'LunarLander-v3')
    # train('DDQN', 'VIRTUAL-v1')
    train('DDPG', 'BipedalWalker-v3')
    # train('DDPG', 'VIRTUAL-v2')
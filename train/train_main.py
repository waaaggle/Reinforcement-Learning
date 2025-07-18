import sys
import time
from train.exploration import  off_policy_expolre, on_policy_expolre
from utils.logger import my_logger
from utils.parse_config import load_training_config
from utils.sample_pool import SamplePool
from models.registry import RegistryInfo

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("总耗时：%.2f 秒" % (end_time - start_time))
        return result
    return wrapper

@timeit
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
    if RegistryInfo.is_on_policy_train(model_name):
        on_policy_expolre(samp_pool, train_model, env_inst, params['num_episodes'])
    else:
        off_policy_expolre(samp_pool, train_model, env_inst, params['num_episodes'], params['warmup_steps'], params['num_episodes_per_train'], params['batch_size'])

    # 关闭采样环境
    env_inst.close()

    #保存参数
    train_model.save_state_dict()

    #显示损失
    train_model.show_train_result()

if __name__ == '__main__':
    train('DDQN', 'LunarLander-v3')
    train('DDQN', 'VIRTUAL-v1')
    train('DDPG', 'BipedalWalker-v3')
    train('DDPG', 'VIRTUAL-v2')
    #
    train('PPOv1', 'LunarLander-v3')
    train('PPOv1', 'VIRTUAL-v1')
    train('PPOv2', 'BipedalWalker-v3')
    train('PPOv2', 'VIRTUAL-v2')
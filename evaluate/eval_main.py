import time
from train.exploration import  evaluate_expolre
from utils.parse_config import load_training_config
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
def eval(model_name, env_name):
    #检查动作类型是否匹配
    if not RegistryInfo.check_model_support_env(model_name, env_name):
        return
    #解析训练参数
    params = load_training_config('TRAINING')

    #准备验证环境
    env_inst = RegistryInfo.get_train_env(env_name)

    #准备模型
    eval_model = RegistryInfo.get_train_model(model_name, env_inst)

    #load参数
    eval_model.load_state_dict_eval()

    #探索学习
    evaluate_expolre(eval_model, env_inst, params['evaluate_num_episodes'])

    # 关闭采样环境
    env_inst.close()

    #显示损失
    eval_model.show_evaluate_result()

if __name__ == '__main__':
    eval('DDQN', 'LunarLander-v3')
    # eval('DDQN', 'VIRTUAL-v1')
    eval('DDPG', 'BipedalWalker-v3')
    # eval('DDPG', 'VIRTUAL-v2')
    #
    eval('PPOv1', 'LunarLander-v3')
    # eval('PPOv1', 'VIRTUAL-v1')
    eval('PPOv2', 'BipedalWalker-v3')
    # eval('PPOv2', 'VIRTUAL-v2')
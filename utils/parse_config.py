import os
import configparser

def load_training_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.ini')

    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')

    # 解析 TRAINING 部分的参数并根据需要转型
    training_params = {}
    training_params['sample_pool_size'] = config.getint('TRAINING', 'sample_pool_size')
    training_params['num_episodes'] = config.getint('TRAINING', 'num_episodes')
    training_params['num_episodes_per_train'] = config.getint('TRAINING', 'num_episodes_per_train')
    training_params['batch_size'] = config.getint('TRAINING', 'batch_size')
    training_params['hidden_dim'] = config.getint('TRAINING', 'hidden_dim')
    training_params['action_bound'] = config.getint('TRAINING', 'action_bound')
    training_params['learning_rate'] = config.getfloat('TRAINING', 'learning_rate')
    training_params['gamma'] = config.getfloat('TRAINING', 'gamma')
    training_params['epsilon'] = config.getfloat('TRAINING', 'epsilon')
    training_params['tau'] = config.getfloat('TRAINING', 'tau')
    training_params['sigma'] = config.getfloat('TRAINING', 'sigma')
    training_params['noise_std'] = config.getfloat('TRAINING', 'noise_std')

    return training_params

# 使用示例
if __name__ == '__main__':
    params = load_training_config()
    print('训练参数:', params)
    # 例如用法
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    # ... 其它参数
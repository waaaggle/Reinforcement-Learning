import os
import configparser

def load_training_config(param_section):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.ini')

    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')

    # 解析 TRAINING 部分的参数并根据需要转型
    training_params = {}
    if param_section == 'TRAINING':
        training_params['sample_pool_size'] = config.getint(param_section, 'sample_pool_size')
        training_params['num_episodes'] = config.getint(param_section, 'num_episodes')
        training_params['evaluate_num_episodes'] = config.getint(param_section, 'evaluate_num_episodes')
        training_params['warmup_steps'] = config.getint(param_section, 'warmup_steps')
        training_params['num_episodes_per_train'] = config.getint(param_section, 'num_episodes_per_train')
        training_params['batch_size'] = config.getint(param_section, 'batch_size')
    elif param_section == 'DDQN':
        training_params['epsilon_decay_steps'] = config.getint(param_section, 'epsilon_decay_steps')
        training_params['hidden_dim'] = config.getint(param_section, 'hidden_dim')
        training_params['learning_rate'] = config.getfloat(param_section, 'learning_rate')
        training_params['gamma'] = config.getfloat(param_section, 'gamma')
        training_params['epsilon_start'] = config.getfloat(param_section, 'epsilon_start')
        training_params['epsilon_end'] = config.getfloat(param_section, 'epsilon_end')
        training_params['tau'] = config.getfloat(param_section, 'tau')
    elif param_section == 'DDPG':
        training_params['hidden_dim'] = config.getint(param_section, 'hidden_dim')
        training_params['action_bound'] = config.getint(param_section, 'action_bound')
        training_params['actor_learning_rate'] = config.getfloat(param_section, 'actor_learning_rate')
        training_params['critic_learning_rate'] = config.getfloat(param_section, 'critic_learning_rate')
        training_params['gamma'] = config.getfloat(param_section, 'gamma')
        training_params['tau'] = config.getfloat(param_section, 'tau')
        training_params['noise_std'] = config.getfloat(param_section, 'noise_std')
    elif param_section == 'PPO':
        training_params['hidden_dim'] = config.getint(param_section, 'hidden_dim')
        training_params['action_bound'] = config.getint(param_section, 'action_bound')
        training_params['epochs'] = config.getint(param_section, 'epochs')
        training_params['actor_learning_rate'] = config.getfloat(param_section, 'actor_learning_rate')
        training_params['critic_learning_rate'] = config.getfloat(param_section, 'critic_learning_rate')
        training_params['eps'] = config.getfloat(param_section, 'eps')
        training_params['gamma'] = config.getfloat(param_section, 'gamma')
        training_params['lamda'] = config.getfloat(param_section, 'lamda')
    else:
        raise Exception('invalid param type:{}'.format(param_section))

    return training_params

# 使用示例
if __name__ == '__main__':
    params = load_training_config('TRAINING')
    print('训练参数:', params)
    # 例如用法
    batch_size = params['batch_size']
    # ... 其它参数
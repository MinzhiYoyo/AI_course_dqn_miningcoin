import os
import time

import torch

from Game.miningcoin import MiningCoinEnv
from mining_coin_train_dqn import DQNAgent, get_time_info
from Game.gameui import GameUI


def dqn_play_game(model_dict_path, model_path=None, remark='', need_train=False, cmd_print=False, memory_mode=None):
    if need_train and memory_mode is None:
        raise ValueError('need_train is True but memory_mode is None.')

    log_dir = './log/play_game/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = open(log_dir + 'play_game_{}.log'.format(get_time_info()), 'w')
    log_file.write(
        'model_path: {}, model_dict_path: {}, remark: {}\n'.format(str(model_path), str(model_dict_path), remark))
    env = MiningCoinEnv()
    log_file.write('game_setting_info: {}\n'.format(env.game_setting_info))
    state_size = env.observation_space_length  # 你的状态参数数量
    action_size = env.action_space_length  # 你的动作参数数量
    agent = DQNAgent(state_size, action_size, model_path=model_path, model_dict_path=model_dict_path)
    state = env.reset()
    env_ui = GameUI(env)

    if memory_mode:
        agent.memory.load(memory_mode)

    while not env.done:
        action = agent.select_action(torch.tensor(state, dtype=torch.float32, device=agent.device)).item()
        next_state, reward, done, info = env.step(action)
        env_ui.draw(info=info, mode='dqn')
        log_file.write(info)
        log_file.write('\n')
        if cmd_print:
            print(info)
        if need_train:
            agent.memory.push(state, action, reward, next_state)
            agent.train()
        state = next_state
        time.sleep(0.5)
    log_file.close()
    print('Game over!')
    print('Current coins: {}'.format(env.current_coins))
    env_ui.quit()

if __name__ == '__main__':
    dqn_play_game(model_dict_path='model/experiment_1/model_dict_2024_01_28_10_32_37_best.pth', cmd_print=True)

import os
import time

from Game import gameui, miningcoin
def get_time_info():
    time_info = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    return time_info
env = miningcoin.MiningCoinEnv()
env.reset()
env_ui = gameui.GameUI(env)
info = None

log_dir = './log/human_game/'
# 如果不存在，则递归创建该文件夹
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = log_dir + 'human_play_{}.log'.format(get_time_info())
log_file = open(log_file_path, 'a')
while True:
    if info:
        print('info is {}.'.format(info))
    action = env_ui.draw(info=info)

    info = None
    if action is not None: 
        print('action is {}.'.format(action))
        if action in miningcoin.ACTION:
            _, _, _, info = env.step(action)
            log_file.write(info)
            log_file.write('\n')
        elif action < 0:
            print('Game over.')
            break
        else:
            env.reset()
            info = None
            env_ui.reset()
            print('reset the game.')
            log_file.write('==========\n')
if log_file:
    log_file.close()
env_ui.quit()

import time

from Game import gameui, miningcoin

env = miningcoin.MiningCoinEnv()
env.reset()
env_ui = gameui.GameUI(env)
info = None

while True:
    if info:
        print('info is {}.'.format(info))
    action = env_ui.draw(info=info)

    info = None
    if action is not None: 
        print('action is {}.'.format(action))
        if action in miningcoin.ACTION:
            _, _, _, info = env.step(action)
        elif action < 0:
            print('Game over.')
            break
        else:
            env.reset()
            info = None
            env_ui.reset()
            print('reset the game.')
    
env_ui.quit()

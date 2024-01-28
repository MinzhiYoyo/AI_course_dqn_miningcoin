import time

from Game import gameui, miningcoin

env = miningcoin.MiningCoinEnv()
env.reset()
env_ui = gameui.GameUI(env)
info = None
while env.done is False:
    if not info:
        print(info)
    action = env_ui.draw(info=info)

    if not action:
        print('no action')
    elif action < 0:
        print('quit game')
        break
    elif action in miningcoin.ACTION:
        print('action: {}'.format(action))
        _, _, _, info = env.step(action)
    else:
        env.reset()
        info = None
        env_ui.reset()
        print('reset the game')
env_ui.quit()

# Introduce

This is my AI course. I use DQN to train a agent to play Mining Coin Game designed by myself.

# The play method of the Game

- There is an agent who can move and mining.
- In the beginning, the agent has some `strength` and `coins`.
- Move needs ***1*** `strength`. Mining needs ***2*** `strength`.
  - The agent will get something after Mining.
  - Gain `coins`; Loss `coins`; Gain `strength`
- There are **FOUR** `regions` in the game.
    - `Region 1` $p_{gain coin}$ > $p_{loss coin}$, $p_{gain strength}==0$
    - `Region 2` $p_{gain coin}$ > $p_{loss coin}$ > $p_{gain strength}==0$
    - `Region 3` $p_{gain coin}$ > $p_{gain strength}$ > $p_{loss coin}$
    - `Region 4` $p_{gain strength}$ > $p_{loss coin}$ > $p_{gain coin}$
- If you move to the outside from `Region 1`, you will gain half of you current coins.

# How to Install

## Install

```bash
git clone https://github.com/MinzhiYoyo/AI_course_dqn_miningcoin.git
cd AI_course_dqn_miningcoin
# linus

pip install -r requirements.txt

# windows

pip install -r requirements.txt

```

After that, you need install `Pytorch` by yourself.

Click [here](https://pytorch.org/) to find the installation method.

# Start the game

## Play by yourself
    
```bash
# linux
python3 mining_coin_play_human.py

# windows
python mining_coin_play_human.py
```

How to play the game?

- Use `WSAD` to move the player.
- Use `Space` or `Enter` to mining.
- Use `R` to restart the game.
- If there is no response, please switch the English input method to lowercase.

## Train

```bash
# linux
python3 main.py

# windows
python main.py
```

## Play by agent

Begin of that, you need a model. You can [Train](#Train) a model of you before playing it by agent.

```bash
# linux
python3 mining_coin_play_dqn.py

# windows
python mining_coin_play_dqn.py
```


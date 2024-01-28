import json
import math
import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as torchF
import torch.optim as optim
import numpy as np
import random

from Game.miningcoin import MiningCoinEnv
import time


# Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


def get_time_info():
    time_info = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    return time_info


# 定义神经网络
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torchF.relu(self.fc1(x))
        x = torchF.relu(self.fc2(x))
        x = torchF.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 定义经验回放缓存
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=self.capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        self.memory.append((state, action, reward, next_state))

    # 保存所有经验到json文件
    def save(self, file_path):
        # 转化成json对象
        data = []
        for state, action, reward, next_state in self.memory:
            data.append({
                'state': state.tolist(),
                'action': action,
                'reward': reward,
                next_state: state.tolist(),
            })
        # 转成json文件
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def load(self, file_path):
        # 从json文件读取
        with open(file_path, 'r') as f:
            data = json.load(f)  # list
            # 遍历
            for item in data:
                state = np.array(item['state'])
                action = item['action']
                reward = item['reward']
                next_state = np.array(item['next_state'])
                self.push(state, action, reward, next_state)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        # batch是一个batch_size长度的列表，列表每个元素是一个长度为5的元组，现在需要将这个列表转换成长度为5的元组，每个元组是一个长度为batch_size的列表
        batch = list(zip(*batch))
        return batch

    def __len__(self):
        return len(self.memory)


# 定义DQN算法
class DQNAgent:
    def __init__(self, state_size, action_size, BATCH_SIZE=128, gamma=0.99, TAU=0.005, epsilon_start=0.9,
                 epsilon_end=0.05, epsilon_decay=1000, lr=1e-4, capacity=10000, model_path=None, model_dict_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Use {} to achieve DQNAgent'.format(self.device))
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        # self.epsilon = epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = BATCH_SIZE
        self.tau = TAU

        # 查看是否有模型输入路径
        if model_path:
            self.policy_net = torch.load(model_path, map_location=lambda storage, loc: storage).to(self.device)
            self.target_net = torch.load(model_path, map_location=lambda storage, loc: storage).to(self.device)
        else:
            self.policy_net = DQN(state_size, action_size).to(self.device)
            self.target_net = DQN(state_size, action_size).to(self.device)

        # 查看是否有模型参数输入路径
        if model_dict_path:
            self.policy_net.load_state_dict(torch.load(model_dict_path, map_location=lambda storage, loc: storage))

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.loss_f = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(capacity=capacity)

        self.steps = 0

        # self.criterion = nn.MSELoss()
        print('state size is {}, action size is {}'.format(self.state_size, self.action_size))

    def select_action(self, state):  # 传入torch tensor
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1. * self.steps / self.epsilon_decay)
        p_random = random.random()

        self.steps += 1

        if p_random > eps_threshold:
            with torch.no_grad():
                ret = self.policy_net(state)
                ret = ret.max(0)
                ret = ret.indices
                # ret = ret.view(1,1)
                # ret = self.policy_net(state).max(1)#.indices.view(1, 1)
                # ret = ret.indices.view(1,1)
                return ret
        else:
            return torch.tensor(np.random.choice(self.action_size), device=self.device, dtype=torch.long)

    def train(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return

        (state, action, reward, next_state) = self.memory.sample(batch_size)
        state = tuple(map(lambda x: torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0), state))
        action = tuple(
            map(lambda x: torch.tensor(x, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(1), action))
        reward = tuple(
            map(lambda x: torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(1), reward))
        next_state = tuple(
            map(lambda x: torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0), next_state))
        # done = torch.tensor(done, dtype=torch.bool, device=self.device).unsqueeze(1)

        # no_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)), device=self.device,
        #                              dtype=torch.bool)
        # no_final_next_states = torch.cat([s for s in next_state if s is not None]).to(self.device)

        state_batch = torch.cat(state).to(self.device)
        action_batch = torch.cat(action).to(self.device)
        reward_batch = torch.cat(reward).to(self.device)
        next_state_batch = torch.cat(next_state).to(self.device)

        # 计算Q(s_t, a) - Q(s_{t+1}, a)的值，然后我们选择最大值。
        q_eval = self.policy_net(state_batch)
        q_eval = q_eval.gather(1, action_batch)

        q_next = self.target_net(next_state_batch).detach()
        q_target = reward_batch + self.gamma * q_next.max(1)[0].view(batch_size, 1)

        loss = self.loss_f(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 计算所有next states的V(s_{t+1})的值
        # next_state_values = torch.zeros(batch_size, device=self.device)
        # with torch.no_grad():
        #     next_state_values[no_final_mask] = self.target_net(no_final_next_states).max(1).values
        #
        # # 计算期望的Q值
        # expected_state_action_values = (next_state_values * self.gamma) + reward_batch.squeeze(1)
        #
        # # 计算Huber loss
        # loss = self.loss_f(q_eval.squeeze(1), expected_state_action_values)
        #
        # # 开始优化模型
        # self.optimizer.zero_grad()
        # loss.backward()
        #
        # # 梯度裁剪
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        # self.optimizer.step()
        self.update_target_model()

    def update_target_model(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = self.tau * policy_net_state_dict[key] + (1 - self.tau) * target_net_state_dict[
                key]
        self.target_net.load_state_dict(target_net_state_dict)


# 如果model_dict_path是None，那么就是重新训练模型，否则是加载模型继续训练
def dqn_tain_model(model_path=None, model_dict_path=None, remark='', need_save=True, memory_file_path=None):
    with open('./experiment_statistics.log', 'r') as f:
        content = f.read()
        content = content.split('\n')
        lines = list(filter(''.__ne__, content))
        last_line = lines[-1]
        last_line_items = last_line.split(',')
        experiment_num = 1 if len(lines) == 1 else int(last_line_items[0]) + 1

    # 新建文件夹 ./log/experiment_num 和 ./model/experiment_num
    # 先检测文件夹是否存在，不在直接新建文件夹
    log_dir = './log/experiment_{}/'.format(experiment_num)
    model_dir = './model/experiment_{}/'.format(experiment_num)
    memory_dir = './model/memory/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(memory_dir):
        os.makedirs(memory_dir)

    if torch.cuda.is_available():
        print('Use GPU to train the model.')
        num_episodes = 1000  # 训练次数
    else:
        print('Use CPU to train the model.')
        num_episodes = 100

    # 使用示例
    env = MiningCoinEnv()
    state_size = env.observation_space_length  # 你的状态参数数量
    action_size = env.action_space_length  # 你的动作参数数量
    agent = DQNAgent(state_size, action_size, model_path=model_path, model_dict_path=model_dict_path)
    episode_reward = -math.inf
    best_dict_path = model_dir + '/model_dict_{}_best.pth'.format(get_time_info())
    best_net_state_dict = None

    episode_interval = 100

    output_mode_dict_path = None

    best_log = ''

    # 经验缓存的最大容量
    capacity = 10000

    # 训练之前先随机生成经验
    print('随机生成经验中...')
    if memory_file_path:
        agent.memory.load(memory_file_path)
    else:
        while len(agent.memory) < capacity:
            memory_env = MiningCoinEnv()
            state = memory_env.reset()
            while not memory_env.done:
                action = random.randint(0, action_size - 1)
                next_state, reward, done, info = memory_env.step(action)
                agent.memory.push(state, action, reward, next_state)
                state = next_state
                if memory_env.done:
                    break

    # 在训练循环中使用agent进行动作选择、存储经验和训练
    for episode in range(num_episodes):
        log_file = None
        once_game_log = ''
        if episode % episode_interval == 1:
            output_log_path = log_dir + 'log_{}_{}.log'.format(get_time_info(), episode)
            if need_save:
                log_file = open(output_log_path, 'a')
        state = env.reset()
        while not env.done:
            action = agent.select_action(torch.tensor(state, dtype=torch.float32, device=agent.device)).item()
            next_state, reward, done, info = env.step(action)
            once_game_log += info
            once_game_log += '\n'

            agent.memory.push(state, action, reward, next_state)
            state = next_state
            agent.train()
            if env.done:
                if env.current_coins > episode_reward:
                    episode_reward = env.current_coins
                    best_net_state_dict = agent.target_net.state_dict().copy()
                    best_log = once_game_log
                break

        if log_file:
            log_file.write(once_game_log)
            log_file.close()
        # 保存模型参数
        if episode % episode_interval == 1:
            print('Episode: {}, Reward: {}'.format(episode, env.current_coins))
            output_mode_dict_path = model_dir + 'model_dict_{}_{}.pth'.format(get_time_info(), episode)
            if need_save:
                torch.save(agent.policy_net.state_dict(), output_mode_dict_path)

    # 保存最好的参数
    print('保存最好的参数中...')
    if need_save:
        torch.save(best_net_state_dict, best_dict_path)

    # 保存整个模型
    print('保存模型中...')
    output_mode_path = model_dir + 'model_{}.pth'.format(get_time_info())
    if need_save:
        torch.save(agent.target_net, output_mode_path)

    info = '{},{},{},{},{},{},{},{},{},{}\n'.format(experiment_num, episode_reward, str(model_path),
                                                    str(model_dict_path), episode_interval,
                                                    num_episodes, log_dir, model_dir, remark, env.game_setting_info)
    print(info)
    if need_save:
        with open('./experiment_statistics.log', 'a') as f:
            f.write(info)
        with open(log_dir + 'log_{}_best.log'.format(get_time_info()), 'w') as f:
            f.write(best_log)
        agent.memory.save(memory_dir + 'memory_{}_{}.json'.format(experiment_num, get_time_info()))

    return experiment_num, output_mode_path, output_mode_dict_path, best_dict_path if best_net_state_dict else None


if __name__ == '__main__':
    dqn_tain_model(need_save=False)

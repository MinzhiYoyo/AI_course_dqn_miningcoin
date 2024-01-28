import gym
import numpy as np

DIG = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
ACTION = [DIG, UP, DOWN, LEFT, RIGHT]  # 分为挖上下左右
ACTION_NAME = ['dig', 'up', 'down', 'left', 'right']
ACTION_DIRECTION = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]


class MiningCoinEnv(gym.Env):
    # 将常量都作为参数传入
    def __init__(self,
                 map_grid=((-5, -5), (5, 5)),  # 地图大小
                 start_position=(0, 0),  # 起始位置
                 Strength=60,  # 总行动力
                 Strength_move=1,  # 移动一步消耗的行动力
                 Strength_mining=2,  # 挖掘一次消耗的行动力

                 gain1=10,  # 区域1挖掘获得的金币最大值
                 gain2=15,  # 区域2挖掘获得的金币最大值
                 gain3=20,  # 区域3挖掘获得的金币最大值
                 gain4=10,  # 区域4挖掘获得的金币最大值
                 gaind=8,  # 距离收益的最大值

                 loss1=10,  # 区域1挖掘损失的金币最大值
                 loss2=10,  # 区域2挖掘损失的金币最大值
                 loss3=5,  # 区域3挖掘损失的金币最大值
                 loss4=5,  # 区域4挖掘损失的金币最大值

                 # 定义概率
                 p1=0.5,  # 区域1的挖掘获得金币的概率
                 q1=0.25,  # 区域1的挖掘损失金币的概率
                 s1=0,  # 区域1的挖掘获得行动力的概率

                 p2=0.6,  # 区域2的挖掘获得金币的概率
                 q2=0.25,  # 区域2的挖掘损失金币的概率
                 s2=0.1,  # 区域2的挖掘获得行动力的概率

                 p3=0.6,  # 区域3的挖掘获得金币的概率
                 q3=0.1,  # 区域3的挖掘损失金币的概率
                 s3=0.3,  # 区域3的挖掘获得行动力的概率

                 p4=0.6,  # 区域4的挖掘获得金币的概率
                 q4=0.25,  # 区域4的挖掘损失金币的概率
                 s4=0.1,  # 区域4的挖掘获得行动力的概率

                 pd=0.75,  # 距离收益的概率
                 strength_gain=5,  # 挖掘获得的行动力
                 start_coins=30,  # 开始有30块钱

                 ):
        super(MiningCoinEnv, self).__init__()

        # 游戏中的一些常量参数
        self.map_grid = map_grid  # 地图 坐下和右上的坐标
        self.start_position = start_position  # 开始的位置
        self.Strength = Strength  # 总共行动力
        self.Strength_move = Strength_move  # 移动消耗的行动力
        self.Strength_mining = Strength_mining  # 挖掘消耗的移动力
        self.start_coins = start_coins
        self.gain_loss = [[0, 0], [gain1, loss1], [gain2, loss2], [gain3, loss3], [gain4, loss4]]
        self.gaind = gaind
        self.strength_gain = strength_gain
        self.probability = [[0, 0, 0], [p1, q1, s1], [p2, q2, s2], [p3, q3, s3], [p4, q4, s4]]
        self.pd = pd

        self.map_flag = None
        self.done = None
        self.current_position = None
        self.current_strength = None
        self.current_coins = None

        # 定义行动空间，一维离散：0=dig, 1=up, 2=down, 3=left, 4=right
        self.action_space_length = len(ACTION)
        # self.action_space = spaces.Discrete(self.action_space_length)

        # 定义观测空间，一维离散，来自一个numpy
        start_state = self.reset()
        self.observation_space_length = len(start_state)
        # self.observation_space = spaces.Tuple(spaces=(
        #
        # ))

        # 保存所有游戏常量参数，以下划线分割
        self.game_setting_info = ('map:(({}_{})({}_{}))|start_position:({}_{})|Strength:{}|Strength_move:{'
                                  '}|Strength_mining:{}|gain_loss_strength_1234:({}_{})({}_{})({}_{})({}_{})|gaind:{'
                                  '}|strength_gain:{}|probability_pqs1234:({}_{}_{})({}_{}_{})({}_{}_{})({}_{}_{})|pd:{'
                                  '}|start_coins:{}').format(
            self.map_grid[0][0], self.map_grid[0][1], self.map_grid[1][0], self.map_grid[1][1],
            self.start_position[0], self.start_position[1],
            self.Strength, self.Strength_move, self.Strength_mining,
            self.gain_loss[1][0], self.gain_loss[1][1], self.gain_loss[2][0], self.gain_loss[2][1],
            self.gain_loss[3][0], self.gain_loss[3][1], self.gain_loss[4][0], self.gain_loss[4][1],
            self.gaind, self.strength_gain,
            self.probability[1][0], self.probability[1][1], self.probability[1][2],
            self.probability[2][0], self.probability[2][1], self.probability[2][2],
            self.probability[3][0], self.probability[3][1], self.probability[3][2],
            self.probability[4][0], self.probability[4][1], self.probability[4][2],
            self.pd,
            self.start_coins
        )

    def reset(self):
        self.current_strength = self.Strength
        self.current_position = self.start_position
        self.current_coins = self.start_coins
        self.done = False
        # 定义11x11的false矩阵，表示地图是否被挖掘
        self.map_flag = np.zeros((11, 11), dtype=bool)
        # 获取状态
        current_state = self._get_state()
        # self.state_length = len(current_state)
        return current_state

    # 获取状态，总共九个点，每个点有三个概率值pqs、获得金币的增益减益最大值gain,loss，除此之外，还有当前坐标，行动力，以及当前金币数，共计9*5+3=48个状态
    def _get_state(self, position=None):
        if not position:
            position = self.current_position
        # 计算当前位置的周围八个点
        position_left_up = (position[0] - 1, position[1] + 1)
        position_up = (position[0], position[1] + 1)
        position_right_up = (position[0] + 1, position[1] + 1)
        position_left = (position[0] - 1, position[1])
        position_right = (position[0] + 1, position[1])
        position_left_down = (position[0] - 1, position[1] - 1)
        position_down = (position[0], position[1] - 1)
        position_right_down = (position[0] + 1, position[1] - 1)

        # 从左上，上，右上，左，当前，右，左下，下，右下的顺序，依次打包状态成np.array
        positions = [position_left_up, position_up, position_right_up, position_left, position, position_right,
                     position_left_down, position_down, position_right_down]

        state = []
        for pos in positions:
            # 获取当前位置的区域编号
            region_num = self.get_region_num(pos)
            # 获取当前位置的
            state += [self.probability[region_num][0], self.probability[region_num][1], self.probability[region_num][2],
                      self.gain_loss[region_num][0], self.gain_loss[region_num][1]]

        # 是否需要在状态中加入后面的金币参数，有待考量
        state += [position[0], position[1], self.current_strength]  # , self.current_coins]

        return np.array(
            state
            , dtype=np.float32)

    # 输入action（0,1,2,3,4），输出state，reward，done，info
    def step(self, action):
        cur_posx, cur_posy = self.current_position[0], self.current_position[1]
        gain_d = 0
        gains = 0
        if not self.done:  # 如果游戏还没有停止
            # 如果可以挖掘，且行动力足够
            if action == DIG:
                if self.current_strength >= self.Strength_mining:
                    # 距离收益只要挖了就有

                    # 计算距离收益
                    gain_d = self._position_gain(self.current_position)
                    self.current_coins += gain_d
                    self.current_strength -= self.Strength_mining  # 减少行动力

                    if not self.map_flag[self.current_position[0]][self.current_position[1]]:

                        # 计算金币增益减益以及行动力增益
                        self.map_flag[self.current_position[0]][self.current_position[1]] = True

                        # 获取区域
                        region_num = self.get_region_num(self.current_position)
                        # 轮赌选择
                        type_gainloss, normal_p = MiningCoinEnv._roulette(self.probability[region_num][0],
                                                                          self.probability[region_num][1],
                                                                          self.probability[region_num][2])

                        # 如果是增益金币
                        if type_gainloss == 1:
                            gains = self.gain_loss[region_num][0] * normal_p
                            self.current_coins += gains
                        # 如果是减益金币
                        elif type_gainloss == 2:
                            gains = -self.gain_loss[region_num][1] * normal_p
                            self.current_coins += gains
                        # 如果是增益行动力
                        elif type_gainloss == 3:
                            gains = self.strength_gain
                            self.current_strength += gains
            elif action in ACTION:
                direction = ACTION_DIRECTION[action]
                target_postion = (self.current_position[0] + direction[0], self.current_position[1] + direction[1])
                if self.get_region_num(target_postion) > 0:
                    # 有效，直接移动
                    self.current_position = target_postion
                    self.current_strength -= self.Strength_move  # 消耗体力
                elif self.get_region_num(target_postion) == 0:
                    if (target_postion[0] > 0 and target_postion[1] > 5) or (
                            target_postion[0] > 5 and target_postion[1] > 0):
                        # 也有效，直接移动，清空体力
                        self.current_strength = 0  # 体力消耗完了
                        self.current_coins *= 1.5  # 金币翻1.5倍
                        self.current_position = target_postion  # 移动位置

        self.done = True if self.current_strength == 0 else False
        # 输出info
        info = '({},{})->({},{}),action={},gains={},gains_dis={},strength={},coins={},done={}'.format(
            cur_posx, cur_posy, self.current_position[0], self.current_position[1],
            ACTION_NAME[action], gains, gain_d, self.current_strength, self.current_coins,
            str(self.done)
        )
        return self._get_state(), self.current_coins, self.done, info

    def _get_dig_reward(self):
        x, y = self.current_position
        if x >= 0 and y >= 0:  # 第一象限
            return self._get_reward_quadrant(0.5, 0.25, 0, (-5, 5))
        elif x < 0 and y >= 0:  # 第二象限
            return self._get_reward_quadrant(0.4, 0.3, 0.2, (-10, 10))
        elif x < 0 and y < 0:  # 第三象限
            return self._get_reward_quadrant(0.2, 0.3, 0.4, (-10, 15))
        elif x >= 0 and y < 0:  # 第四象限
            return self._get_reward_quadrant(0.35, 0.35, 0.2, (-10, 20))
        else:
            return 0

    def _get_reward_quadrant(self, gain_prob, loss_prob, actions_prob, reward_range):
        if np.random.rand() < gain_prob:
            return np.random.uniform(*reward_range)
        elif np.random.rand() < gain_prob + loss_prob:
            return -np.random.uniform(*reward_range)
        elif np.random.rand() < gain_prob + loss_prob + actions_prob:
            return 0
        else:
            return 5

    def get_region_num(self, position):
        # 原点，x正轴，以及第一象限返回1: (0<x<=5, 0<=y<=5) or (x == 0 and y == 0)
        if (0 < position[0] <= self.map_grid[1][0] and 0 <= position[1] <= self.map_grid[1][1]) or (
                position[0] == 0 and position[
            1] == 0):
            return 1
        # y正轴，第二象限返回2: -5<=x<=0, 0<y<=5
        elif self.map_grid[0][0] <= position[0] <= 0 < position[1] <= self.map_grid[1][1]:
            return 2
        # x负轴，第三象限返回3: -5<=x<0, -5<=y<=0
        elif 0 > position[0] >= self.map_grid[0][0] and 0 >= position[1] >= self.map_grid[0][1]:
            return 3
        # y负轴，第四象限返回4: 0<=x<=5, -5<=y<0
        elif self.map_grid[1][0] >= position[0] >= 0 > position[1] >= self.map_grid[0][1]:
            return 4
        else:
            return 0

    # 定义距离收益
    def _position_gain(self, position):
        # 获取当前位置到最左下角点的距离
        distance = position[0] - self.map_grid[0][0] + position[1] - self.map_grid[0][1]
        max_distance = self.map_grid[1][0] - self.map_grid[0][0] + self.map_grid[1][1] - self.map_grid[0][1]
        # 获取一个0-1的伪随机数
        pnum = np.random.rand()
        if pnum < self.pd:
            return self.gaind * pnum * distance / max_distance / self.pd
        else:
            return 0

    # 定义轮盘选择函数，返回选择的区域以及归一化后的概率
    @staticmethod
    def _roulette(p, q, s):
        # 获取一个0-1的伪随机数
        pnum = np.random.rand()
        if pnum < p:
            return 1, pnum / p
        elif pnum < p + q:
            return 2, (pnum - p) / q
        elif pnum < p + q + s:
            return 3, 0
        else:
            return 4, 0

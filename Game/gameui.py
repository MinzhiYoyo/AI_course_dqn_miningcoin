import pygame
from Game.miningcoin import MiningCoinEnv, ACTION, ACTION_NAME


# 游戏主体大小及位置：x = 20, y = 20, w = 40*11, h = 40*11
# 游戏需要的按键有上下左右和空格，上下作业也可以是wsad来替代，空格可以是回车替代，两种操作方案
# 游戏主体右边40像素有游戏玩法说明及分数统计，称为游戏信息栏
# 游戏信息栏宽度和游戏主体宽度一样，40*11，高度也和游戏主体高度一样
# 因此，计算窗口总大小时，宽度为40*12*2，高度为40*12

class GameUI:
    def __init__(self,
                 env: MiningCoinEnv,
                 w_unit=40,  # 游戏主体单元格宽
                 h_unit=40,  # 游戏主体单元格高
                 interval_size=20,  # 间距大小
                 color_invisible=(200, 200, 200),  # 不可见单元格颜色
                 color_mining=(255, 0, 255),  # 被开采过的单元格颜色
                 color_region_1=(100, 0, 0),  # 区域1颜色
                 color_region_2=(0, 255, 0),  # 区域2颜色
                 color_region_3=(0, 0, 255),  # 区域3颜色
                 color_region_4=(255, 255, 0),  # 区域4颜色
                 game_setting_info: str = None,  # 游戏设置信息
                 ):

        self.env = env
        self.screen = None
        self.data = {
            'map': None,
            'start position': None,
            'Strength': None,
            'Strength move': None,
            'Strength mining': None,
            'gain loss strength 1234': None,
            'gaind': None,
            'strength gain': None,
            'probability pqs1234': None,
            'pd': None,
            'start coins': None
        }
        if env:
            self.data['map'] = env.map_grid
            self.data['start position'] = env.start_position
            self.data['Strength'] = env.Strength
            self.data['Strength move'] = env.Strength_move
            self.data['Strength mining'] = env.Strength_mining
            self.data['gain loss strength 1234'] = env.gain_loss
            self.data['gaind'] = env.gaind
            self.data['strength gain'] = env.strength_gain
            self.data['probability pqs1234'] = env.probability
            self.data['pd'] = env.pd
            self.data['start coins'] = env.start_coins
        elif game_setting_info:
            try:
                game_setting_info_list = game_setting_info.split('|')
                map_info_list = game_setting_info_list[0].split('_')  # ['map:(({}', '{})({}' ,'{}))']
                left_down_x = int(map_info_list[0][6:])
                left_down_y = int(map_info_list[1].split(')(')[0])
                right_up_x = int(map_info_list[1].split(')(')[-1])
                right_up_y = int(map_info_list[2][:-2])
                self.data['map'] = ((left_down_x, left_down_y), (right_up_x, right_up_y))

                start_position_info_list = game_setting_info_list[1].split('_')  # ['start_position:({}', '{})']
                start_position_x = int(start_position_info_list[0][16:])
                start_position_y = int(start_position_info_list[1][:-1])
                self.data['start position'] = (start_position_x, start_position_y)

                self.data['Strength'] = int(game_setting_info_list[2].split(':')[1])
                self.data['Strength move'] = int(game_setting_info_list[3].split(':')[1])
                self.data['Strength mining'] = int(game_setting_info_list[4].split(':')[1])

                gain_loss_strength_1234_info_list = game_setting_info_list[5].split(':')[1].split(
                    ')(')  # ['({}_{}', '{}_{}', '{}_{}', '{}_{})']
                gain_loss_strength_1_info_list = gain_loss_strength_1234_info_list[0].split('_')  # ['({}', '{}']
                gain_loss_strength_2_info_list = gain_loss_strength_1234_info_list[1].split('_')  # ['{}', '{}']
                gain_loss_strength_3_info_list = gain_loss_strength_1234_info_list[2].split('_')  # ['{}', '{}']
                gain_loss_strength_4_info_list = gain_loss_strength_1234_info_list[3].split('_')  # ['{}', '{})']
                self.data['gain loss strength 1234'] = [(0, 0), (
                    int(gain_loss_strength_1_info_list[0][1:]),
                    int(gain_loss_strength_1_info_list[1])
                ), (
                                                            int(gain_loss_strength_2_info_list[0]),
                                                            int(gain_loss_strength_2_info_list[1])
                                                        ), (
                                                            int(gain_loss_strength_3_info_list[0]),
                                                            int(gain_loss_strength_3_info_list[1])
                                                        ), (
                                                            int(gain_loss_strength_4_info_list[0]),
                                                            int(gain_loss_strength_4_info_list[1][:-1])
                                                        )]

                self.data['gaind'] = float(game_setting_info_list[6].split(':')[1])
                self.data['strength gain'] = int(game_setting_info_list[7].split(':')[1])

                probability_pqs1234_info_list = game_setting_info_list[8].split(':')[1].split(
                    ')(')  # ['({}_{}_{}', '{}_{}_{}', '{}_{}_{}', '{}_{}_{})']
                probability_pqs_1_info_list = probability_pqs1234_info_list[0].split('_')  # ['({}', '{}', '{}']
                probability_pqs_2_info_list = probability_pqs1234_info_list[1].split('_')  # ['{}', '{}', '{}']
                probability_pqs_3_info_list = probability_pqs1234_info_list[2].split('_')  # ['{}', '{}', '{}']
                probability_pqs_4_info_list = probability_pqs1234_info_list[3].split('_')  # ['{}', '{}', '{})']

                self.data['probability pqs1234'] = [(0, 0, 0), (
                    float(probability_pqs_1_info_list[0][1:]),
                    float(probability_pqs_1_info_list[1]),
                    float(probability_pqs_1_info_list[2])
                ), (
                                                        float(probability_pqs_2_info_list[0]),
                                                        float(probability_pqs_2_info_list[1]),
                                                        float(probability_pqs_2_info_list[2])
                                                    ), (
                                                        float(probability_pqs_3_info_list[0]),
                                                        float(probability_pqs_3_info_list[1]),
                                                        float(probability_pqs_3_info_list[2])
                                                    ), (
                                                        float(probability_pqs_4_info_list[0]),
                                                        float(probability_pqs_4_info_list[1]),
                                                        float(probability_pqs_4_info_list[2][:-1])
                                                    )]

                self.data['pd'] = float(game_setting_info_list[9].split(':')[1])
                self.data['start coins'] = float(game_setting_info_list[10].split(':')[1])
            except Exception as e:
                print('Parse Error Game setting info')
                print(self.data)
                print(e)
                raise e

        # 总格子有
        w_num = self.data['map'][1][1] - self.data['map'][0][1] + 1
        h_num = self.data['map'][1][0] - self.data['map'][0][0] + 1
        self.w_num = w_num
        self.h_num = h_num

        # 间隔
        self.interval_size = interval_size

        self.window_size = (w_unit * w_num * 2 + self.interval_size * 4, h_unit * h_num + self.interval_size * 2)

        # 游戏主体大小及位置
        self.game_body_size = (w_unit * w_num, h_unit * h_num)
        self.game_body_pos = (self.interval_size, self.interval_size)

        # 游戏信息栏大小及位置
        self.game_info_size = (w_unit * w_num, h_unit * h_num)
        self.game_info_pos = (self.interval_size * 2 + w_unit * w_num, self.interval_size)

        # 不可见单元格颜色为灰色
        self.color_invisible = color_invisible

        # 被开采过的单元格颜色为粉红色
        self.color_mining = color_mining

        # 区域颜色
        self.color_region = [
            (0, 0, 0),
            color_region_1,
            color_region_2,
            color_region_3,
            color_region_4
        ]

        # 游戏主体单元格大小
        self.w_unit = w_unit
        self.h_unit = h_unit

        # 初始化窗口
        self._init_window()

    def reset(self):
        # 初始化地图状态
        self.map_mining_flag = [[False for _ in range(self.w_num)] for _ in range(self.h_num)]
        self.screen.fill((255, 255, 255))
        # 绘制游戏主体
        # 初试位置在(0,0)
        visible_list = [
            (-1, 1), (0, 1), (1, 1),
            (-1, 0), (0, 0), (1, 0),
            (-1, -1), (0, -1), (1, -1)
        ]
        for x in range(self.data['map'][0][0], self.data['map'][1][0] + 1):
            for y in range(self.data['map'][0][1], self.data['map'][1][1] + 1):
                color = self.color_invisible if (x, y) not in visible_list else self.color_region[
                    self.env.get_region_num((x, y))]
                # color = self.color_region[self.env.get_region_num((x, y))]
                self._draw_unit(color, x, y)

        self._draw_body_grid()
        self._draw_information_bar(self.data['start coins'], self.data['Strength'], 0, 0)
        # 更新显示
        pygame.display.flip()

    # 初始化窗口
    def _init_window(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('Mining Coins Game')
        self.reset()



    # 传入的x,y是游戏主体的坐标，需要转换成窗口坐标
    def _draw_unit(self, color, x, y):
        
        # 将输入坐标系转化为pygame坐标系
        wx = (x - self.data['map'][0][0])  # 先计算在游戏主体中的坐标x
        wy = (-y - self.data['map'][0][1])  # 先计算在游戏主体中的坐标y

        # 转化为像素点，相对坐标
        wx = wx * self.w_unit
        wy = wy * self.h_unit

        # 计算在游戏窗口中的坐标，绝对坐标
        wx = wx + self.game_body_pos[0]
        wy = wy + self.game_body_pos[1]

        # 绘制
        pygame.draw.rect(self.screen, color, (wx, wy, self.w_unit, self.h_unit))

    # 绘制游戏信息栏
    def _draw_information_bar(self, coins, strength, dis_coins=0, gain_coins=0):
        game_play_info = """
        金币：{}，行动力：{}，金币收益：{}，距离收益：{}
        
        游戏玩法：玩家可以通过上下左右键（或wsad）来移动，空格键（或回车）来挖矿
        移动与挖矿均会消耗行动力，移动消耗：{}，挖矿消耗：{}
        每次挖矿会获得增益金币，减益金币，增益行动力，无任何收益，均有其特定概率
        游戏分为4个区域/象限
            区域/象限1概率：增益金币：{}，减益金币：{}，增益行动力：{}；最大增益金币为：{}，最大减益金币为：{}；
            区域/象限2概率：增益金币：{}，减益金币：{}，增益行动力：{}；最大增益金币为：{}，最大减益金币为：{}；
            区域/象限3概率：增益金币：{}，减益金币：{}，增益行动力：{}；最大增益金币为：{}，最大减益金币为：{}；
            区域/象限4概率：增益金币：{}，减益金币：{}，增益行动力：{}；最大增益金币为：{}，最大减益金币为：{}；
        
        此外每次挖掘，无论是否被挖都会获得距离收益，概率为：{}，最大距离收益为：{}，越往右上方收益越大。
        
        如果从第一象限出去，则游戏结束，并且金币翻倍。注，不是区域/象限1，而是第一象限
        如果行动力用完，那么游戏也结束
        """.format(coins, strength, gain_coins, dis_coins,
                   self.data['Strength move'], self.data['Strength mining'],
                   self.data['probability pqs1234'][1][0], self.data['probability pqs1234'][1][1],
                   self.data['probability pqs1234'][1][2], self.data['gain loss strength 1234'][1][0],
                   self.data['gain loss strength 1234'][1][1],
                   self.data['probability pqs1234'][2][0], self.data['probability pqs1234'][2][1],
                   self.data['probability pqs1234'][2][2], self.data['gain loss strength 1234'][2][0],
                   self.data['gain loss strength 1234'][2][1],
                   self.data['probability pqs1234'][3][0], self.data['probability pqs1234'][3][1],
                   self.data['probability pqs1234'][3][2], self.data['gain loss strength 1234'][3][0],
                   self.data['gain loss strength 1234'][3][1],
                   self.data['probability pqs1234'][4][0], self.data['probability pqs1234'][4][1],
                   self.data['probability pqs1234'][4][2], self.data['gain loss strength 1234'][4][0],
                   self.data['gain loss strength 1234'][4][1],
                   self.data['pd'], self.data['gaind']
                   )
        game_play_info = 'strength = {}, coins = {:2f} + {:2f} + {:2f}'.format(strength, coins, gain_coins, dis_coins)

        self.screen.fill((255, 255, 255), (self.game_info_pos[0], self.game_info_pos[1], self.game_info_size[0],
                                            self.game_info_size[1]))

        # 绘制游戏信息栏
        font = pygame.font.Font(None, 20)
        text = font.render(game_play_info, True, (0, 0, 0))
        self.screen.blit(text, (self.game_info_pos[0], self.game_info_pos[1]))

    # 绘制游戏主体的边框网格，用虚线
    def _draw_body_grid(self):
        # map[1][0]- map[0][0]+2条竖线，map[1][1]-map[0][1] + 2条横线
        # 绘制竖线，虚线
        for i in range(self.data['map'][1][0] - self.data['map'][0][0] + 2):
            pygame.draw.line(self.screen, (0, 0, 0),
                             (self.game_body_pos[0] + i * self.w_unit, self.game_body_pos[1]),
                             (self.game_body_pos[0] + i * self.w_unit, self.game_body_pos[1] + self.game_body_size[1]),
                             1)

        # 绘制横线，虚线
        for i in range(self.data['map'][1][1] - self.data['map'][0][1] + 2):
            pygame.draw.line(self.screen, (0, 0, 0),
                             (self.game_body_pos[0], self.game_body_pos[1] + i * self.h_unit),
                             (self.game_body_pos[0] + self.game_body_size[0], self.game_body_pos[1] + i * self.h_unit),
                             1)

    # 挖掘动画，需要颜色变化
    def _mining_animation(self, x, y):
        # 直接更新颜色即可，还要更新map_mining_flag
        self.map_mining_flag[x][y] = True
        color = self.color_mining
        self._draw_unit(color, x, y)

    # 移动动画，只需要变化一下颜色就行了
    def _move_animation(self, sx, sy, tx, ty):
        # 需要不可见的单元格列表
        invisible_list = [
            (sx - 1, sy + 1), (sx, sy + 1), (sx + 1, sy + 1),
            (sx - 1, sy), (sx, sy), (sx + 1, sy),
            (sx - 1, sy - 1), (sx, sy - 1), (sx + 1, sy - 1)
        ]

        # 需要可见的单元格列表
        visible_list = [
            (tx - 1, ty + 1), (tx, ty + 1), (tx + 1, ty + 1),
            (tx - 1, ty), (tx, ty), (tx + 1, ty),
            (tx - 1, ty - 1), (tx, ty - 1), (tx + 1, ty - 1)
        ]

        # 绘制不可见的单元格
        for x, y in invisible_list:
            if self.data['map'][0][0] <= x <= self.data['map'][1][0] and self.data['map'][0][1] <= y <= \
                    self.data['map'][1][1]:
                self._draw_unit(self.color_invisible, x, y)

        # 绘制可见的单元格
        for x, y in visible_list:
            if self.data['map'][0][0] <= x <= self.data['map'][1][0] and self.data['map'][0][1] <= y <= \
                    self.data['map'][1][1]:
                color = self.color_mining if self.map_mining_flag[x][y] else self.color_region[
                    self.env.get_region_num((x, y))]
                self._draw_unit(color, x, y)

    def draw(self, info: str = None, game_setting_info: str = None, mode='human'):
        if not info:
            # self._draw_information_bar(self.data['start coins'], self.data['Strength'], 0, 0)
            # pygame.display.flip()
            return self._check_input(mode)
        # 提取info数据：'({},{})->({},{}),action={},gains={},gains_dis={},strength={},coins={},done={}'
        try:
            info_list = info.split(',')
            sx = int(info_list[0][1:])
            sy = int(info_list[1].split(')->(')[0])
            tx = int(info_list[1].split(')->(')[-1])
            ty = int(info_list[2][:-1])

            data = {
                'action': info_list[3].split('=')[-1],
                'gains': float(info_list[4].split('=')[-1]),
                'gains_dis': float(info_list[5].split('=')[-1]),
                'strength': int(info_list[6].split('=')[-1]),
                'coins': float(info_list[7].split('=')[-1]),
                'done': bool(info_list[8].split('=')[-1])
            }
        except Exception as e:
            # 打印解析信息出错，用英文打印
            print('Parse Error: ')
            print(e)
            raise e

        if data['action'] == ACTION_NAME[0]:
            self._mining_animation(tx, ty)
        elif data['action'] in ACTION_NAME[1:]:
            self._move_animation(sx, sy, tx, ty)

        # 绘制游戏信息栏
        self._draw_information_bar(data['coins'], data['strength'], data['gains_dis'], data['gains'])
        self._draw_body_grid()
        pygame.display.flip()
        return self._check_input(mode)

    # 检测输入
    def _check_input(self, mode='human'):
        if mode == 'human':
            # for event in pygame.event.wait():
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                return -1
            elif event.type == pygame.KEYDOWN:
                # 如果按下的是空格键，或者回车键
                if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                    return ACTION[0]
                # 如果按下的是上下左右键，或者wsad
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    return ACTION[1]
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    return ACTION[2]
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    return ACTION[3]
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    return ACTION[4]
                elif event.key == pygame.K_r:
                    return 100
            return None

    def quit(self):
        pygame.quit()
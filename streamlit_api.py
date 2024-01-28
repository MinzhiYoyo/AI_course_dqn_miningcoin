import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
import time
import random
import os
import numpy as np

from Game import miningcoin
from streamlit_extras.grid import grid
DIG = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
ACTION = [DIG, UP, DOWN, LEFT, RIGHT]  # 分为挖上下左右

# -------------- app config ---------------
env = miningcoin.MiningCoinEnv()
env.reset()
info = None

st.set_page_config(page_title="Mining Coin Game", page_icon="💰")

def btn_click(btn_text):
    st.toast(f"Clicked {btn_text}!", icon="⛏")
    position_list = list(st.session_state.current_position)
    if btn_text == "up":
        position_list[1] += 1
        st.session_state.view, st.session_state.current_position, st.session_state.current_strength, st.session_state.current_coins, st.session_state.done = env.step(ACTION[1])
    elif btn_text == "left":
        position_list[0] -= 1
        st.session_state.view, st.session_state.current_position, st.session_state.current_strength, st.session_state.current_coins, st.session_state.done = env.step(ACTION[3])
    elif btn_text == "right":
        position_list[0] += 1
        st.session_state.view, st.session_state.current_position, st.session_state.current_strength, st.session_state.current_coins, st.session_state.done = env.step(ACTION[4])
    elif btn_text == "down":
        position_list[1] -= 1
        st.session_state.view, st.session_state.current_position, st.session_state.current_strength, st.session_state.current_coins, st.session_state.done = env.step(ACTION[2])
    elif btn_text == "dig":
        st.session_state.view, st.session_state.current_position, st.session_state.current_strength, st.session_state.current_coins, st.session_state.done = env.step(ACTION[0])
    st.session_state.current_position = tuple(position_list)
    print(st.session_state.view)
    env.current_position = st.session_state.current_position
    env.current_strength = st.session_state.current_strength
    env.current_coins = st.session_state.current_coins
    btn_styles = np.full((11, 11), "⬜", dtype=str)
    btn_styles[st.session_state.current_position[0]+4][st.session_state.current_position[0]+5] = colors[st.session_state.view[1]]
    btn_styles[st.session_state.current_position[0]+4][st.session_state.current_position[0]+4] = colors[st.session_state.view[0]]
    btn_styles[st.session_state.current_position[0]+4][st.session_state.current_position[0]+6] = colors[st.session_state.view[2]]
    btn_styles[st.session_state.current_position[0]+5][st.session_state.current_position[0]+4] = colors[st.session_state.view[3]]
    btn_styles[st.session_state.current_position[0]+5][st.session_state.current_position[0]+5] = colors[0]
    btn_styles[st.session_state.current_position[0]+5][st.session_state.current_position[0]+6] = colors[st.session_state.view[5]]
    btn_styles[st.session_state.current_position[0]+6][st.session_state.current_position[0]+4] = colors[st.session_state.view[6]]
    btn_styles[st.session_state.current_position[0]+6][st.session_state.current_position[0]+5] = colors[st.session_state.view[7]]
    btn_styles[st.session_state.current_position[0]+6][st.session_state.current_position[0]+6] = colors[st.session_state.view[8]]
    print(st.session_state.current_position)
    print(btn_styles)


###############################################
#
#           VARIABLES DEFINITION
#
################################################

# variable responsible for checking if player provided his name and game can be started
start = False

# set session states
# this is streamlit specific. For more contex please check streamlit documenation

if "view" not in st.session_state:
    st.session_state["view"] = [1, 1, 0, 2, 0, 0, 2, 3, 3]  # 从左到右，从上到下。1红色：第一象限，2黄色：第二象限，3绿色：第三象限，4蓝色：第四象限
if "done" not in st.session_state:
    st.session_state["done"] = env.done
if "current_position" not in st.session_state:
    st.session_state["current_position"] = env.current_position
if "current_strength" not in st.session_state:
    st.session_state["current_strength"] = env.current_strength
if "current_coins" not in st.session_state:
    st.session_state["current_coins"] = env.current_coins


###############################################
#
#               GAME ENGINE
#
################################################

# ---------------- CSS ----------------

# local_css("style.css")

# ----------------- game start --------


welcome = st.empty()
welcome.title("Welcome to Mining Coins!")

player_name_container = st.empty()
player_name_container.text_input("Input your name and type ENTER", key="player_name")
main_text_container = st.empty()

if st.session_state.player_name != "":
    player_name_container.empty()
    main_text_container.empty()
    start = True

# START THE GAME

if start:
    # delete welcome
    welcome.empty()

    # grid
    map_pos = np.zeros((11, 11), dtype=bool)
    map_pos[5, 5] = True
    colors = ["🤪", "🟧", "🟨", "🟩", "🟦", "⬜"]
    btn_styles = np.full((11, 11), "⬜", dtype=str)
    with st.container():
        forest = grid(11, 11, vertical_align="center")
        for j in range(10, -1, -1): # j是纵坐标，从上到下是10到0，表示5到-5，真实坐标是（i-5，j-5）
            for i in range(11): # i是横坐标，从左到右是0到10，表示-5到5
                forest.button(btn_styles[i][j], key=f"forest_{i}_{j}")

        btn_styles[st.session_state.current_position[0]+4][st.session_state.current_position[0]+5] = colors[st.session_state.view[1]]
        btn_styles[st.session_state.current_position[0]+4][st.session_state.current_position[0]+4] = colors[st.session_state.view[0]]
        btn_styles[st.session_state.current_position[0]+4][st.session_state.current_position[0]+6] = colors[st.session_state.view[2]]
        btn_styles[st.session_state.current_position[0]+5][st.session_state.current_position[0]+4] = colors[st.session_state.view[3]]
        btn_styles[st.session_state.current_position[0]+5][st.session_state.current_position[0]+5] = colors[0]
        btn_styles[st.session_state.current_position[0]+5][st.session_state.current_position[0]+6] = colors[st.session_state.view[5]]
        btn_styles[st.session_state.current_position[0]+6][st.session_state.current_position[0]+4] = colors[st.session_state.view[6]]
        btn_styles[st.session_state.current_position[0]+6][st.session_state.current_position[0]+5] = colors[st.session_state.view[7]]
        btn_styles[st.session_state.current_position[0]+6][st.session_state.current_position[0]+6] = colors[st.session_state.view[8]]
        print(btn_styles)

    # action buttons
    with st.container():
        col1, col2, col3 = st.columns(3)
        col1.button(
            '',
            key="up_left",
            on_click=None,
            args=("up_left",),
        )
        col2.button(
            "🔼",
            key="up",
            on_click=btn_click,
            args=("up",),
        )  # "🔼"
        col3.button(
            '',
            key="up_right",
            on_click=None,
            args=("up_right",),
        )
        col1.button(
            "◀️", 
            key="left", 
            on_click=btn_click, 
            args=("left",)
        )  # "◀️"
        col2.button("⛏", key="current_pos", on_click=btn_click, args=("dig",))
        col3.button(
            "▶️", 
            key="right", 
            on_click=btn_click, 
            args=("right",)
        )  # "▶️"
        col1.button(
            '',
            key="down_left",
            on_click=None,
            args=("down_left",),
        )
        col2.button(
            "🔽", key="down", on_click=btn_click, args=("down",)
        )  # "🔽"
        col3.button(
            '',
            key="down_right",
            on_click=None,
            args=("down_right",),
        )

    # player stats

    col1, col2, col3 = st.columns(3)
    position_str = f"{st.session_state.current_position}"
    col1.metric(label="Position", value=position_str, delta=None)
    col2.metric(label="Strength", value=st.session_state.current_strength, delta=None)
    col3.metric(label="Coins", value=st.session_state.current_coins, delta=None)
    style_metric_cards(
        background_color="#black", border_color="#21212f", border_left_color="#21212f"
    )

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ------------ footer  ---------------------------

st.markdown(
    f"""
    <footer>
    <div>
    <p align="center">Designed by 
    <a href="https://github.com/MinzhiYoyo" style="color: #42d7f5; text-decoration:none;">MinzhiYoyo</a>
     and 
    <a href="https://github.com/heyiWF" style="color: #42d7f5; text-decoration:none;">iBristlecone</a>
    </p>
    </div>
    </footer>""",
    unsafe_allow_html=True,
)

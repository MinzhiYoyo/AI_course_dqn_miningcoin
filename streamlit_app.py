import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
import time
import random
import numpy as np

from game_config import global_config
from streamlit_extras.grid import grid

# -------------- app config ---------------

purple_btn_color = """
                        <style>
                            div.stButton > button:first-child {background-color: #4b0082; color:#ffffff;}
                            div.stButton > button:hover {background-color: RGB(0,112,192); color:#ffffff;}
                            div.stButton > button:focus {background-color: RGB(47,117,181); color:#ffffff;}
                        </style>
                    """

st.set_page_config(page_title="Mining Coin Game", page_icon="💰")

# define external css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def btn_click(btn_text):
    st.toast(f"Clicked {btn_text}!", icon="⛏")
    position_list = list(st.session_state.current_position)
    if btn_text == "up_left":
        position_list[0] -= 1
        position_list[1] += 1
    elif btn_text == "up":
        position_list[1] += 1
    elif btn_text == "up_right":
        position_list[0] += 1
        position_list[1] += 1
    elif btn_text == "left":
        position_list[0] -= 1
    elif btn_text == "right":
        position_list[0] += 1
    elif btn_text == "down_left":
        position_list[0] -= 1
        position_list[1] -= 1
    elif btn_text == "down":
        position_list[1] -= 1
    elif btn_text == "down_right":
        position_list[0] += 1
        position_list[1] -= 1
    st.session_state.current_position = tuple(position_list)
    # TODO: update strength and coins
    if st.session_state.current_position[0] > 5 or st.session_state.current_position[0] < -5 or st.session_state.current_position[1] > 5 or st.session_state.current_position[1] < -5:
        st.session_state.done = True
        st.balloons()
        st.warning("You are out of the map! Game Over!")
    if st.session_state.current_strength <= 0:
        st.session_state.done = True
        st.balloons()
        st.warning("You have no strength! Game Over!")

###############################################
#
#
#           VARIABLES DEFINITION
#
#
################################################

# variable responsible for checking if player provided his name and game can be started
start = False

# set session states
# this is streamlit specific. For more contex please check streamlit documenation

if "map_flag" not in st.session_state:
    st.session_state["map_flag"] = np.zeros((11, 11), dtype=bool)
if "done" not in st.session_state:
    st.session_state["done"] = False
if "current_position" not in st.session_state:
    st.session_state["current_position"] = global_config["start_position"]
if "current_strength" not in st.session_state:
    st.session_state["current_strength"] = global_config["Strength"]
if "current_coins" not in st.session_state:
    st.session_state["current_coins"] = global_config["start_coins"]


###############################################
#
#
#               GAME ENGINE
#
#
################################################

# ---------------- CSS ----------------

#local_css("style.css")

# ----------------- game start --------


welcome = st.empty()
welcome.title("Welcome to Mining Coins!")

player_name_container = st.empty()
player_name_container.text_input(
    "Input your name and type ENTER", key="player_name"
)
main_text_container = st.empty()

if st.session_state.player_name != "":
    player_name_container.empty()
    main_text_container.empty()
    start = True

# START THE GAME

if start:

    # delete welcome
    welcome.empty()

    # action buttons
    with st.container():
        col1, col2, col3 = st.columns(3)
        col1.button("⛏", key="up_left", on_click=btn_click, args=("up_left", ), disabled=True)
        col2.button("🔼", key="up", on_click=btn_click, args=("up", ))
        col3.button("⛏", key="up_right", on_click=btn_click, args=("up_right", ), disabled=True)
        col1.button("◀", key="left", on_click=btn_click, args=("left", ))
        col2.button("⛏", key="dig", on_click=btn_click, args=("dig", ))
        col3.button("▶", key="right", on_click=btn_click, args=("right", ))
        col1.button("⛏", key="down_left", on_click=btn_click, args=("down_left", ), disabled=True)
        col2.button("🔽", key="down", on_click=btn_click, args=("down", ))
        col3.button("⛏", key="down_right", on_click=btn_click, args=("down_right", ), disabled=True)

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
#st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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

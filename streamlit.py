import streamlit as st
from st_clickable_images import clickable_images
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import pandas as pd
import random
import joblib
from horoscope_webscraping import get_star_ratings
from features_polar_plot import polar_plot


@st.cache_data
def load_data():
    df = pd.read_csv("./datasets/spotify_final.csv")
    segment_11 = pd.read_csv("./segment_datasets/segment_11.csv")
    segment_12 = pd.read_csv("./segment_datasets/segment_12.csv")
    segment_13 = pd.read_csv("./segment_datasets/segment_13.csv")
    segment_21 = pd.read_csv("./segment_datasets/segment_21.csv")
    segment_22 = pd.read_csv("./segment_datasets/segment_22.csv")
    segment_23 = pd.read_csv("./segment_datasets/segment_23.csv")
    segment_31 = pd.read_csv("./segment_datasets/segment_31.csv")
    segment_32 = pd.read_csv("./segment_datasets/segment_32.csv")
    segment_33 = pd.read_csv("./segment_datasets/segment_33.csv")

    return df, segment_11, segment_12, segment_13, segment_21, segment_22, segment_23, segment_31, segment_32, segment_33


def load_css():
    with open(".streamlit/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def spotify_player(track_id):
    embed_link = f"https://open.spotify.com/embed/track/{track_id}"

    return components.html(
        f'<iframe src="{embed_link}" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>',
        height=400)


class SegmentSelector:
    def __init__(self, dataset):
        self.dataset = dataset
        self.segments = [11, 12, 13, 21, 22, 23, 31, 32, 33]
        random.shuffle(self.segments)
        
        self.current_segments = self.segments.copy()
        self.round_number = 1
        self.current_pair_index = 0
        self.winners = []
        self.is_complete = False
        self.final_winner = None

    def create_pairs(self, list_to_pair):
        pairs = list(zip(list_to_pair[::2], list_to_pair[1::2]))
        if len(list_to_pair) % 2 != 0:
            pairs.append((list_to_pair[-1],))
        return pairs

    def get_random_song(self, segment):
        songs = [song for song in self.dataset if song["segment"] == segment]
        return random.choice(songs)

    def get_next_pair(self):
        if self.is_complete:
            return None

        if not self.current_segments:
            self.start_new_round()

        if self.current_pair_index < len(self.current_segments):
            if isinstance(self.current_segments[self.current_pair_index], tuple):
                return self.current_segments[self.current_pair_index]
            else:
                return (self.current_segments[self.current_pair_index],)
        else:
            return None

    def start_new_round(self):
        if len(self.winners) == 1:
            self.final_winner = self.winners[0]
            self.is_complete = True
        else:
            self.current_segments = self.create_pairs(self.winners)
            self.winners = []
            self.current_pair_index = 0
            self.round_number += 1

    def make_choice(self, choice):
        if self.is_complete:
            return self.final_winner, self.round_number

        current_pair = self.get_next_pair()
        if not current_pair:
            self.start_new_round()
            return None, self.round_number

        if len(current_pair) == 1:
            winner = current_pair[0]
        else:
            winner = current_pair[0] if choice == 1 else current_pair[1]

        self.winners.append(winner)
        self.current_pair_index += 1

        if self.current_pair_index >= len(self.current_segments):
            self.start_new_round()

        if self.is_complete:
            return self.final_winner, self.round_number
        else:
            return None, self.round_number

    def get_total_rounds(self):
        n = len(self.segments)
        return (n - 1).bit_length()


questions = [
    {
        "type": "slider",
        "question": "Please Enter Your Age",
        "min_value": 1,
        "max_value": 100,
        "step":1
    },
    {
        "type": "slider",
        "question":"How Many Hours Listen to Music in a Day ?",
        "min_value": 0.0,
        "max_value": 24.0,
        "step": 0.25
    },
    {
        "type": "image_2",
        "question": "Please Select The Music Platform That You Use",
        "choices": ["Spotify", "YouTube Music", "Apple Music", "Other"],
        "image_urls": ["https://i.ibb.co/60kxcRC/015-spotify.png",
                       "https://i.ibb.co/sJw4ymT/016-music.png",
                       "https://i.ibb.co/5LGQRX7/017-apple.png",
                       "https://i.ibb.co/HYLqd4m/018-more.png"]
    },
    {
        "type": "image_2",
        "question": "Do You Listen to Music While Working ?",
        "choices": ["Yes", "No"],
        "image_urls": ["https://i.ibb.co/nw72Gr3/013-check.png",
                       "https://i.ibb.co/6ZTNRC8/014-cancel.png"]
    },
    {
        "type": "image_3",
        "question": "What's Your Favorite Music Genre ?",
        "choices": ["Dance", "Instrumental", "Rap", "Rock",
                    "Metal", "Pop", "Jazz", "Traditional", "R&B"],
        "image_urls": ["https://i.ibb.co/qmgbg2M/022-dance.png",
                       "https://i.ibb.co/YhNddTN/23-instrumental.png",
                       "https://i.ibb.co/WBGY2T0/24-rap.png",
                       "https://i.ibb.co/K0Zzkhp/25-rock.png",
                       "https://i.ibb.co/2gcSHn7/26-metal.png",
                       "https://i.ibb.co/806CwDY/27-pop.png",
                       "https://i.ibb.co/ypbW2vF/28-jazz.png",
                       "https://i.ibb.co/R4QNXZg/29-traditional.png",
                       "https://i.ibb.co/MGYKKgv/30-rnb.png"]
    },
    {
        "type": "image_2",
        "question": "Do You Play Any Musical Instrument ?",
        "choices": ["Yes", "No"],
        "image_urls": ["https://i.ibb.co/nw72Gr3/013-check.png",
                       "https://i.ibb.co/6ZTNRC8/014-cancel.png"]
    },
    {
        "type": "image_2",
        "question": "Are You Open to Listening to New Music ?",
        "choices": ["Yes", "No"],
        "image_urls": ["https://i.ibb.co/nw72Gr3/013-check.png",
                       "https://i.ibb.co/6ZTNRC8/014-cancel.png"]
    },
    {
        "type": "multi_question",
        "main_question": "Please Select the Frequency of Listening to Music Genres",
        "sub_questions": [
            {
                "question": "Dance",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Instrumental",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Traditional",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Rap",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "R&B",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Rock",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Metal",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Pop",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Jazz",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            }
        ]
    },
    {
        "type": "image_3",
        "question": "Is Listening to Music Good For Your Mental Health ?",
        "choices": ["Improve", "No Effect", "Worsen"],
        "image_urls": ["https://i.ibb.co/sQqjgbt/019-thumb-up.png",
                       "https://i.ibb.co/m5RLhMt/021-line.png",
                       "https://i.ibb.co/94gkgTD/020-thumb-down.png"]
    },
    {
        "type": "image_3",
        "question": "What's Your Zodiac Sign ?",
        "choices": ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra",
                    "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"],
        "image_urls": ["https://i.ibb.co/XV3GDzw/001-aries.png",
                       "https://i.ibb.co/t4K7t9J/002-taurus.png",
                       "https://i.ibb.co/VN8PpmM/003-gemini.png",
                       "https://i.ibb.co/dDQhHJ2/004-cancer.png",
                       "https://i.ibb.co/q14xXsn/005-leo.png",
                       "https://i.ibb.co/zGLnSby/006-virgo.png",
                       "https://i.ibb.co/kHYt5Fd/007-libra.png",
                       "https://i.ibb.co/mBTmVrV/008-scorpio.png",
                       "https://i.ibb.co/JKbxwhb/009-sagittarius.png",
                       "https://i.ibb.co/2KRqsBn/010-capricorn.png",
                       "https://i.ibb.co/YB6WYkt/011-aquarius.png",
                       "https://i.ibb.co/YQjfWyy/012-pisces.png"]
    },
    {
        "type": "segment_selector",
        "question": "Which One ?",
    }
]


def get_question(index):
    if index < len(questions):
        return questions[index]
    else:
        return None


def initialize_session_state():
    if "question_index" not in st.session_state:
        st.session_state.question_index = 0
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = get_question(st.session_state.question_index)
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = []
    if "segment_selector" not in st.session_state:
        dataset = [
            {"segment": 11, "track_id": segment_11.sample(1)["track_id"].values[0]},
            {"segment": 12, "track_id": segment_12.sample(1)["track_id"].values[0]},
            {"segment": 13, "track_id": segment_13.sample(1)["track_id"].values[0]},
            {"segment": 21, "track_id": segment_21.sample(1)["track_id"].values[0]},
            {"segment": 22, "track_id": segment_22.sample(1)["track_id"].values[0]},
            {"segment": 23, "track_id": segment_23.sample(1)["track_id"].values[0]},
            {"segment": 31, "track_id": segment_31.sample(1)["track_id"].values[0]},
            {"segment": 32, "track_id": segment_32.sample(1)["track_id"].values[0]},
            {"segment": 33, "track_id": segment_33.sample(1)["track_id"].values[0]}
        ]
        st.session_state.segment_selector = SegmentSelector(dataset)
    if "recommendations" not in st.session_state:
            st.session_state.recommendations = []
    

def run_quiz():
    quiz_data = st.session_state.quiz_data

    if quiz_data:
        if quiz_data["type"] == "slider":
            st.subheader(f"**{quiz_data['question']}**")
            st.divider()
            slider_value = st.slider(label="question", label_visibility="hidden",  min_value=quiz_data["min_value"], max_value=quiz_data["max_value"], step=quiz_data["step"])
            
            if st.button("Submit", key="submit"):
                st.session_state.user_answers.append({quiz_data["question"]: slider_value})
                st.session_state.question_index += 1
                st.session_state.quiz_data = get_question(st.session_state.question_index)
                st.rerun()

        elif quiz_data["type"] == "multi_question":
            st.subheader(f"**{quiz_data['main_question']}**")
            st.divider()
            sub_answers = {}
            col1, empty_col_1, col2, empty_col_2, col3 = st.columns([1, 0.2, 1, 0.2, 1])
            
            columns = [col1, col2, col3]
            for i, sub_question in enumerate(quiz_data["sub_questions"]):
                with columns[i % 3]:
                    st.markdown(f"**{sub_question['question']}**")
                    options = sub_question["options"]
                    
                    selected_option = st.select_slider(label="question", label_visibility="hidden", options=options, key=f"slider_{i}")
                    sub_answers[sub_question["question"]] = selected_option

            if st.button("Submit", key="submit_answers"):
                st.session_state.user_answers.append({quiz_data["main_question"]: sub_answers})
                st.session_state.question_index += 1
                st.session_state.quiz_data = get_question(st.session_state.question_index)
                st.rerun()

        elif quiz_data["type"] == "image_2":
            st.subheader(f"**{quiz_data['question']}**")
            st.divider()
            clicked = clickable_images(
                quiz_data["image_urls"],
                titles=[choice for choice in quiz_data["choices"]],
                div_style={"display": "grid",
                           "grid-template-columns": "repeat(2, 1fr)",
                           "gap": "10px",
                           "justify-content": "center",
                           "background-color": "#E8E8E8",
                           "padding": "20px"},
                img_style={"width": "150px",
                           "height": "150px",
                           "object-fit": "cover",
                           "margin": "auto"})

            if clicked > -1:
                selected_answer = quiz_data["choices"][clicked]
                st.session_state.user_answers.append({quiz_data["question"]: selected_answer})
                st.session_state.question_index += 1
                st.session_state.quiz_data = get_question(st.session_state.question_index)
                st.rerun()

        elif quiz_data["type"] == "image_3":
            st.subheader(f"**{quiz_data['question']}**")
            st.divider()
            clicked = clickable_images(
                quiz_data["image_urls"],
                titles=[choice for choice in quiz_data["choices"]],
                div_style={"display": "grid",
                        "grid-template-columns": "repeat(3, 1fr)",
                        "gap": "10px",
                        "justify-content": "center",
                        "background-color": "#E8E8E8",
                        "padding": "20px"},
                img_style={"width": "150px",
                        "height": "150px",
                        "object-fit": "cover",
                        "margin": "auto"})

            if clicked > -1:
                selected_answer = quiz_data["choices"][clicked]
                st.session_state.user_answers.append({quiz_data["question"]: selected_answer})
                st.session_state.question_index += 1
                st.session_state.quiz_data = get_question(st.session_state.question_index)
                st.rerun()

        elif quiz_data["type"] == "segment_selector":
            st.subheader(f"**{quiz_data['question']}**")
            st.divider()
            
            if not st.session_state.segment_selector.is_complete:
                current_pair = st.session_state.segment_selector.get_next_pair()
                
                if current_pair:
                    emptycol1, col1, col2, emptycol2= st.columns([1.1,1,1,1])
                    
                    with col1:
                        segment1 = current_pair[0]
                        song1 = st.session_state.segment_selector.get_random_song(segment1)
                        spotify_player(song1["track_id"])
                        empty1, col_button, empty2= st.columns([1,1,1])

                        with col_button:
                            if st.button("Select", key="select_1"):
                                winner, round_number = st.session_state.segment_selector.make_choice(1)
                                if winner:
                                    st.session_state.user_answers.append({"selected_segment": winner})
                                    st.session_state.question_index += 1
                                    st.session_state.quiz_data = get_question(st.session_state.question_index)
                                st.rerun()
                    
                    if len(current_pair) > 1:
                        with col2:
                            segment2 = current_pair[1]
                            song2 = st.session_state.segment_selector.get_random_song(segment2)
                            spotify_player(song2["track_id"])
                            empty1, col_button, empty2= st.columns([1,1,1])

                            with col_button:
                                if st.button("Select", key="select_2"):
                                    winner, round_number = st.session_state.segment_selector.make_choice(2)
                                    if winner:
                                        st.session_state.user_answers.append({"selected_segment": winner})
                                        st.session_state.question_index += 1
                                        st.session_state.quiz_data = get_question(st.session_state.question_index)
                                    st.rerun()
                    
                    else:
                        winner, round_number = st.session_state.segment_selector.make_choice(1)
                        if winner:
                            st.session_state.user_answers.append({"selected_segment": winner})
                            st.session_state.question_index += 1
                            st.session_state.quiz_data = get_question(st.session_state.question_index)
                        st.rerun()

    else:
        answer_dict ={}
        for answer in st.session_state.user_answers:
            for question, response in answer.items():
                if isinstance(response, dict):
                    for sub_q, sub_r in response.items():
                        answer_dict[sub_q] = sub_r
                else:
                    answer_dict[question] = response
        ###
        answer_dict["age"] = answer_dict.pop("Please Enter Your Age")
        answer_dict["hours_per_day"] = answer_dict.pop("How Many Hours Listen to Music in a Day ?")
        answer_dict["streaming_service"] = answer_dict.pop("Please Select The Music Platform That You Use")
        answer_dict["while_working"] = answer_dict.pop("Do You Listen to Music While Working ?")
        answer_dict["instrumentalist"] = answer_dict.pop("Do You Play Any Musical Instrument ?")
        answer_dict["fav_genre"] = answer_dict.pop("What's Your Favorite Music Genre ?")
        answer_dict["exploratory"] = answer_dict.pop("Are You Open to Listening to New Music ?")
        answer_dict["frequency_dance"] = answer_dict.pop("Dance")
        answer_dict["frequency_instrumental"] = answer_dict.pop("Instrumental")
        answer_dict["frequency_traditional"] = answer_dict.pop("Traditional")
        answer_dict["frequency_rap"] = answer_dict.pop("Rap")
        answer_dict["frequency_rnb"] = answer_dict.pop("R&B")
        answer_dict["frequency_rock"] = answer_dict.pop("Rock")
        answer_dict["frequency_metal"] = answer_dict.pop("Metal")
        answer_dict["frequency_pop"] = answer_dict.pop("Pop")
        answer_dict["frequency_jazz"] = answer_dict.pop("Jazz")
        answer_dict["music_effects"] = answer_dict.pop("Is Listening to Music Good For Your Mental Health ?")
        answer_dict["pc_segment"] = answer_dict.pop("selected_segment")
        
        answer_dict["zodiac"] = answer_dict.pop("What's Your Zodiac Sign ?")
        hustle_star, vibe_star = get_star_ratings(answer_dict.get("zodiac"))
        answer_dict["hustle"] = int(hustle_star)
        answer_dict["vibe"] = int(vibe_star)
        ###
        st.markdown(answer_dict)

        ### BURAYA streamlit_add.py KODU EKLENECEK ###

    
def analysis_content():
    st.divider()

    col1, col2, col3 = st.columns([0.7,1,0.7])

    with col2:
        options_analysis = option_menu(None, ["Mental Survey", "Spotify"], 
                        icons=["card-checklist", "spotify"], 
                        menu_icon="cast", default_index=0, orientation="horizontal",
                        styles={"container": {"padding": "0!important", 
                                                "backgroundColor": "#E8E8E8",
                                                "width": "100%",
                                                "max-width": "100%"}})


    if options_analysis == "Mental Survey":
        st.write("Mental Survey Analysis")

    elif options_analysis == "Spotify":
        st.write("Spotify Analysis")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col3:
            st.markdown("Music Genre Features")
            st.divider()
            selected_genre = st.selectbox(label="question", label_visibility="hidden", options=["Dance", "Instrumental", "Rap",
                                                                                                "Rock", "Metal", "Pop",
                                                                                                "Jazz", "Traditional", "R&B"])
            polar_plot(df, selected_genre)


def team_content():
    st.divider()
    
    # CSS for centering subheaders
    st.markdown("""
    <style>
    .centered-subheader {
        text-align: center;
        font-weight: bold;
        font-size: 1.5em;
        margin-bottom: 1em;
    }
    .social-icons {
        display: flex;
        justify-content: center;
        padding: 10px;
    }
    .social-icons a {
        color: #0e1117;
        text-decoration: none;
        margin: 0 10px;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown('<p class="centered-subheader">Büşra Sürücü</p>', unsafe_allow_html=True)
        st.write("Veri Bilimci")
        st.write("Python ve veri analizi konularında uzman.")
        st.divider()
        st.markdown("""
        <div class="social-icons">
            <a href="https://linkedin.com/in/busrasurucu" target="_blank" title="LinkedIn">
                <i class="fab fa-linkedin"></i>
            </a>
            <a href="https://github.com/busrasurucu" target="_blank" title="GitHub">
                <i class="fab fa-github"></i>
            </a>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="centered-subheader">Hilal Alpak</p>', unsafe_allow_html=True)
        st.write("Veri Bilimci")
        st.write("Python ve veri analizi konularında uzman.")
        st.divider()
        st.markdown("""
        <div class="social-icons">
            <a href="https://linkedin.com/in/hilal-alpak" target="_blank" title="LinkedIn">
                <i class="fab fa-linkedin"></i>
            </a>
            <a href="https://github.com/hilalalpak" target="_blank" title="GitHub">
                <i class="fab fa-github"></i>
            </a>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown('<p class="centered-subheader">Kürşat Dinç</p>', unsafe_allow_html=True)
        st.write("Veri Bilimci")
        st.write("Python ve veri analizi konularında uzman.")
        st.divider()
        st.markdown("""
        <div class="social-icons">
            <a href="https://linkedin.com/in/kursatdinc" target="_blank" title="LinkedIn">
                <i class="fab fa-linkedin"></i>
            </a>
            <a href="https://github.com/kursatdinc" target="_blank" title="GitHub">
                <i class="fab fa-github"></i>
            </a>
        </div>
        """, unsafe_allow_html=True)
###############
###############


st.set_page_config(layout="wide", page_title="Therapy Tunes", page_icon="🎶")
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">', unsafe_allow_html=True)
load_css()
df, segment_11, segment_12, segment_13, segment_21, segment_22, segment_23, segment_31, segment_32, segment_33 = load_data()
col1, col2, col3 = st.columns([0.8,1,0.7])

with col2:
    st.markdown("""
    <div style="display: flex; align-items: center;">
        <img src="https://i.ibb.co/bQDZpKc/theraphytunes-logo.png" alt="Logo" style="height: 150px; margin-right: 10px;">
        <h3 style="display: inline;">Mood Music Recommender</h3>
    </div>
    """, unsafe_allow_html=True)
    
st.divider()

options = option_menu(None, ["Quiz", "Analysis", "Team"], 
                      icons=["music-note-list", "bar-chart-steps", "people"], 
                      menu_icon="cast", default_index=0, orientation="horizontal",
                      styles={"container": {"padding": "0!important", 
                                            "backgroundColor": "#E8E8E8",
                                            "margin": "0!important",
                                            "width": "100%",
                                            "max-width": "100%"}})


if options == "Quiz":
    initialize_session_state()
    run_quiz()

elif options == "Analysis":
    analysis_content()

elif options == "Team":
    team_content()
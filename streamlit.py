import streamlit as st
from st_clickable_images import clickable_images
import streamlit.components.v1 as components
import pandas as pd
import random
from horoscope_webscraping import get_star_ratings

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
        "question": "Enter your age.",
        "min_value": 1,
        "max_value": 100,
        "step":1
    },
    {
        "type": "slider",
        "question":"How many hours a day do you listen to music?",
        "min_value": 0.0,
        "max_value": 24.0,
        "step": 0.25
    },
    {
        "type": "image_2",
        "question": "Select the music platform that you use?",
        "choices": ["Spotify", "YouTube Music", "Apple Music", "Other"],
        "image_urls": ["https://i.ibb.co/60kxcRC/015-spotify.png",
                       "https://i.ibb.co/sJw4ymT/016-music.png",
                       "https://i.ibb.co/5LGQRX7/017-apple.png",
                       "https://i.ibb.co/HYLqd4m/018-more.png"]
    },
    {
        "type": "image_2",
        "question": "Do you listen to music while working?",
        "choices": ["Yes", "No"],
        "image_urls": ["https://i.ibb.co/nw72Gr3/013-check.png",
                       "https://i.ibb.co/6ZTNRC8/014-cancel.png"]
    },
    {
        "type": "image_3",
        "question": "What's your favorite music genre?",
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
        "question": "Are you open to listening to new music?",
        "choices": ["Yes", "No"],
        "image_urls": ["https://i.ibb.co/nw72Gr3/013-check.png",
                       "https://i.ibb.co/6ZTNRC8/014-cancel.png"]
    },
    {
        "type": "multi_question",
        "main_question": "Select the frequency of listening to music genres?",
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
        "question": "Is listening to music good for mental health?",
        "choices": ["Improve", "No Effect", "Worsen"],
        "image_urls": ["https://i.ibb.co/sQqjgbt/019-thumb-up.png",
                       "https://i.ibb.co/m5RLhMt/021-line.png",
                       "https://i.ibb.co/94gkgTD/020-thumb-down.png"]
    },
    {
        "type": "image_3",
        "question": "What's your zodiac sign?",
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
        "question": "Which One?",
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


def run_quiz():
    quiz_data = st.session_state.quiz_data

    if quiz_data:
        if quiz_data["type"] == "slider":
            st.markdown(f"**{quiz_data['question']}**")
            slider_value = st.slider("Your answer", min_value=quiz_data["min_value"], max_value=quiz_data["max_value"], step=quiz_data["step"])
            
            if st.button("Submit"):
                st.session_state.user_answers.append({quiz_data["question"]: slider_value})
                st.session_state.question_index += 1
                st.session_state.quiz_data = get_question(st.session_state.question_index)
                st.rerun()

        elif quiz_data["type"] == "multi_question":
            st.markdown(f"**{quiz_data['main_question']}**")
            sub_answers = {}
            col1, col2, col3 = st.columns(3)
            columns = [col1, col2, col3]
            for i, sub_question in enumerate(quiz_data["sub_questions"]):
                with columns[i % 3]:
                    st.markdown(f"**{sub_question['question']}**")
                    options = sub_question["options"]
                    min_value = 0
                    max_value = len(options) - 1
                    value = 0
                    
                    selected_value = st.slider("", min_value=min_value, max_value=max_value, value=value, step=1, key=f"slider_{i}")
                    selected_option = options[selected_value]
                    
                    st.write(f"Selected: {selected_option}")
                    
                    sub_answers[sub_question["question"]] = selected_option

            if st.button("Submit Answers"):
                st.session_state.user_answers.append({quiz_data["main_question"]: sub_answers})
                st.session_state.question_index += 1
                st.session_state.quiz_data = get_question(st.session_state.question_index)
                st.rerun()

        elif quiz_data["type"] == "image_2":
            st.markdown(f"**{quiz_data['question']}**")
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
            st.markdown(f"**{quiz_data['question']}**")
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
            st.markdown(f"**{quiz_data['question']}**")
            
            if not st.session_state.segment_selector.is_complete:
                current_pair = st.session_state.segment_selector.get_next_pair()
                
                if current_pair:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        segment1 = current_pair[0]
                        song1 = st.session_state.segment_selector.get_random_song(segment1)
                        spotify_player(song1["track_id"])
                        
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
        answer_dict["age"] = answer_dict.pop("Enter your age.")
        answer_dict["hours_per_day"] = answer_dict.pop("How many hours a day do you listen to music?")
        answer_dict["streaming_service"] = answer_dict.pop("Select the music platform that you use?")
        answer_dict["while_working"] = answer_dict.pop("Do you listen to music while working?")
        answer_dict["fav_genre"] = answer_dict.pop("What's your favorite music genre?")
        answer_dict["exploratory"] = answer_dict.pop("Are you open to listening to new music?")
        answer_dict["frequency_dance"] = answer_dict.pop("Dance")
        answer_dict["frequency_instrumental"] = answer_dict.pop("Instrumental")
        answer_dict["frequency_traditional"] = answer_dict.pop("Traditional")
        answer_dict["frequency_rap"] = answer_dict.pop("Rap")
        answer_dict["frequency_rnb"] = answer_dict.pop("R&B")
        answer_dict["frequency_rock"] = answer_dict.pop("Rock")
        answer_dict["frequency_metal"] = answer_dict.pop("Metal")
        answer_dict["frequency_pop"] = answer_dict.pop("Pop")
        answer_dict["frequency_jazz"] = answer_dict.pop("Jazz")
        answer_dict["music_effects"] = answer_dict.pop("Is listening to music good for mental health?")
        answer_dict["pc_segment"] = answer_dict.pop("selected_segment")
        
        answer_dict["zodiac"] = answer_dict.pop("What's your zodiac sign?")
        hustle_star, vibe_star = get_star_ratings(answer_dict.get("zodiac"))
        answer_dict["hustle"] = int(hustle_star)
        answer_dict["vibe"] = int(vibe_star)
        ###
        st.markdown(answer_dict)


def tab2_content():
    st.title("Tab 2")
    st.write("This is the content for Tab 2. You can add your desired content here.")


def tab3_content():
    st.title("Tab 3")
    st.write("This is the content for Tab 3. You can add your desired content here.")


###############
###############


st.set_page_config(layout="wide", page_title="Therapy Tunes", page_icon="🎶")
load_css()
df, segment_11, segment_12, segment_13, segment_21, segment_22, segment_23, segment_31, segment_32, segment_33 = load_data()


tab1, tab2, tab3 = st.tabs(["Quiz", "Tab 2", "Tab 3"])

with tab1:
    st.title("Music Habits Quiz")
    initialize_session_state()
    run_quiz()

with tab2:
    tab2_content()

with tab3:
    tab3_content()
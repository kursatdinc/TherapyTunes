import streamlit as st
from st_clickable_images import clickable_images
import streamlit.components.v1 as components
import pandas as pd
import random

@st.cache_data
def load_data():
    df = pd.read_csv("./datasets/spotify_final.csv")
    df_segment = pd.read_csv("./datasets/segment.csv")

    return df, df_segment

def load_css():
    with open(".streamlit/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

def spotify_player(track_id):
    embed_link = f"https://open.spotify.com/embed/track/{track_id}"
    return components.html(
        f"<iframe src="{embed_link}" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>",
        height=400)

df, df_segment = load_data()


class SegmentSelector:
    def __init__(self, dataset):
        self.dataset = dataset
        self.segments = [0, 1, 2, 3, 4, 5, 6, 7, 8]
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
        "type": "image",
        "question": "Select the music platform that you use?",
        "choices": ["Spotify", "YouTube Music", "Apple Music", "Other"],
        "image_urls": ["https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Spotify_logo_with_text.svg/1200px-Spotify_logo_with_text.svg.png",
                       "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/YT_Music.svg/1024px-YT_Music.svg.png",
                       "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Apple_Music_logo.svg/2560px-Apple_Music_logo.svg.png",
                       "https://www.edagroups.com/files/3214/7969/6498/other.png"]
    },
    {
        "type": "image",
        "question": "Do you listen to music while working?",
        "choices": ["Yes", "No"],
        "image_urls": ["https://icon2.cleanpng.com/20180319/opq/kisspng-computer-icons-clip-art-check-yes-ok-icon-5ab061dfcd38e3.7297168415215088318406.jpg",
                       "https://banner2.cleanpng.com/20180424/wzw/kisspng-no-symbol-sign-clip-art-prohibited-signs-5adf452cc063a7.6278734415245816767881.jpg"]
    },
    {
        "type": "image",
        "question": "What's your favorite music genre?",
        "choices": ["Classical", "Country", "EDM", "Folk", "Gospel", "Hip-Hop", "Jazz",
                    "K-Pop", "Latin", "Lofi", "Metal", "Pop", "R&B", "Rock"],
        "image_urls": ["https://www.onlinelogomaker.com/blog/wp-content/uploads/2017/06/music-logo-design.jpg"] * 14
    },
    {
        "type": "image",
        "question": "Are you open to listening to new music?",
        "choices": ["Yes", "No"],
        "image_urls": ["https://icon2.cleanpng.com/20180319/opq/kisspng-computer-icons-clip-art-check-yes-ok-icon-5ab061dfcd38e3.7297168415215088318406.jpg",
                       "https://banner2.cleanpng.com/20180424/wzw/kisspng-no-symbol-sign-clip-art-prohibited-signs-5adf452cc063a7.6278734415245816767881.jpg"]
    },
    {
        "type": "multi_question",
        "main_question": "Select the frequency of listening to music genres?",
        "sub_questions": [
            {
                "question": "Classical",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Country",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "EDM",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Folk",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Gospel",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Hip-Hop",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Jazz",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "K-Pop",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Latin",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Lofi",
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
                "question": "R&B",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            },
            {
                "question": "Rock",
                "options": ["Never", "Rarely", "Sometimes", "Often"]
            }]
    },
    {
        "type": "image",
        "question": "Is listening to music good for mental health?",
        "choices": ["Improve", "No Effect", "Worsen"],
        "image_urls": ["https://icon2.cleanpng.com/20180319/opq/kisspng-computer-icons-clip-art-check-yes-ok-icon-5ab061dfcd38e3.7297168415215088318406.jpg",
                       "https://banner2.cleanpng.com/20180424/wzw/kisspng-no-symbol-sign-clip-art-prohibited-signs-5adf452cc063a7.6278734415245816767881.jpg",
                       "https://banner2.cleanpng.com/20180424/wzw/kisspng-no-symbol-sign-clip-art-prohibited-signs-5adf452cc063a7.6278734415245816767881.jpg"]
    },
    {
        "type": "image",
        "question": "What's your zodiac sign?",
        "choices": ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra",
                    "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"],
        "image_urls": ["https://miro.medium.com/v2/resize:fit:800/0*2llmBOj3G8yJ4Jil.jpg"] * 12
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
            {"segment": 0, "track_id": df_segment[df_segment["segment"] == 0].sample(1)["track_id"].values[0]},
            {"segment": 1, "track_id": df_segment[df_segment["segment"] == 1].sample(1)["track_id"].values[0]},
            {"segment": 2, "track_id": df_segment[df_segment["segment"] == 2].sample(1)["track_id"].values[0]},
            {"segment": 3, "track_id": df_segment[df_segment["segment"] == 3].sample(1)["track_id"].values[0]},
            {"segment": 4, "track_id": df_segment[df_segment["segment"] == 4].sample(1)["track_id"].values[0]},
            {"segment": 5, "track_id": df_segment[df_segment["segment"] == 5].sample(1)["track_id"].values[0]},
            {"segment": 6, "track_id": df_segment[df_segment["segment"] == 6].sample(1)["track_id"].values[0]},
            {"segment": 7, "track_id": df_segment[df_segment["segment"] == 7].sample(1)["track_id"].values[0]},
            {"segment": 8, "track_id": df_segment[df_segment["segment"] == 8].sample(1)["track_id"].values[0]}
        ]
        st.session_state.segment_selector = SegmentSelector(dataset)

st.title("Music Habits Quiz")

initialize_session_state()

quiz_data = st.session_state.quiz_data

if quiz_data:
    if quiz_data["type"] == "slider":
        st.markdown(f"**{quiz_data["question"]}**")
        slider_value = st.slider("Your answer", min_value=quiz_data["min_value"], max_value=quiz_data["max_value"], step=quiz_data["step"])
        
        if st.button("Submit"):
            st.session_state.user_answers.append({quiz_data["question"]: slider_value})
            st.session_state.question_index += 1
            st.session_state.quiz_data = get_question(st.session_state.question_index)
            st.rerun()

    elif quiz_data["type"] == "multi_question":
        st.markdown(f"**{quiz_data["main_question"]}**")
        sub_answers = {}
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
        for i, sub_question in enumerate(quiz_data["sub_questions"]):
            with columns[i % 3]:
                st.markdown(f"**{sub_question["question"]}**")
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

    elif quiz_data["type"] == "image":
        st.markdown(f"**{quiz_data["question"]}**")
        clicked = clickable_images(
            quiz_data["image_urls"],
            titles=[choice for choice in quiz_data["choices"]],
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px"},
        )

        if clicked > -1:
            selected_answer = quiz_data["choices"][clicked]
            st.session_state.user_answers.append({quiz_data["question"]: selected_answer})
            st.session_state.question_index += 1
            st.session_state.quiz_data = get_question(st.session_state.question_index)
            st.rerun()

    elif quiz_data["type"] == "segment_selector":
        st.markdown(f"**{quiz_data["question"]}**")
        
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
    st.markdown("Quiz completed. Here are your answers:")
    for answer in st.session_state.user_answers:
        for question, response in answer.items():
            st.write(f"**{question}**")
            if isinstance(response, dict):
                for sub_q, sub_r in response.items():
                    st.write(f"  {sub_q}: {sub_r}")
            else:
                st.write(f"  {response}")
        st.write("---")
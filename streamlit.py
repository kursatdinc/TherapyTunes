import streamlit as st
from st_clickable_images import clickable_images

def load_css():
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

questions = [
    {
        "type": "slider",
        "question": "Enter your age.",
        "min_value": 1,
        "max_value": 100
    },
    {
        "type": "slider",
        "question":"How many hours a day do you listen to music?",
        "min_value": 0,
        "max_value": 24
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

st.title("Music Habits Quiz")

initialize_session_state()

quiz_data = st.session_state.quiz_data

if quiz_data:
    if quiz_data["type"] == "slider":
        st.markdown(f"**{quiz_data['question']}**")
        slider_value = st.slider("Your answer", min_value=quiz_data["min_value"], max_value=quiz_data["max_value"])
        
        if st.button("Submit"):
            st.session_state.user_answers.append({quiz_data['question']: slider_value})
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
                selected_option = st.radio("", sub_question["options"], key=f"radio_{i}")
                sub_answers[sub_question["question"]] = selected_option
        
        if st.button("Submit Answers"):
            st.session_state.user_answers.append({quiz_data['main_question']: sub_answers})
            st.session_state.question_index += 1
            st.session_state.quiz_data = get_question(st.session_state.question_index)
            st.rerun()

    elif quiz_data["type"] == "image":
        st.markdown(f"**{quiz_data['question']}**")
        clicked = clickable_images(
            quiz_data["image_urls"],
            titles=[choice for choice in quiz_data["choices"]],
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px"},
        )

        if clicked > -1:
            selected_answer = quiz_data["choices"][clicked]
            st.session_state.user_answers.append({quiz_data['question']: selected_answer})
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
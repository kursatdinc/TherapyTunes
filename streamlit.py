import streamlit as st
from st_clickable_images import clickable_images

questions = [
        {
        "type": "slider",
        "question": "Enter your age.",
        "min_value": 1,
        "max_value": 100
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
        "type": "slider",
        "question": "How many hours a day do you listen to music?",
        "min_value": 1,
        "max_value": 24
    },
    {
        "type": "image",
        "question": "Do you listen to music while working?",
        "choices": ["Yes", "No"],
        "image_urls": ["https://icon2.cleanpng.com/20180319/opq/kisspng-computer-icons-clip-art-check-yes-ok-icon-5ab061dfcd38e3.7297168415215088318406.jpg",
                       "https://banner2.cleanpng.com/20180424/wzw/kisspng-no-symbol-sign-clip-art-prohibited-signs-5adf452cc063a7.6278734415245816767881.jpg"]
    },
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

st.title("Quiz with Clickable Images and Sliders")

initialize_session_state()

quiz_data = st.session_state.quiz_data

if quiz_data:
    st.markdown(f"**Question {st.session_state.question_index + 1}: {quiz_data['question']}**")

    if quiz_data["type"] == "image":
        clicked = clickable_images(
            quiz_data["image_urls"],
            titles=[choice for choice in quiz_data["choices"]],
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px"},
        )

        if clicked > -1:
            selected_answer = quiz_data["choices"][clicked]
            st.session_state.user_answers.append(selected_answer)
            st.session_state.question_index += 1
            st.session_state.quiz_data = get_question(st.session_state.question_index)
            st.rerun()

    elif quiz_data["type"] == "slider":
        slider_value = st.slider("Your answer", min_value=quiz_data["min_value"], max_value=quiz_data["max_value"])
        
        if st.button("Submit"):
            st.session_state.user_answers.append(slider_value)
            st.session_state.question_index += 1
            st.session_state.quiz_data = get_question(st.session_state.question_index)
            st.rerun()

else:
    st.markdown("Quiz tamamlandı. İşte cevaplarınız:")
    for i, answer in enumerate(st.session_state.user_answers):
        st.write(f"Soru {i+1}: {answer}")
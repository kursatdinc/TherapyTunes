import streamlit as st
from st_clickable_images import clickable_images

questions = [
    {
        "question": "What is the capital of France?",
        "choices": ["Paris", "London", "Berlin", "Rome"],
        "image_urls": ["https://images.unsplash.com/photo-1499856871958-5b9627545d1a?w=700",
                       "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=700",
                       "https://images.unsplash.com/photo-1560969184-10fe8719e047?w=700",
                       "https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=700"]
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["Earth", "Mars", "Jupiter", "Saturn"],
        "image_urls": ["https://images.unsplash.com/photo-1614730321146-b6fa6a46bcb4?w=700",
                       "https://images.unsplash.com/photo-1614728263952-84ea256f9679?w=700",
                       "https://images.unsplash.com/photo-1630839437035-dac17da580d0?w=700",
                       "https://images.unsplash.com/photo-1614314107768-6018061b5b72?w=700"]
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


st.title("Quiz with Clickable Images")

initialize_session_state()

quiz_data = st.session_state.quiz_data

if quiz_data:
    st.markdown(f"**Question {st.session_state.question_index + 1}: {quiz_data["question"]}**")

    clicked = clickable_images(
        quiz_data["image_urls"],
        titles=[choice for choice in quiz_data["choices"]],
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": "200px"},
    )

    if clicked > -1:
        selected_answer = quiz_data["choices"][clicked]
        st.session_state.user_answers.append(selected_answer)

        # Sonraki soruya geç
        st.session_state.question_index += 1
        st.session_state.quiz_data = get_question(st.session_state.question_index)
        
        if st.session_state.quiz_data is None:
            st.markdown("Quiz tamamlandı. İşte cevaplarınız:")
            for i, answer in enumerate(st.session_state.user_answers):
                st.write(f"Soru {i+1}: {answer}")
        else:
            st.rerun()

else:
    st.markdown("Quiz tamamlandı. İşte cevaplarınız:")
    for i, answer in enumerate(st.session_state.user_answers):
        st.write(f"Soru {i+1}: {answer}")
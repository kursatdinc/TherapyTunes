import streamlit as st

questions = [
    {
        "question": "What is the capital of France?",
        "choices": ["Paris", "London", "Berlin", "Rome"],
        "correct_answer": "Paris",
        "explanation": "Paris is the capital and most populous city of France."
    },
    {
        "question": "What is 2 + 2?",
        "choices": ["3", "4", "5", "6"],
        "correct_answer": "4",
        "explanation": "2 + 2 equals 4."
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["Earth", "Mars", "Jupiter", "Saturn"],
        "correct_answer": "Mars",
        "explanation": "Mars is often called the 'Red Planet' because of its reddish appearance."
    },
    {
        "question": "What is the chemical symbol for water?",
        "choices": ["H2O", "O2", "CO2", "HO2"],
        "correct_answer": "H2O",
        "explanation": "H2O is the chemical formula for water, consisting of two hydrogen atoms and one oxygen atom."
    },
    {
        "question": "Who wrote 'Romeo and Juliet'?",
        "choices": ["William Shakespeare", "Charles Dickens", "Mark Twain", "Jane Austen"],
        "correct_answer": "William Shakespeare",
        "explanation": "William Shakespeare is the author of the famous play 'Romeo and Juliet'."
    },
    {
        "question": "What is the largest mammal in the world?",
        "choices": ["Elephant", "Blue Whale", "Giraffe", "Great White Shark"],
        "correct_answer": "Blue Whale",
        "explanation": "The Blue Whale is the largest mammal in the world, reaching lengths of up to 100 feet."
    },
    {
        "question": "What is the boiling point of water at sea level?",
        "choices": ["90°C", "100°C", "110°C", "120°C"],
        "correct_answer": "100°C",
        "explanation": "At sea level, water boils at 100 degrees Celsius (212 degrees Fahrenheit)."
    },
    {
        "question": "Which element has the atomic number 1?",
        "choices": ["Hydrogen", "Helium", "Lithium", "Carbon"],
        "correct_answer": "Hydrogen",
        "explanation": "Hydrogen is the first element on the periodic table with the atomic number 1."
    },
    {
        "question": "Who painted the Mona Lisa?",
        "choices": ["Leonardo da Vinci", "Vincent van Gogh", "Pablo Picasso", "Claude Monet"],
        "correct_answer": "Leonardo da Vinci",
        "explanation": "The Mona Lisa was painted by the Italian artist Leonardo da Vinci."
    },
    {
        "question": "What is the hardest natural substance on Earth?",
        "choices": ["Gold", "Iron", "Diamond", "Silver"],
        "correct_answer": "Diamond",
        "explanation": "Diamond is the hardest natural substance known on Earth."
    }
]

def get_question(index):
    if index < len(questions):
        return questions[index]
    else:
        return None

def initialize_session_state():
    session_state = st.session_state
    session_state.form_count = 0
    session_state.quiz_data = get_question(session_state.form_count)

# Streamlit uygulaması
st.title('Initial Quiz')

if 'form_count' not in st.session_state:
    initialize_session_state()

quiz_data = st.session_state.quiz_data

if quiz_data:
    st.markdown(f"Question: {quiz_data['question']}")
    
    form = st.form(key=f"quiz_form_{st.session_state.form_count}")
    user_choice = form.radio("Choose an answer:", quiz_data['choices'])
    
    if form.form_submit_button("Submit your answer"):
        
        st.session_state.form_count += 1
        st.session_state.quiz_data = get_question(st.session_state.form_count)
        
        if st.session_state.quiz_data is None:
            st.markdown("No more questions available.")
        else:
            st.experimental_rerun()
else:
    st.markdown("No more questions available.")
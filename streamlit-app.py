from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import PromptTemplate
import streamlit as st

load_dotenv()

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📘",
    layout="wide"
)

mistral = ChatMistralAI(
    model="mistral-small",
    temperature=0.7
)

summ_temp = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the following text into 5 bullet points.
{text}
"""
)

quiz_temp = PromptTemplate(
    input_variables=["text"],
    template="""
Create a quiz based on the following text.
Have 5 questions with 4 options each and indicate the correct answer.
{text}
"""
)

chat_temp = PromptTemplate(
    input_variables=["text", "question"],
    template="""
You are a helpful AI study assistant.

Answer only from the given context.
If the answer is not available in the text, say: "I don't know based on the provided text."

Context:
{text}

User Question:
{question}
"""
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "summary_output" not in st.session_state:
    st.session_state.summary_output = ""

if "quiz_output" not in st.session_state:
    st.session_state.quiz_output = ""

st.title("📘 AI Study Assistant")


with st.sidebar:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

study_text = st.text_area(
    "Enter your study text",
    height=300,
    placeholder="Paste your notes, article, or chapter here..."
)

tab1, tab2, tab3 = st.tabs(["Summary", "Quiz", "Chat"])

with tab1:
    st.subheader("Text Summary")
    if st.button("Generate Summary", use_container_width=True):
        if study_text.strip():
            chain = summ_temp | mistral
            res = chain.invoke({"text": study_text})
            st.session_state.summary_output = res.content
        else:
            st.warning("Please enter some study text first.")

    st.text_area(
        "Summary Output",
        value=st.session_state.summary_output,
        height=300
    )

with tab2:
    st.subheader("Quiz Generator")
    if st.button("Generate Quiz", use_container_width=True):
        if study_text.strip():
            chain = quiz_temp | mistral
            res = chain.invoke({"text": study_text})
            st.session_state.quiz_output = res.content
        else:
            st.warning("Please enter some study text first.")

    st.text_area(
        "Quiz Output",
        value=st.session_state.quiz_output,
        height=350
    )

with tab3:
    st.subheader("Chat with your text")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question from the study text...")

    if prompt:
        if not study_text.strip():
            st.warning("Please enter study text before starting the chat.")
        else:
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt
            })

            with st.chat_message("user"):
                st.markdown(prompt)

            chain = chat_temp | mistral
            res = chain.invoke({
                "text": study_text,
                "question": prompt
            })

            answer = res.content

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

            with st.chat_message("assistant"):
                st.markdown(answer)
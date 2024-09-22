import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryMemory
#from dotenv import load_dotenv
import os

# Load API key from .env file
#load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI model with the API key and controlled temperature
llm = ChatOpenAI(
    api_key=openai_api_key,
    model='ft:gpt-3.5-turbo-0125:personal::AAD3yAdF',
    temperature=0.15,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# Title and description of the chatbot
st.title("Gastroenterology Symptom Checker Chatbot")
st.write(
    "This chatbot helps collect your symptoms and provides guidance on gastrointestinal conditions. "
    "Please enter your details and symptoms to begin. The chatbot will ask relevant follow-up questions based on your responses."
)

# Template for the updated prompt that includes demographic details
prompt_template = """
You are a virtual health assistant specializing in gastroenterology. A patient is describing their symptoms.

- The patient is {input}.
- Based on their demographic details and symptoms, your task is to:
    1. Ask logical follow-up questions based on the user prompt.
    2. Avoid repeating previously asked questions or information.
    3. If the patient provides a clear symptom (e.g., bloating), don't ask about pain unless explicitly mentioned.
    4. Your responses should flow naturally, reflecting the patient's answers and moving the conversation forward.
    5. Ask concise questions in one line only to help identify gastrointestinal conditions (e.g., GERD, IBS, Crohn's disease).
    6. Ask one clear and concise follow-up question at a time, ensuring that each question is written in one line.
    7. Narrow down possibilities by focusing on key symptoms.
    8. If you find a good enough match to a gastrointestinal condition, provide a preliminary diagnosis without waiting for all symptoms to be collected.
    9. Suggest when the patient should seek immediate care or consult a doctor.
    10. Provide an action plan or possible medication suggestions if relevant.

- The previous conversation history is: {history}. Remember the chat history and customize the questions based on it.
"""

# Create a LangChain PromptTemplate
prompt = PromptTemplate(input_variables=["input", "history"], template=prompt_template)

# Initialize conversation summary memory with the LLM
memory = ConversationSummaryMemory(llm=llm, return_messages=True)

# Create an LLMChain with the OpenAI model, prompt, and conversation memory
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Initialize session state to store user details and chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []
if "age" not in st.session_state:
    st.session_state.age = ""
if "gender" not in st.session_state:
    st.session_state.gender = ""

# Collect user demographic details (age and gender)
if not st.session_state.age or not st.session_state.gender:
    st.write("Please enter your demographic details to proceed.")
    
    st.session_state.age = st.text_input("Please enter your age:", value=st.session_state.age)
    st.session_state.gender = st.selectbox("Please select your gender:", ["", "Male", "Female", "Other"], index=0)

# Ensure both age and gender are provided before continuing
if st.session_state.age and st.session_state.gender and st.session_state.gender != "":

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field for user symptoms or follow-up responses
    if user_input := st.chat_input("Please describe your gastrointestinal symptoms or answer the assistant's question:"):

        # Combine demographic details and symptoms
        combined_input = f"age: {st.session_state.age}, gender: {st.session_state.gender}, symptoms: {user_input}"

        # Store and display the user's message
        st.session_state.messages.append({"role": "user", "content": combined_input})
        with st.chat_message("user"):
            st.markdown(combined_input)

        # Pass the combined user details and conversation history to the LLMChain
        response = chain.invoke({
            "input": combined_input,
            "history": "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])  # Ensure full history is passed
        })

        # Display the AI assistant's response
        assistant_response = response['text']
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Store the assistant's response in session state
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

else:
    st.info("Please fill in both your age and gender to proceed.")

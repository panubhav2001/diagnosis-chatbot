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
    "Please enter your details and symptoms to begin. Please enter all your symptoms and the assistant will provide triage and diagnosis."
    "PLEASE REMEMBER THAT THE DIAGNOSIS MAY BE WRONG. IN CASE OF SEVERE HEALTH ISSUES, KINDLY CONSULT A DOCTOR."
)

# Template for the updated prompt that includes demographic details
prompt_template = """
You are a virtual health assistant specializing in Gastroenterology. A patient is describing their symptoms in a single message.

- The patient has reported: {input}.
- Your task is to:
    1. Collect all symptoms in one go, asking the patient to provide a complete description of their symptoms in a single message.
    2. Once the symptoms are provided, analyze them and diagnose the most likely gastrointestinal condition (e.g., GERD, IBS, Crohn's disease).
    3. Avoid repetitive or unnecessary follow-up questions by using the provided information effectively.
    4. Based on the symptoms, immediately provide a diagnosis for the condition that most closely matches the symptoms.
    5. Offer triage advice, informing the patient whether they need urgent medical care or a regular doctor's visit.
    6. Provide an action plan, including possible lifestyle changes, dietary recommendations, or over-the-counter medications if relevant.
    7. Ensure the response includes both the diagnosis and a clear recommendation for the next steps the patient should take.
    8. After you make the diagnosis, ask if you can help them with anything else. If the users says yes then start over.

- Use the previous conversation history: {history} to ensure that no information is repeated and the diagnosis and triage are personalized to the patient's context.
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

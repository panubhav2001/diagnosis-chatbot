import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.vectorstores.cassandra import Cassandra
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import cassio
import os

# Load environment variables
load_dotenv()

# Load API key and model details
openai_api_key = st.secrets["OPENAI_API_KEY"]
model_name = st.secrets["FINE_TUNED_MODEL"]

# Astra DB connection setup
app_token = st.secrets['ASTRA_DB_APPLICATION_TOKEN']
db_id = st.secrets['ASTRA_DB_ID']
cassio.init(token=app_token, database_id=db_id)

# Initialize embeddings and vector store for Astra DB
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
astra_vector_store = Cassandra(embedding=embeddings, table_name="chatbot_embeddings")

# Initialize the OpenAI model with controlled temperature
llm = ChatOpenAI(
    api_key=openai_api_key,
    model=model_name,
    temperature=0.15,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# Title and description of the chatbot with enhanced formatting
st.title("🩺 Gastroenterology Symptom Checker Chatbot")
st.write(
    "This chatbot assists in identifying gastrointestinal conditions based on the symptoms you provide. "
    "Please enter your symptoms and follow the instructions.\n"
    "⚠️ **Note**: This tool is not a substitute for professional medical advice. In case of severe symptoms, consult a doctor."
)

# Updated prompt template with follow-up questions for diagnosis
prompt_template = """
You are a virtual health assistant specializing in Gastroenterology. A patient is describing their symptoms.

- The patient has reported: {input_with_data}.
- Based on relevant past interactions and the current symptoms, your task is to:
    1. Immediately identify potential gastrointestinal conditions related to the reported symptoms.
    2. Diagnose the most likely gastrointestinal condition(s).
    3. Provide triage advice based on symptom severity:
        - For mild symptoms, suggest home care or lifestyle changes.
        - For moderate symptoms, recommend a doctor's visit.
        - For severe symptoms, advise urgent medical attention.
    4. Offer a detailed action plan including lifestyle changes and medications if relevant.
    5. Summarize the diagnosis and triage at the end of each response.
"""

# Create a LangChain PromptTemplate
prompt = PromptTemplate(input_variables=["input_with_data", "history"], template=prompt_template)

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
    st.write("### Please enter your demographic details to proceed.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.age = st.text_input("Enter your age:", value=st.session_state.age)
    with col2:
        st.session_state.gender = st.selectbox("Select your gender:", ["", "Male", "Female", "Other"], index=0)

# Ensure both age and gender are provided before continuing
if st.session_state.age and st.session_state.gender and st.session_state.gender != "":
    
    # Display previous chat messages with color differentiation using native Streamlit methods
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.write(f"**Assistant:** {message['content']}", key=message['content'])
        else:
            st.write(f"**User:** {message['content']}", key=message['content'])

    # Input field for user symptoms or follow-up responses
    if user_input := st.chat_input("Please describe your gastrointestinal symptoms or answer the assistant's question:"):

        # Combine demographic details and symptoms
        combined_input = f"age: {st.session_state.age}, gender: {st.session_state.gender}, symptoms: {user_input}"

        # Store and display the user's message
        st.session_state.messages.append({"role": "user", "content": combined_input})
        st.write(f"**User:** {combined_input}")

        # Spinner while waiting for AI response
        with st.spinner('Processing your input...'):
            # Retrieve relevant conversation history from Astra DB using vector search
            relevant_docs = astra_vector_store.similarity_search(combined_input, k=3)

            # If no relevant documents found, provide an empty string for retrieved_data
            if relevant_docs:
                retrieved_data = "\n".join([doc.page_content for doc in relevant_docs])
            else:
                retrieved_data = ""

            # Combine input and retrieved_data into one key
            input_with_data = f"{combined_input}\nRetrieved data: {retrieved_data}"

            # Store new user input in Astra DB
            astra_vector_store.add_documents([Document(page_content=combined_input)])

            # Pass the combined input, conversation history, and retrieved data to the LLMChain
            response = chain.invoke({
                "input_with_data": input_with_data,  
                "history": "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            })

            # Display the AI assistant's response
            assistant_response = response['text']
            st.write(f"**Assistant:** {assistant_response}")

            # Store the assistant's response in session state and Astra DB
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            astra_vector_store.add_documents([Document(page_content=assistant_response)])

else:
    st.info("Please fill in both your age and gender to proceed.")

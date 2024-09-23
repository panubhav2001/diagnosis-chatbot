# Gastroenterology Symptom Checker

## Project Overview
The **Gastroenterology Symptom Checker** is a web-based chatbot designed to assist users in diagnosing gastrointestinal issues based on the symptoms they provide. Built using **Streamlit** and **LangChain**, this application leverages natural language processing to deliver accurate medical advice and triage recommendations.

## Key Features
- **Symptom Analysis**: Users can describe their symptoms, and the chatbot analyzes this information to suggest potential gastrointestinal conditions.
- **Contextual Data Retrieval**: Integrates **Astra DB** with **OpenAI embeddings** for effective vector search and retrieval of relevant medical data.
- **Document Management**: Utilizes **LangChain Documents** to store and manage pertinent details for informed recommendations.
- **Memory-Driven Conversations**: Implements a custom **LLMChain with memory** to maintain the context of the conversation and provide real-time symptom analysis.
- **Focused Responses**: Logic is included to ensure responses are based solely on current symptoms, enhancing diagnostic accuracy.


## Technologies Used
- **Streamlit**: For creating the web interface.
- **LangChain**: For managing language models and conversation memory.
- **Astra DB**: For document storage and vector search capabilities.
- **OpenAI Embeddings**: For effective semantic understanding of user inputs.

## Getting Started
To run this project locally:
1. Clone the repository.
2. Install the required dependencies.
3. Set up the environment variables for OpenAI and Astra DB.
4. Run the Streamlit app.

## Note
This tool is intended for informational purposes only and is not a substitute for professional medical advice. In case of severe symptoms, consult a healthcare professional.


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diagnosis-chatbot-lgtpbzfrtffg2duvntwv5i.streamlit.app/)



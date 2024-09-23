import cassio
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores.cassandra import Cassandra
from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document  # Import Document class
from tqdm import tqdm  # Import tqdm for progress bar
load_dotenv()
from documents import docs  

embeddings = OpenAIEmbeddings()

# Connection to the AstraDB
app_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
db_id = os.getenv('ASTRA_DB_ID')

cassio.init(token=app_token, database_id=db_id)

def create_db_instance(vector_embeddings):
    astra_vector_store = Cassandra(
        embedding=vector_embeddings,
        table_name="chatbot_embeddings",  # Ensure this table exists or create it
        session=None,
        keyspace=None
    )
    return astra_vector_store

astra_vector_store = create_db_instance(embeddings)

if __name__ == '__main__':
    # Initialize tqdm progress bar
    with tqdm(total=len(docs), desc="Inserting documents", unit="doc") as pbar:
        for doc in docs:
            astra_vector_store.add_documents([doc])  # Upload each document one by one
            pbar.update(1)  # Update the progress bar after each document is added
    
    # Print the number of documents inserted
    print(f"Inserted {len(docs)} documents.")
    
    # Create an index wrapper around the vector store
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    
    # Set up retriever from vector store
    retriever = astra_vector_store.as_retriever()
    
    # Run a query
    result = retriever.get_relevant_documents("What is gastroenterology?")
    for doc in result:
        print(doc.page_content)

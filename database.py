import cassio
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores.cassandra import Cassandra
from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import chunking module
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

# Define text chunking parameters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Choose based on document size & retrieval needs
    chunk_overlap=100
)

if __name__ == '__main__':
    chunked_docs = []
    
    # Split each document into chunks before inserting into the DB
    for doc in docs:
        split_chunks = text_splitter.split_text(doc.page_content)
        
        # Convert chunks into Document objects while retaining metadata
        chunked_docs.extend([
            Document(page_content=chunk, metadata=doc.metadata)
            for chunk in split_chunks
        ])

    # Insert chunked documents into the vector store
    with tqdm(total=len(chunked_docs), desc="Inserting documents", unit="doc") as pbar:
        for chunk in chunked_docs:
            astra_vector_store.add_documents([chunk])  # Upload each chunk one by one
            pbar.update(1)  # Update the progress bar

    print(f"Inserted {len(chunked_docs)} document chunks.")

    # Create an index wrapper around the vector store
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

    # Set up retriever from vector store
    retriever = astra_vector_store.as_retriever()

    # Run a query
    result = retriever.get_relevant_documents("What is gastroenterology?")
    for doc in result:
        print(doc.page_content)

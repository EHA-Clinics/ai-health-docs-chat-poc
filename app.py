from dotenv import load_dotenv
import os
import pickle
from datetime import datetime
import streamlit as st
from langchain.llms  import OpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
API_KEY = os.environ.get('API_KEY')
llm = OpenAI(temperature=0.9, openai_api_key=API_KEY)
VECTOR_STORE_FILE= "knowledge_base.pkl"
VECTOR_STORE_FILE_PATH = f"./{VECTOR_STORE_FILE}"

# set up streamlit page and elements
st.set_page_config(page_title="Icon Test Main", layout="wide", menu_items=None)
st.title = "Community Health Q & A"
prompt = st.text_input("Please  type your question here")

def load_documents():
    '''Load documents from directory'''
    loader = DirectoryLoader('./hesperian', glob="**/*.pdf", use_multithreading=True)
    docs = loader.load()
    return docs

def split_documents(docs):
    '''split loaded documents into chunks'''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    documents = text_splitter.split_documents(docs)
    return documents

def create_vector_embeddings(documents):
    '''create vector embeddings'''
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def store_vector_embeddings(vector_store):
    '''store vector  embeddings on file system'''
    with open(VECTOR_STORE_FILE, 'wb') as f:
        pickle.dump(vector_store, f)

def get_vector_embeddings():
    '''store vector embeddings on file system'''
    vector_store = None
    if not os.path.isfile(VECTOR_STORE_FILE_PATH):
        return vector_store
    try:
        with open(VECTOR_STORE_FILE, 'rb') as f:
            vector_store = pickle.load(f)
    except IOError as e:
        print(e)
    return vector_store

# if the user enters a question
if prompt:
    vector_store = get_vector_embeddings()
    if vector_store is None:
        # load PDFs from directory, create vector embeddings and pickle
        documents = load_documents()
        document_chunks = split_documents(documents)
        vector_store = create_vector_embeddings(document_chunks)
        store_vector_embeddings(vector_store)

    docs = vector_store.similarity_search(prompt)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=prompt)

    st.write(response)


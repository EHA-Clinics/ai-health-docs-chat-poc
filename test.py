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
    with open('knowledge_base.pkl', 'wb') as f:
        pickle.dump(vector_store, f)


if prompt:
    st.write(f"started loading docs: {datetime.now()}")
    documents = load_documents()
    st.write(f"ended loading docs: {datetime.now()}")

    st.write(f"started splitting docs: {datetime.now()}")
    document_chunks = split_documents(documents)
    st.write(f"ended splitting docs: {datetime.now()}")

    st.write(f"started creating embeddings: {datetime.now()}")
    vector_store = create_vector_embeddings(document_chunks)
    st.write(f"ended creating embeddings: {datetime.now()}")

    st.write(f"started similarity search: {datetime.now()}")
    docs = vector_store.similarity_search(prompt)
    st.write(f"ended similarity search: {datetime.now()}")

    st.write(f"started loading qa chain: {datetime.now()}")
    chain = load_qa_chain(llm, chain_type="stuff")
    st.write(f"ended loading qa chain: {datetime.now()}")

    st.write(f"ended chhain run chain: {datetime.now()}")
    response = chain.run(input_documents=docs, question=prompt)
    st.write(f"ended chhain run chain: {datetime.now()}")

    st.write(response)


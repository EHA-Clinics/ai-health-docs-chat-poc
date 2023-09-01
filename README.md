# Document Chat based AI Project
This project is a simple proof-of-concept that uses AI tools to answer questions about information loaded from PDF files. 
The project uses:
- Langchain and OpenAI to load PDF documents from a directory.
- Facebook AI Similarity Search (FAISS) library to search vector embeddings.
- Streamlit to create text inputs, collect data from the UI and display responses.

# How to run the project (short version using Docker)
- Clone the repository
- Run `docker-compose up --build`
- Open a browser window and navigate to `http://localhost:8501`

# How to run the project (Long version without Docker)
- Clone the repository
- Create a virtual environment (The project uses python 3.11)
- Activate the virtual environment
- Run `pip install -r requirements.txt`
- Run `streamlit run app.py`
- Open a browser window and navigate to `http://localhost:8501`

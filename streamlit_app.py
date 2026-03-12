import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

st.title("📄 PDF AI Chatbot")

st.write("Ask questions from your PDFs")

# Load PDFs
@st.cache_resource
def load_vector_db():
    documents = []
    pdf_folder = "pdfs"

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    # embeddings = HuggingFaceEmbeddings()
    embeddings = HuggingFaceEmbeddings()

    vector_db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="db"
    )
    print(vector_db)

    return vector_db

vector_db = load_vector_db()

# Load LLM
llm = Ollama(model="mistral")

# User input
query = st.text_input("Ask a question about your PDFs")

if query:

    docs = vector_db.similarity_search(query, k=1)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response)
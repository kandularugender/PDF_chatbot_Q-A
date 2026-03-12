import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Load PDFs
documents = []

pdf_folder = "pdfs"

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, file))
        documents.extend(loader.load())

print("PDFs loaded")

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Text split into chunks")

# Create embeddings
embeddings = HuggingFaceEmbeddings()

# Create vector database
vector_db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="db"
)

print("Vector database created")

# Load LLM
llm = Ollama(model="mistral")

# Question loop
while True:
    query = input("\nAsk a question (type 'exit' to quit): ")

    if query == "exit":
        break

    docs = vector_db.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:", response)
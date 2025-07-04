import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import pickle

# 1. Load and extract text from PDFs
def load_pdfs_from_folder(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    all_text += page.get_text()
    return all_text

# 2. Split into chunks
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])

# 3. Create or load FAISS index
def create_faiss_index(documents, embeddings, save_path="faiss_index"):
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(save_path)
    return db

def load_faiss_index(embeddings, path="faiss_index"):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# 4. Embedder
def get_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

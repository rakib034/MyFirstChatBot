import streamlit as st
from utils import load_pdfs_from_folder, chunk_text, create_faiss_index, load_faiss_index, get_embedder
from groq import Groq

# Set your Groq API key
GROQ_API_KEY = "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"
MODEL_NAME = "gemma2-9b-it"  # or use gemma-7b, llama3-8b, etc.

client = Groq(api_key=GROQ_API_KEY)

# Initialize
st.title("💬 PDF Chatbot (Groq + FAISS)")
st.write("Ask questions based on your PDF knowledge base.")

# Load PDF knowledge base
if "vectorstore" not in st.session_state:
    with st.spinner("Loading documents and building vector store..."):
        text = load_pdfs_from_folder("pdfs")
        documents = chunk_text(text)
        embedder = get_embedder()
        db = create_faiss_index(documents, embedder)  # or load_faiss_index(embedder)
        st.session_state.vectorstore = db
        st.session_state.embedder = embedder

# Ask a question
query = st.text_input("Ask a question:")
if query:
    db = st.session_state.vectorstore
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the following question using the context below. 
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
"""

    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        st.write(response.choices[0].message.content)


from dotenv import load_dotenv
load_dotenv()

import os

print("✅ .env loaded successfully!")
print("GOOGLE_API_KEY found:", "YES" if os.getenv("GOOGLE_API_KEY") else "NO")


import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# --- Configuration ---

BOOKS_DIR = "./books"
VECTOR_DB_PATH = "./anat_vector_db"

def run_ingestion():
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    if not os.path.exists(BOOKS_DIR):
        os.makedirs(BOOKS_DIR)
        print(f"Created {BOOKS_DIR} folder. Please add PDFs and run again.")
        return

    print("📚 Loading  PDFs...")
    loader = DirectoryLoader(BOOKS_DIR, glob="./*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("❌ No PDFs found in the 'books' folder.")
        return

    print(f"✂️ Splitting {len(documents)} pages into semantic chunks...")
    # Recursive splitting maintains medical context better than simple character counts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    texts = text_splitter.split_documents(documents)

    print(f"🧬 Generating embeddings for {len(texts)} segments...")


    
    # Create and persist the vector database
    vector_db = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=VECTOR_DB_PATH
    )
    
    print(f"✅ Success! Knowledge base created at {VECTOR_DB_PATH}")

if __name__ == "__main__":
    run_ingestion()

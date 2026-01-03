import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

def run_ingestion():
    source_dir = "data/Syllabi"
    db_dir = "db"
    
    if not os.path.exists(source_dir):
        print(f"Error: Folder {source_dir} not found.")
        return

    all_docs = []
    
    for file in os.listdir(source_dir):
        if file.endswith(".pdf"):
            print(f"Loading {file}...")
            loader = PyPDFLoader(os.path.join(source_dir, file))
            all_docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} knowledge chunks.")
    print("Generating embeddings and saving to vault...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )
    
    print(f"Success! Knowledge vault created at /{db_dir}")

if __name__ == "__main__":
    run_ingestion()
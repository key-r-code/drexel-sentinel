import os
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

def test_query():
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

    db = Chroma(persist_directory="db", embedding_function=embeddings)
    

    query = "When is the midterm exam and what is its weight?"
    print(f"\nQuestion: {query}")
    

    docs = db.similarity_search(query, k=2)
    
    print("\n--- Sentinel Findings ---")
    for i, doc in enumerate(docs):
        print(f"\nMatch {i+1} from {doc.metadata.get('source')}:")
        print(doc.page_content)

if __name__ == "__main__":
    test_query();
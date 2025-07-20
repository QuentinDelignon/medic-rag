from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OllamaEmbeddings(model="qwen3:0.6b")
text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="medic-rag",
    connection=f"postgresql+psycopg://{os.getenv('DB_URI')}",
)


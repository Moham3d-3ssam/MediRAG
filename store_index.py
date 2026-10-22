from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# Load env Values
load_dotenv()


# Set Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY


# Handle Data
extracted_data = load_pdf_files(data="data/")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(minimal_docs)


# Download Embedding Model
embeddings = download_embeddings()
embedding_dimension = embeddings.client.get_sentence_embedding_dimension()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(pinecone_api_key)


# Create Index
index_name = "medical-rag"

if not pc.has_index(index_name):
  pc.create_index(
    name=index_name,
    dimension=embedding_dimension,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
  )

index = pc.Index(index_name)


# Store Vector Database
docsearch = PineconeVectorStore.from_documents(
  documents=text_chunks,
  embedding=embeddings,
  index_name=index_name
)

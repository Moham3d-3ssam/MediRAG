from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


# Filter To Minimal Docs
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
  """
  Given a list of document objects, return a new list of Document objects
  containing only the original page_content and 'source' in metadata
  """
  minimal_docs: List[Document] = []
  for doc in docs:
    src = doc.metadata.get("source")
    minimal_docs.append(
      Document(
        page_content=doc.page_content,
        metadata={"source": src}
      )
    )
  
  return minimal_docs


# Split Documents Into Smaller Chunks
def text_split(minimal_docs):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
  )
  
  text_chunks = text_splitter.split_documents(minimal_docs)
  return text_chunks


# Download Embedding Model
def download_embeddings():
  """
  Download and return HuggingFace Embeddings Model
  """
  model_name = "sentence-transformers/all-MiniLM-L6-v2"
  embeddings = HuggingFaceEmbeddings(
    model_name = model_name
  )
  
  return embeddings


# Extract Text From PDF Files
def load_pdf_files(data):
  loader = DirectoryLoader(
    data,
    glob="*.pdf",
    loader_cls=PyPDFLoader
  )
  
  documents = loader.load()
  
  return documents

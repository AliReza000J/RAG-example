import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_community.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

connection_string = os.getenv("DATABASE_URL")

loader = DirectoryLoader(
    os.path.abspath("../pdf-docs"),
    glob="**/*.pdf",
    use_multithreading=True,
    show_progress=True,
    max_concurrency=50,
    loader_cls=UnstructuredPDFLoader,
)

docs = loader.load()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 8, "normalize_embeddings": True}
)

text_splitter = SemanticChunker(
    embeddings=embeddings
)

docs = [doc for doc in docs if doc]
chunks = text_splitter.split_documents(docs)

print(type(docs))          # should be list
print(type(docs[0]))       # should be <class 'langchain.schema.document.Document'> or similar
print(docs[0].page_content[:100])

PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_chunks",
    connection_string=connection_string,
    pre_delete_collection=True,
    distance_strategy=DistanceStrategy.COSINE,
    use_jsonb=True,
)

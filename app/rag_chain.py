import os
from operator import itemgetter
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

connection_string = os.getenv("DATABASE_URL")

emb = HuggingFaceEmbeddings(
    model_name="Alibaba-NLP/gte-base-en-v1.5",
    model_kwargs={"trust_remote_code": True}
)

vector_store = PGVector(
    collection_name="rag_chunks",
    connection_string=connection_string,
    embedding_function=emb
)

template = """
Answer given the following context:
{context}

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# init model
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

class RagInput(TypedDict):
    question: str


final_chain = (
    {
    "context": (itemgetter("question") | vector_store.as_retriever()),
    "question": itemgetter("question")
    }
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
).with_types(input_type=RagInput)
import os
from operator import itemgetter
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel


load_dotenv()

connection_string = os.getenv("DATABASE_URL")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 8, "normalize_embeddings": True}
)

vector_store = PGVector(
    collection_name="rag_chunks",
    connection_string=connection_string,
    embedding_function=embeddings
)

template = """
Answer given the following context and if you don't have the answer just 'say hmm... DON'T KNOW!':
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
        RunnableParallel(
            context=(itemgetter("question") | vector_store.as_retriever()),
            question=itemgetter("question")
        ) |
        RunnableParallel(
            answer=(ANSWER_PROMPT | llm),
            docs=itemgetter("context")
        )
).with_types(input_type=RagInput)
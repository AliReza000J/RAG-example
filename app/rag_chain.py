import os
from operator import itemgetter
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import get_buffer_string

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
Answer the given following context and if you don't know the answer just 'say HMM... DON'T KNOW!':
{context}

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# init model
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

class RagInput(TypedDict):
    question: str

multiquery = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(),
    llm=llm,
)

chain = (
        RunnableParallel(
            context=(itemgetter("question") | multiquery),
            question=itemgetter("question")
        ) |
        RunnableParallel(
            answer=(ANSWER_PROMPT | llm),
            docs=itemgetter("context")
        )
).with_types(input_type=RagInput)

postgres_memory_url = os.getenv("MEMORY_URL")

get_session_history = lambda session_id: SQLChatMessageHistory(
    connection_string=postgres_memory_url,
    session_id=session_id,
)

template_with_history = """
Given the following conversation and a follow up question, rephrase the follow up
question to be a standalone questionm in its original language

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

standalone_question_prompt = PromptTemplate.from_template(template_with_history)

standalone_question_mini_chain = RunnableParallel(
    question=RunnableParallel(
        question=RunnablePassthrough(),
        chat_history=lambda x:get_buffer_string(
            x['chat_history']
        )
    )
    | standalone_question_prompt
    | llm
    | StrOutputParser()
)

final_chain = RunnableWithMessageHistory(
    runnable=standalone_question_mini_chain | chain,
    input_messages_key="question",
    history_messages_key="chat_history",
    output_messages_key="answer",
    get_session_history=get_session_history,
)
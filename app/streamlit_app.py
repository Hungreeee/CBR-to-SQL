import sys, os
sys.dont_write_bytecode = True

import csv

import torch
torch.classes.__path__ = []

import time
from dotenv import load_dotenv

import streamlit as st
import openai
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

import chatbot_verbosity as chatbot_verbosity

from modules.generator import OllamaGenerator
from modules.retriever import QdrantRetriever
from modules.embedder import HuggingFaceEmbedder
from modules.rag_pipeline import RAGPipeline

load_dotenv()


CHAT_LOG_FILE = "chat_logs.csv"

welcome_message = """
    #### Introduction 🚀

    The system is a RAG pipeline designed to assist patients in managing their health more effectively.

    The idea is to use a similarity retriever to identify the most relevant documents given your query.
    This data is then augmented into an LLM generator for question answering. 

    #### Getting started 🛠️

    1. To set up, please add your API key. 🔑 
    2. Type in your query. 💬
"""

st.set_page_config(page_title="Healthcare Chatbot")
st.title("Healthcare Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content=welcome_message)]

# if "df" not in st.session_state:
#     st.session_state.df = pd.read_csv(DATA_PATH)

# if "embedding_model" not in st.session_state:
#     st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

# if "rag_pipeline" not in st.session_state:
#     vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
#     st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

if "document_list" not in st.session_state:
    st.session_state.document_list = []


def initialize_chat_log():
    if not os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "user_message", "ai_response"])


def log_chat(user_message, ai_response):
    with open(CHAT_LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), user_message, ai_response])


def check_openai_api_key(api_key: str):
    return True
    openai.api_key = api_key
    try:
        _ = openai.chat.completions.create(
        model="gpt-4o-mini",  # Use a model you have access to
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=3
        )
        return True
    except openai.AuthenticationError as e:
        return False
    else:
        return True
  
  
def check_model_name(model_name: str, api_key: str):
    return True
    openai.api_key = api_key
    model_list = [model.id for model in openai.models.list()]
    return True if model_name in model_list else False


def clear_message():
    st.session_state.document_list = []
    st.session_state.chat_history = [AIMessage(content=welcome_message)]


user_query = st.chat_input("Type your message here...")

with st.sidebar:
    st.markdown("# Control Panel")

    st.text_input("OpenAI's API Key", type="password", key="api_key")
    st.text_input("GPT Model", "gpt-4o-mini", key="gpt_selection")
    st.button("Clear conversation", on_click=clear_message)


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
    else:
        with st.chat_message("AI"):
            message[0].render(*message[1:])


if not st.session_state.api_key:
    st.info("Please add your API key to continue. Learn more about [API keys](https://platform.openai.com/api-keys).")
    st.stop()

if not check_openai_api_key(st.session_state.api_key):
    st.error("The API key is incorrect. Please set a valid OpenAI API key to continue. Learn more about [API keys](https://platform.openai.com/api-keys).")
    st.stop()

if not check_model_name(st.session_state.gpt_selection, st.session_state.api_key):
    st.error("The model you specified does not exist. Learn more about [OpenAI models](https://platform.openai.com/docs/models).")
    st.stop()


embedder = HuggingFaceEmbedder()
retriever = QdrantRetriever(embedder=embedder)
llm_generator = OllamaGenerator()

rag_pipeline = RAGPipeline(
    retriever=retriever, 
    llm_generator=llm_generator,
)

initialize_chat_log()

if user_query is not None and user_query != "":
    with st.chat_message("Human"):
        st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("AI"):
        start = time.time()
        with st.spinner("Generating answers..."):
            stream_message, document_list = rag_pipeline.query(user_query, stream=True)
            st.session_state.document_list = document_list
        end = time.time()

        response = st.write_stream(stream_message)
        
        retriever_message = chatbot_verbosity
        retriever_message.render(document_list, end-start)

        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.chat_history.append((retriever_message, document_list, end-start))

        log_chat(user_query, response)
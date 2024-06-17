# from navigation import make_sidebar
# import streamlit as st

# make_sidebar()

# st.write(
#     """
# # 🔓 Secret Company Stuff

# This is a secret page that only logged-in users can see.

# Don't tell anyone.

# For real.

# """
# )
# from navigation import make_sidebar
# import streamlit as st

# make_sidebar()

# st.write(
#     """
# # 🔓 Secret Company Stuff

# This is a secret page that only logged-in users can see.

# Don't tell anyone.

# For real.

# """
# )
# %%writefile my_app3.py
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:55:20 2024

@author: prashant.kumar
"""

# tools
import glob
from tempfile import NamedTemporaryFile
import logging
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
# ollama
import ollama
from langchain.embeddings import HuggingFaceEmbeddings
# llamaindex
from langchain.llms import Ollama
from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer 
from time import time    
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA ,ConversationalRetrievalChain
from langchain.vectorstores import Chroma
import streamlit as st
import glob
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain import PromptTemplate
# import streamlit_authenticator as stauth
# import yaml
# from yaml.loader import SafeLoader
# with open('Mycred.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)
from navigation import *

def response_generator(stream):
    """Generator that yields chunks of data from a stream response.
    Args:
        stream: The stream object from which to read data chunks.
    Yields:
        bytes: The next chunk of data from the stream response.
    """
    for chunk in stream.response_gen:
        yield chunk

@st.cache_resource(show_spinner=True)
def load_data(document, model_name:str,chunk:int,overlap:int):
    """Loads and indexes Streamlit documentation using Ollama and Llamaindex.

    This function takes a model name as input and performs the following actions:

    1. Ollama Initialization: Initializes an Ollama instance using the provided model name. Ollama is a library that facilitates communication with large language models (LLMs).
    2. Data Ingestion: Reads the Streamlit documentation (assumed to be a PDF file) using the SimpleDirectoryReader class.
    3. Text Splitting and Embedding: Splits the loaded documents into sentences using the SentenceSplitter class and generates embeddings for each sentence using the HuggingFaceEmbedding model.
    4. Service Context Creation: Creates a ServiceContext object that holds all the necessary components for processing the data, including the Ollama instance, embedding model, text splitter, and a system prompt for the LLM.
    5. VectorStore Indexing: Creates a VectorStoreIndex instance from the processed documents and the service context. VectorStore is a library for efficient searching of high-dimensional vectors.

    Args:
        # docs_path  (str): Path of the documents to query.
        model_name (str): The name of the LLM model to be used by Ollama.

    Returns:
        VectorStoreIndex: An instance of VectorStoreIndex containing the indexed documents and embeddings.
    """

    # llm
    llm = Ollama(model=model_name, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

    # data ingestion
    new_data = []
    pages=None
    
    with NamedTemporaryFile(dir='./', suffix='.pdf') as f:
        f.write(document.getbuffer())
        with st.spinner(text="Loading and indexing the Streamlit docs.."):
            # loading document
            data_path = "./"

            files = glob.glob(data_path+"*.pdf")

            
            pdf_load = PyPDFLoader(files[0])
            pages = pdf_load.load_and_split()
            



#             docs = SimpleDirectoryReader(".").load_data()

        # embeddings | query container
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk), chunk_overlap=int(overlap))

            # Split the loaded documents into smaller text chunks
            all_splits = text_splitter.split_documents(pages)
            print(all_splits)

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {"device": "cuda"}
            embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
#             embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2") # BAAI/bge-small-en-v1.5 | BAAI/bge-base-en-v1.5

            template = """Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use three sentences maximum and keep the answer as concise as possible.
            {context}
            Question: {question}
            Helpful Answer:"""
            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=template,
            )

            vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings,persist_directory="chroma_db")
            # Initialize a RetrievalQA object with specified settings
            retriever = vectordb.as_retriever()
            qa = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=retriever, 
                
    #                 verbose=True
            )

                    # indexing db
        #             index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return qa,llm
def main() -> None:
    """Controls the main chat application logic using Streamlit and Ollama.

    This function serves as the primary orchestrator of a chat application with the following tasks:

    1. Page Configuration: Sets up the Streamlit page's title, icon, layout, and sidebar using st.set_page_config.
    2. Model Selection: Manages model selection using st.selectbox and stores the chosen model in Streamlit's session state.
    3. Chat History Initialization: Initializes the chat history list in session state if it doesn't exist.
    4. Data Loading and Indexing: Calls the load_data function to create a VectorStoreIndex from the provided model name.
    5. Chat Engine Initialization: Initializes the chat engine using the VectorStoreIndex instance, enabling context-aware and streaming responses.
    6. Chat History Display: Iterates through the chat history messages and presents them using Streamlit's chat message components.
    7. User Input Handling:
          - Accepts user input through st.chat_input.
          - Appends the user's input to the chat history.
          - Displays the user's message in the chat interface.
    8. Chat Assistant Response Generation:
          - Uses the chat engine to generate a response to the user's prompt.
          - Displays the assistant's response in the chat interface, employing st.write_stream for streaming responses.
          - Appends the assistant's response to the chat history.

    Args:
        docs_path (str): Path of the documents to query.
    """

    st.set_page_config(page_title="Chatbot", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("Chat with documents 💬")
    
    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False
    if "chank_size" not in st.session_state:
        st.session_state.chank_size = 0
    if "overlap" not in st.session_state:
        st.session_state.overlap = 0
    
    with st.sidebar:
        st.image("./gspann-horizontal-hires.jpg", width=150)
        # model selection
        if "model" not in st.session_state:
            st.session_state["model"] = ""
        models = [model["name"] for model in ollama.list()["models"]]
        st.session_state["model"] = st.selectbox("Select a model", models)
        
        
        # llm
#         llm = Ollama(model=st.session_state["model"], request_timeout=30.0)

        # data ingestion
        document = st.file_uploader("Upload a PDF file to query", type=['pdf'], accept_multiple_files=False)
        ch = st.slider(" SELECT CUNCK SIZE", min_value=100, max_value=10000, value=500, step=1)
        ov = st.slider(" SELECT OVERLAP SIZE", min_value=10, max_value=1000, value=20, step=1)
        
        # file processing                
        if st.button('Process file'):
            st.session_state.overlap = ov
            st.session_state.chank_size = ch
            index,lodded_model = load_data(document, st.session_state["model"],st.session_state['chank_size'],st.session_state['overlap'])
            st.session_state.activate_chat = True
            

    if st.session_state.activate_chat == True:
        # initialize chat history                   
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # index = load_data(docs_path, st.session_state["model"])
        if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
            st.session_state.chat_engine = index
        if "lodded_model" not in st.session_state.keys(): # Initialize the chat engine
            st.session_state.lodded_model = lodded_model
        # display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # accept user input
        if prompt := st.chat_input("How I can help you?"):
            # add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # chat assistant
            if st.session_state.messages[-1]["role"] != "RAG":
#                 message_placeholder = st.empty()
                with st.chat_message("RAG"):
                    stream = st.session_state.chat_engine({"query":'consider yourself QA assistance and provide results only from the given context. '+prompt})
                    response = st.write(stream['result'])
                    st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            if st.session_state.messages[-1]["role"] != "LLM":
#                 message_placeholder = st.empty()
                with st.chat_message("LLM"):
                    llm_stream = st.session_state.lodded_model.invoke(prompt)
                    llm_response = st.write(llm_stream)
                    st.markdown(llm_response)
                st.session_state.messages.append({"role": "LLM", "content": llm_response})
                
    else:
        st.markdown("<span style='font-size:15px;'><b>Upload a PDF to start chatting</span>", unsafe_allow_html=True)

# if __name__=='__main__':
if st.session_state.get("logged_in", False):
    main()
elif get_current_page_name() != "streamlit_app":
        # If anyone tries to access a secret page without being logged in,
        # redirect them to the login page
        st.switch_page("streamlit_app.py")


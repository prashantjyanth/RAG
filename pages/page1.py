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


from time import time    

import streamlit as st
import glob

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
    st.session_state.activate_chat = True
    # if "activate_chat" not in st.session_state:
    #     st.session_state.activate_chat = False
    # if "chank_size" not in st.session_state:
    #     st.session_state.chank_size = 0
    # if "overlap" not in st.session_state:
    #     st.session_state.overlap = 0
    
    with st.sidebar:
        st.image("./gspann-horizontal-hires.jpg", width=150)
        # # model selection
        # if "model" not in st.session_state:
        #     st.session_state["model"] = ""
        # models = [model["name"] for model in ollama.list()["models"]]
        # st.session_state["model"] = st.selectbox("Select a model", models)
        
        
        # llm
#         llm = Ollama(model=st.session_state["model"], request_timeout=30.0)

        # data ingestion
        # document = st.file_uploader("Upload a PDF file to query", type=['pdf'], accept_multiple_files=False)
        # ch = st.slider(" SELECT CUNCK SIZE", min_value=100, max_value=10000, value=500, step=1)
        # ov = st.slider(" SELECT OVERLAP SIZE", min_value=10, max_value=1000, value=20, step=1)
        
        # file processing                
        # if st.button('Process file'):
        #     st.session_state.overlap = ov
        #     st.session_state.chank_size = ch
        #     index,lodded_model = load_data(document, st.session_state["model"],st.session_state['chank_size'],st.session_state['overlap'])
        #     st.session_state.activate_chat = True
            

    if st.session_state.activate_chat == True:
        # initialize chat history                   
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # index = load_data(docs_path, st.session_state["model"])
        # if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        #     st.session_state.chat_engine = index
        # if "lodded_model" not in st.session_state.keys(): # Initialize the chat engine
        #     st.session_state.lodded_model = lodded_model
        # # display chat messages from history on app rerun
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

#             # chat assistant
#             if st.session_state.messages[-1]["role"] != "RAG":
# #                 message_placeholder = st.empty()
#                 with st.chat_message("RAG"):
#                     stream = st.session_state.chat_engine({"query":'consider yourself QA assistance and provide results only from the given context. '+prompt})
#                     response = st.write(stream['result'])
#                     st.markdown(response)
                
#                 st.session_state.messages.append({"role": "assistant", "content": response})
#             if st.session_state.messages[-1]["role"] != "LLM":
# #                 message_placeholder = st.empty()
#                 with st.chat_message("LLM"):
#                     llm_stream = st.session_state.lodded_model.invoke(prompt)
#                     llm_response = st.write(llm_stream)
#                     st.markdown(llm_response)
#                 st.session_state.messages.append({"role": "LLM", "content": llm_response})
                
    else:
        st.markdown("<span style='font-size:15px;'><b>Upload a PDF to start chatting</span>", unsafe_allow_html=True)

# if __name__=='__main__':
if st.session_state.get("logged_in", False):
    main()
elif get_current_page_name() != "streamlit_app":
        # If anyone tries to access a secret page without being logged in,
        # redirect them to the login page
        st.switch_page("streamlit_app.py")


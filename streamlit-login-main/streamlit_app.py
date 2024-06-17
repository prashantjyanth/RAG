import streamlit as st
from time import sleep
from navigation import make_sidebar

# make_sidebar()

st.title("Welcome to GSPANN ChatBot")
with st.sidebar:
    st.image("./gspann-horizontal-hires.jpg", width=150)
# st.write("Please log in to continue (username `test`, password `test`).")
    option = st.selectbox(
        "How would you like to be contacted?",
        (("CHAT WITH AVALIBLE DOCUMENTS","UPLOAD NEW DOCUMENT")))
    if option == "UPLOAD NEW DOCUMENT":
        st.warning('To upload new doc Admin privilage required', icon="⚠️")
    st.session_state['option']=option
if st.session_state['option']=="UPLOAD NEW DOCUMENT":
    adminusername = st.text_input("Admin Username")
    adminpassword = st.text_input("Admin Password", type="password")

else:

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

if st.button("Log in", type="primary"):
    if st.session_state['option']=="CHAT WITH AVALIBLE DOCUMENTS":
        if username == "test" and password == "test" :
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            sleep(0.5)
            st.switch_page("pages/page1.py")
        else:
            st.error("Incorrect username or password")
    elif st.session_state['option']=="UPLOAD NEW DOCUMENT":
        if adminusername == "admin" and adminpassword == "admin" :
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            sleep(0.5)
            st.switch_page("pages/page2.py")
        else:
            st.error("Incorrect username or password")

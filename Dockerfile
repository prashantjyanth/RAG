FROM python:3.12.0
ENV DEBIAN_FRONTEND=nonintercative

RUN mkdir -p /my_app_

WORKDIR /my_app_/
COPY requirements.txt /my_app_/
ENV QT_X11_NO_MITSHM=1
COPY streamlit_app.py /my_app_/
COPY navigation.py /my_app_/
COPY gspann-horizontal-hires.jpg /my_app_/
COPY pages /my_app_/
RUN pip3 install -r /my_app_/requirements.txt
EXPOSE 8686
RUN streamlit run /my_app_/streamlit_app.py server.port=8686 server.address=0.0.0.0


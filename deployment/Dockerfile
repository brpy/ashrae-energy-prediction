FROM ubuntu:20.04
COPY . /deploy
RUN apt update && apt upgrade
RUN apt install python3 python3-pip -y
RUN pip3 install -r deploy/requirements.txt
CMD python3 deploy/app.py

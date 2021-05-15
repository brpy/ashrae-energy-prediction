FROM ubuntu:20.04
COPY . /deploy
RUN sudo apt install python3 pip3 -y
RUN pip3 install -r deploy/requirements.txt
CMD python3 deploy/app.py

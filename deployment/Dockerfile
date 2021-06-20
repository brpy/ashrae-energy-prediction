FROM ubuntu:20.04
RUN apt update && apt upgrade -y
RUN apt install python3 python3-pip -y
RUN python3 -m pip install -U pip
COPY . /deploy
WORKDIR deploy
RUN python3 -m pip install -r requirements.txt
ENTRYPOINT ["python3", "app.py"]

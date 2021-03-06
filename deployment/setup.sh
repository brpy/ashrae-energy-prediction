#!/bin/sh
sudo apt update && sudo apt upgrade
sudo apt install python3 python3-pip
pip3 install -r requirements.txt
python3 app.py

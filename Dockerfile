FROM python:3.7-slim

COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

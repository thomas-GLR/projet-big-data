FROM python:3.10-slim

WORKDIR /app

COPY serving/api.py ./api.py
COPY serving/requirements.txt ./requirements.txt
COPY scripts ./scripts

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD uvicorn api:app --host 0.0.0.0 --port 8080

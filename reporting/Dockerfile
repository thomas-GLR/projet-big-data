FROM python:3.12-slim

WORKDIR /app

COPY project.py .
COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENV WORKSPACE_PATH=/app/evidently_workspace

CMD python project.py && evidently ui --workspace $WORKSPACE_PATH --host 0.0.0.0 --port 8082

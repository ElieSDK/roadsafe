FROM python:3.12.9
WORKDIR /prod
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT

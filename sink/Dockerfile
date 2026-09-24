FROM python:3.12-slim
RUN pip install --no-cache-dir "psycopg[binary]==3.2.13" confluent-kafka==2.15.1
COPY consumer_sink.py /app/consumer_sink.py
WORKDIR /app
CMD ["python", "consumer_sink.py"]

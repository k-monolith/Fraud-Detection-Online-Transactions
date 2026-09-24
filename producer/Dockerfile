FROM python:3.12-slim
RUN pip install --no-cache-dir pandas==2.3.3 numpy==2.0.2 confluent-kafka==2.15.1
COPY replay_producer.py /app/replay_producer.py
WORKDIR /app
CMD ["python", "replay_producer.py"]

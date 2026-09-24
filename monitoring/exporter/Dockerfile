FROM python:3.12-slim
RUN pip install --no-cache-dir confluent-kafka==2.15.1 prometheus-client==0.26.0
COPY drift.py metrics_exporter.py /app/
WORKDIR /app
CMD ["python", "metrics_exporter.py"]

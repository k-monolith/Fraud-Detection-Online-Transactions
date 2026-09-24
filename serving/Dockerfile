# serving/Dockerfile — Flink 2.1 + Python 3.12 + pinned ML stack.
# Python 3.12 and numpy 2.0.2 match the environment that produced
# artifacts/preprocessing.pkl; PyFlink 2.1 provides Python 3.12 wheels.
#
# Build from the repository root (copies artifacts/ into the image):
#   docker build -f serving/Dockerfile -t fraud-serving .
ARG FLINK_IMAGE=flink:2.1.0-java17
FROM ${FLINK_IMAGE}

USER root

# --- Python 3.12 (matches training's 3.12.13 minor version) -------------------
# Flink images are Ubuntu-based; install python3.12 from apt if the release has
# it (noble), otherwise via deadsnakes (jammy).
RUN set -eux; \
    apt-get update; \
    if apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev; then \
        echo "python3.12 from distribution"; \
    else \
        apt-get install -y --no-install-recommends software-properties-common gnupg; \
        add-apt-repository -y ppa:deadsnakes/ppa; \
        apt-get update; \
        apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev; \
    fi; \
    apt-get install -y --no-install-recommends build-essential libgomp1 curl; \
    rm -rf /var/lib/apt/lists/*

# Dedicated venv so system Python stays untouched.
RUN python3.12 -m venv /opt/pyenv
ENV PATH="/opt/pyenv/bin:${PATH}"

# --- Stage 1: PyFlink (brings apache-beam, pemja) ------------------------------
COPY requirements-serve.txt /tmp/requirements-serve.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir "apache-flink==2.1.0"

# --- Stage 2: pinned ML stack --------------------------------------------------
# numpy 2.0.2 and scikit-learn 1.6.1 match the training environment (pickle
# compatibility). `pip check` must come back clean.
RUN pip install --no-cache-dir -r /tmp/requirements-serve.txt \
 && pip check \
 && python -c "import numpy, pandas, sklearn, xgboost, lightgbm, catboost; \
print('numpy', numpy.__version__, '| pandas', pandas.__version__, \
      '| sklearn', sklearn.__version__)"

# --- Kafka connector jar (Java side) ------------------------------------------
# Connector line 5.0.0 targets Flink 2.1.x/2.2.x (4.0.x targets 2.0.x only)
ARG KAFKA_CONNECTOR_VERSION=5.0.0-2.1
RUN curl -fsSL -o /opt/flink/lib/flink-sql-connector-kafka.jar \
    "https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/${KAFKA_CONNECTOR_VERSION}/flink-sql-connector-kafka-${KAFKA_CONNECTOR_VERSION}.jar"

# --- Job code + artifacts ------------------------------------------------------
COPY serving/schemas.py serving/scoring_udf.py serving/flink_job.py /opt/serving/
COPY artifacts/ /opt/artifacts/
ENV PYTHONPATH=/opt/serving
ENV ARTIFACTS_DIR=/opt/artifacts

# Thread execution mode + interpreter paths for PyFlink.
RUN printf '%s\n' \
    "python.execution-mode: thread" \
    "python.executable: /opt/pyenv/bin/python" \
    "python.client.executable: /opt/pyenv/bin/python" \
    >> /opt/flink/conf/config.yaml || true

USER flink

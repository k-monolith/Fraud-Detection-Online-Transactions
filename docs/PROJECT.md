# Real-Time Financial Fraud Detection: Project Documentation

This document describes the system as implemented in this repository: the model, the streaming pipeline, persistence, monitoring, operation, testing, and the design decisions behind them. Figures come from the repository files named in each section. Day-to-day commands and troubleshooting are in [operations.md](operations.md).

## Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Model](#3-model)
4. [Model artifacts](#4-model-artifacts)
5. [Streaming pipeline](#5-streaming-pipeline)
6. [Persistence](#6-persistence)
7. [Monitoring](#7-monitoring)
8. [Running the project](#8-running-the-project)
9. [Testing and CI](#9-testing-and-ci)
10. [Reproducibility](#10-reproducibility)
11. [Design decisions](#11-design-decisions)
12. [Limitations](#12-limitations)
13. [Repository layout](#13-repository-layout)

## 1. Overview

The system scores card transactions one at a time as they arrive. A replay producer streams the unlabeled IEEE-CIS test transactions to Kafka. An Apache Flink job validates each event, applies the fitted preprocessing pipeline, runs a stacking ensemble and publishes one of three decisions: `ALERT`, `PASS` or `REVIEW`. A sink service writes the results and dead letters to PostgreSQL, while a metrics exporter, Prometheus and Grafana monitor the system and the model's behavior. The whole stack runs locally with Docker Compose.

The ensemble combines XGBoost, LightGBM and CatBoost base models through an XGBoost meta-learner, following the architecture of [arXiv:2505.10050](https://arxiv.org/abs/2505.10050).

| | |
|---|---|
| Dataset | [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle); the data files are not included in the repository |
| Training data | 590,540 labeled transactions (`train_transaction.csv` joined with `train_identity.csv`) |
| Held-out period | the most recent 20% of the timeline, 118,108 transactions |
| Held-out ROC-AUC | 0.918 |
| Held-out accuracy | 0.972 at the decision threshold |
| Decision threshold | 0.2883 |
| Model version | `20260921T144658Z` |
| Replay data | `test_transaction.csv` and `test_identity.csv` (unlabeled) |

Sources: `artifacts/manifest.json`, `artifacts/threshold.json`; accuracy from the holdout evaluation in `notebook/fraud-detection-training.ipynb`.

## 2. Architecture

```
                data/test_transaction.csv + data/test_identity.csv
                                      │
                                      ▼
                                 ┌──────────┐
                                 │ producer │
                                 └────┬─────┘
                                      │ transactions.raw
                                      ▼
                ┌──────────────────────────────────────────┐
                │ Flink job  fraud-detection-scoring       │
                │ validate → preprocess → score → decide   │
                └─────────┬──────────────────────┬─────────┘
                          │ transactions.scored  │ transactions.dlq
                          ▼                      ▼
              ┌──────────────────────┐  ┌───────────────────────────┐
              │ sink                 │  │ metrics-exporter          │
              │ reads scored + dlq   │  │ reads raw + scored + dlq  │
              └──────────┬───────────┘  └─────────────┬─────────────┘
                         ▼                            ▼
                  ┌────────────┐              ┌────────────┐
                  │ PostgreSQL │              │ Prometheus │ ◄── Flink metrics (port 9249)
                  └──────┬─────┘              └──────┬─────┘
                         └─────────────┬─────────────┘
                                       ▼
                                  ┌─────────┐
                                  │ Grafana │
                                  └─────────┘
```

`docker-compose.yml` defines eleven services:

| Service | Image or build | Role | Host port |
|---|---|---|---|
| `kafka` | `apache/kafka:3.9.0` | single broker in KRaft mode (broker and controller) | 29092 |
| `topics-init` | `apache/kafka:3.9.0` | creates the three topics, then exits | – |
| `jobmanager` | `fraud-serving` (`serving/Dockerfile`) | Flink JobManager and web UI | 8081 |
| `taskmanager` | `fraud-serving` | Flink TaskManager with 2 task slots; runs the scoring operator | – |
| `job-submit` | `fraud-serving` | submits `serving/flink_job.py` to the JobManager, then exits | – |
| `producer` | `producer/Dockerfile` | replays the Kaggle test files to `transactions.raw` | – |
| `sink` | `sink/Dockerfile` | writes scored events and dead letters to PostgreSQL | – |
| `postgres` | `postgres:17.4` | database `fraud`, schema from `sink/schema.sql` | 5432 |
| `metrics-exporter` | `monitoring/exporter/Dockerfile` | model-behavior metrics at `/metrics` | 8000 |
| `prometheus` | `prom/prometheus:v3.2.1` | scrapes metrics and evaluates alert rules | 9090 |
| `grafana` | `grafana/grafana:12.2.1` | dashboards over Prometheus and PostgreSQL | 3000 |

Kafka topics:

| Topic | Written by | Read by |
|---|---|---|
| `transactions.raw` | `producer` | Flink job (consumer group `fraud-scoring`), `metrics-exporter` |
| `transactions.scored` | Flink job | `sink` (consumer group `scored-sink`), `metrics-exporter` |
| `transactions.dlq` | Flink job | `sink`, `metrics-exporter` |

Automatic topic creation is disabled on the broker. `topics-init` creates each topic with 1 partition and replication factor 1. Services reach the broker at `kafka:9092`; tools on the host use `localhost:29092`.

## 3. Model

### 3.1 Ensemble

Each base model returns a fraud probability for the event. The meta-learner receives the three probabilities in the order XGBoost, LightGBM, CatBoost (`base_model_order_for_meta_features` in `threshold.json`) and returns the final fraud probability.

| Model | Trees | Max depth | Learning rate | Row subsample | Column subsample | Positive-class weight |
|---|---|---|---|---|---|---|
| XGBoost | 400 | 10 | 0.121 | 0.84 | 0.66 | 17.04 (`scale_pos_weight`) |
| LightGBM | 700 | 10 | 0.0156 | 0.63 | 0.69 | 15.20 (`scale_pos_weight`) |
| CatBoost | 600 | 9 | 0.0411 | 0.87 | – | 15.88 (class weight) |
| Meta-learner (XGBoost) | 200 | 2 | 0.1 | – | – | – |

Values come from `hyperparameters` and `meta_learner` in `manifest.json`. The manifest records the source of all three base-model parameter sets as `warm_start`: they are the `warm_start_params` of the training configuration, which also records a tuning setup of 30 Optuna trials and 5 cross-validation folds.

### 3.2 Input features

The model uses 30 of the 432 raw columns, ranked by importance from an XGBoost model trained on all features of the training split (`artifacts/features.json`). In rank order:

`C13`, `TransactionAmt`, `card1`, `C1`, `TransactionDT`, `V258`, `card6`, `card2`, `V70`, `C14`, `addr1`, `D1`, `P_emaildomain`, `C11`, `C5`, `C2`, `V294`, `D15`, `D4`, `D10`, `M5`, `card5`, `dist1`, `M6`, `D2`, `V91`, `C6`, `D11`, `D3`, `R_emaildomain`

Five of them are categorical: `card6`, `P_emaildomain`, `M5`, `M6` and `R_emaildomain`. `TransactionDT`, ranked fifth, is the dataset's time offset (see [Limitations](#12-limitations)).

### 3.3 Preprocessing

A single fitted scikit-learn object, `artifacts/preprocessing.pkl` (saved with scikit-learn 1.6.1), prepares every event:

| Columns | Count | Steps |
|---|---|---|
| Numeric | 401 | median imputation |
| Categorical | 31 | most-frequent imputation, then ordinal encoding; categories not seen in training map to -1 |

The object is a `ColumnTransformer` configured to return a pandas DataFrame, so the 30 model features are selected by name after the transform. Before it runs, the serving code builds a complete raw row following the data contract in `manifest.json`: every expected column is present (missing ones as null), numeric values are converted to float, and non-null categorical values are converted to strings.

### 3.4 Time-ordered split and threshold

The data is split in time on `TransactionDT` (`split` in `manifest.json`):

| Part | Rows | `TransactionDT` | Fraud rate |
|---|---|---|---|
| Training split | 472,432 (first 80% of the timeline) | ends at 12,192,842 | 3.51% |
| Core training rows | 425,188 (first 90% of the training split) | – | – |
| Validation slice | 47,244 (last 10% of the training split) | starts at 10,766,698 | 3.50% |
| Held-out period | 118,108 (last 20% of the timeline) | starts at 12,192,900 | 3.44% |

The decision threshold, 0.2883, is the F1-maximizing point on the precision-recall curve of the validation slice, scored by the stack trained on the core training rows before the final refit (`threshold.json`). On the held-out period the model reaches a ROC-AUC of 0.918 (`holdout_metrics` in the manifest).

### 3.5 Class imbalance

Fraud makes up about 3.5% of the transactions. The base models handle this with the positive-class weights listed in section 3.1. The recorded fraud rates of the validation slice (3.50%) and the held-out period (3.44%) match the dataset's natural rate, which shows that the evaluation data was not resampled.

## 4. Model artifacts

The serving code depends only on the contents of `artifacts/` (about 24 MB in total):

| File | Content | Format |
|---|---|---|
| `model_xgb.json` | XGBoost base model | XGBoost JSON |
| `model_lgbm.txt` | LightGBM base model | LightGBM text |
| `model_catboost.cbm` | CatBoost base model | CatBoost binary |
| `meta_learner.json` | XGBoost meta-learner | XGBoost JSON |
| `preprocessing.pkl` | fitted preprocessing object | joblib pickle |
| `features.json` | the 30 selected features in rank order and the selection method | JSON |
| `threshold.json` | decision threshold, how it was selected, base-model order for the meta-learner | JSON |
| `manifest.json` | model version and run metadata (below) | JSON |

`manifest.json` records the model version and creation time (UTC), the random seed (42), the library versions and hardware of the training run, the SHA-256 hashes of `train_transaction.csv` and `train_identity.csv`, the data contract (the 31 categorical columns, the number of numeric columns and the unseen-category code), the split sizes, boundaries and fraud rates, the training configuration, the selected features, the hyperparameters and their source, the meta-learner settings, the threshold and the held-out metrics.

`ScoringService` in `serving/scoring_udf.py` loads the bundle once per operator instance. Each model is loaded with its library's native loader, `preprocessing.pkl` with joblib, and the numeric columns are derived from the pipeline's input columns minus the categorical columns of the data contract.

The serving image copies `artifacts/` at build time, and the metrics exporter mounts it read-only. A new model therefore requires rebuilding the serving image (section 8.6). An optional file, `reference_distributions.json`, is used by the metrics exporter for drift detection (section 7.2); it is not included in the repository.

## 5. Streaming pipeline

### 5.1 Replay producer

`producer/replay_producer.py` reads `test_transaction.csv` in chunks of 20,000 rows and attaches the matching `test_identity.csv` record by `TransactionID`. It renames identity columns such as `id-01` to `id_01`, converts missing values to JSON `null`, and sends one JSON object per message to `transactions.raw`. No labels are sent.

The producer waits `1 / RATE_EPS` seconds after each message. With `LOOP` enabled, it starts again from the first row when the file ends. To exercise the dead-letter path, each message is cut to a third of its length with probability `MALFORMED_PPM` / 1,000,000, which makes it invalid JSON. Optional bursts send `BURST_SIZE` extra messages every `BURST_EVERY_S` seconds; burst messages are taken from the start of the file and are never malformed. The Kafka producer uses `linger.ms` 20, `batch.num.messages` 1000 and lz4 compression.

| Variable | Default | Meaning |
|---|---|---|
| `KAFKA_BOOTSTRAP` | `kafka:9092` | broker address |
| `TOPIC_RAW` | `transactions.raw` | output topic |
| `DATA_DIR` | `/data` | folder with the replay CSVs (mounted read-only from `./data`) |
| `RATE_EPS` | 50 | events per second |
| `LOOP` | true | restart from the first row when the file ends |
| `MALFORMED_PPM` | 200 | malformed events per million |
| `BURST_EVERY_S` | 0 | seconds between bursts (0 disables bursts) |
| `BURST_SIZE` | 500 | events per burst |

### 5.2 Flink job

`serving/flink_job.py` defines the job `fraud-detection-scoring`.

**Source.** A Kafka source reads `transactions.raw` with consumer group `fraud-scoring`, starting from the earliest offset. The job uses no watermarks or windows; each event is processed on its own.

**Validation.** `ValidateFunction` parses the message, requires a non-empty JSON object, trims whitespace from keys, renames `id-XX` keys to `id_XX`, and requires a usable `TransactionID`. Events that fail go to a side output written to `transactions.dlq` as `{"error": ..., "raw": ...}`, with the original payload cut to 65,536 characters. Bad input never stops the job.

| Dead-letter reason | Cause |
|---|---|
| `json_parse_error: <detail>` | the message is not valid JSON |
| `not_a_json_object` | valid JSON, but not a non-empty object |
| `missing_transaction_id` | no usable `TransactionID` |

**Scoring.** `ScoreFunction` (operator `stacking-ensemble-scoring`) creates a `ScoringService` in `open()`, so the models load once per operator instance. For each event it builds the complete raw row. Numeric values are converted to float, with empty strings, NaN and infinity becoming null. Non-null categorical values become trimmed strings, and unknown fields are ignored. It then applies the preprocessing object, selects the 30 features, computes the three base probabilities and passes them to the meta-learner.

**Decision.** A probability at or above the threshold gives `ALERT` (`is_fraud_pred` = 1); otherwise the decision is `PASS` (0). If any scoring step raises an error, the event is still published, as `REVIEW` with a null probability and prediction and the error message (up to 500 characters). A value that cannot be converted to a number does not block scoring: the column is treated as missing and a warning is attached. The possible warnings are `uncoercible_numeric:<column>` and `ignored_unknown_fields:<count>`.

**Output.** Two Kafka sinks write to `transactions.scored` (`sink-scored`) and `transactions.dlq` (`sink-dlq`) with an at-least-once delivery guarantee.

**Checkpointing and parallelism.** Checkpoints run every `CHECKPOINT_MS` (30,000 ms) in exactly-once mode, and the job runs with parallelism `PARALLELISM` (1) on a TaskManager with 2 slots. Exactly-once applies to Flink's internal state. The Kafka outputs are at-least-once, so after a recovery some results can be published again. Because the source starts from the earliest offset, a newly submitted job reads `transactions.raw` from the beginning and scores the retained events again.

Fields of a scored event:

| Field | Description |
|---|---|
| `transaction_id` | `TransactionID` of the input event |
| `fraud_probability` | meta-learner output; null for `REVIEW` |
| `is_fraud_pred` | 1 when the probability is at or above the threshold, 0 otherwise; null for `REVIEW` |
| `decision` | `ALERT`, `PASS` or `REVIEW` |
| `model_version` | `model_version` from `manifest.json` |
| `processing_ts` | Unix time when the event was scored |
| `latency_ms` | time spent in the scoring function, in milliseconds; null for `REVIEW` |
| `error` | error message for `REVIEW` events |
| `warnings` | non-blocking input issues, or null |

Job configuration (environment variables with their defaults in `flink_job.py`):

| Variable | Default | Meaning |
|---|---|---|
| `KAFKA_BOOTSTRAP` | `kafka:9092` | broker address |
| `TOPIC_RAW`, `TOPIC_SCORED`, `TOPIC_DLQ` | `transactions.raw`, `transactions.scored`, `transactions.dlq` | topic names |
| `CONSUMER_GROUP` | `fraud-scoring` | consumer group of the Kafka source |
| `ARTIFACTS_DIR` | `/opt/artifacts` | artifact bundle inside the image |
| `CHECKPOINT_MS` | 30000 | checkpoint interval |
| `PARALLELISM` | 1 | job parallelism |

**Runtime image.** `serving/Dockerfile` starts from `flink:2.1.0-java17` and adds Python 3.12 in a virtual environment at `/opt/pyenv`, PyFlink (`apache-flink==2.1.0`), the pinned libraries from `requirements-serve.txt` (checked with `pip check` at build time), and the Kafka connector `flink-sql-connector-kafka` 5.0.0-2.1. PyFlink runs the Python code in thread mode. The job code is copied to `/opt/serving` and the artifacts to `/opt/artifacts`.

## 6. Persistence

`sink/consumer_sink.py` consumes `transactions.scored` and `transactions.dlq` with consumer group `scored-sink`, starting from the earliest offset on first start, with automatic offset commits disabled. Rows are collected in batches that are written when they reach `BATCH_ROWS` (500) rows or `BATCH_WAIT_S` (2.0) seconds after their first row. Each write inserts both batches in one database transaction and commits it. Only then are the Kafka offsets committed, synchronously.

Scored rows are inserted with `ON CONFLICT (transaction_id, processing_ts) DO NOTHING`. A message redelivered after a crash between the database commit and the offset commit is therefore not stored twice. An event scored again by Flink gets a new `processing_ts` and is stored as a separate row; this happens, for example, after a recovery, after a job resubmission or when the producer loops over the file. Scored messages that are not JSON objects are skipped. Dead letters that cannot be parsed are stored with the error `unparseable_dlq_record`.

The tables are created from `sink/schema.sql` when the PostgreSQL container first starts, and data persists in the `pgdata` volume.

`scored_transactions`, primary key `(transaction_id, processing_ts)`:

| Column | Type | Content |
|---|---|---|
| `transaction_id` | text | `TransactionID` |
| `processing_ts` | double precision | Unix time of scoring |
| `event_time` | timestamptz | `processing_ts` as a UTC timestamp (time of scoring, not of the transaction) |
| `fraud_probability` | double precision | meta-learner output |
| `is_fraud_pred` | smallint | 1 or 0; null for `REVIEW` |
| `decision` | text | `ALERT`, `PASS` or `REVIEW` |
| `model_version` | text | model version |
| `latency_ms` | double precision | scoring time |
| `error` | text | error message for `REVIEW` |
| `warnings` | jsonb | input warnings |
| `inserted_at` | timestamptz | insert time |

Indexes: `(decision, event_time DESC)` and `(event_time DESC)`.

`dlq_events`: `id` (bigserial primary key), `error`, `raw`, `inserted_at`.

Example queries (`psql postgresql://fraud:fraud@localhost:5432/fraud`):

```sql
-- review queue, newest first
SELECT transaction_id, event_time, error
FROM scored_transactions WHERE decision = 'REVIEW'
ORDER BY event_time DESC LIMIT 50;

-- alerts in the last hour, highest score first
SELECT transaction_id, fraud_probability, event_time
FROM scored_transactions
WHERE decision = 'ALERT' AND event_time > now() - interval '1 hour'
ORDER BY fraud_probability DESC;

-- hourly flag rate
SELECT date_trunc('hour', event_time) AS h,
       avg(is_fraud_pred::float) AS flag_rate, count(*) AS n
FROM scored_transactions GROUP BY 1 ORDER BY 1 DESC LIMIT 24;
```

## 7. Monitoring

### 7.1 Metric sources

Prometheus (`monitoring/prometheus.yml`) scrapes every 15 seconds and evaluates the alert rules every 15 seconds:

| Scrape job | Target | Metrics |
|---|---|---|
| `flink-jobmanager` | `jobmanager:9249` | Flink job metrics (checkpoints, restarts) |
| `flink-taskmanager` | `taskmanager:9249` | Flink task metrics (backpressure) |
| `model-exporter` | `metrics-exporter:8000` | model-behavior and throughput metrics |
| `prometheus` | `localhost:9090` | Prometheus itself |

The metrics exporter (`monitoring/exporter/metrics_exporter.py`) consumes all three topics with consumer group `metrics-exporter` (starting from the latest offset on first start). It recomputes its gauges every 15 seconds and polls the consumer lag of the scoring job every 15 seconds. Its counters, rolling window and drift reference are held in memory and start again when the exporter restarts.

| Metric | Type | Meaning |
|---|---|---|
| `raw_events_total` | counter | events seen on `transactions.raw` |
| `scored_events_total{decision}` | counter | scored events by decision |
| `dlq_events_total` | counter | dead letters |
| `fraud_score` | histogram | fraud probability, 20 buckets of width 0.05 |
| `scoring_latency_ms` | histogram | `latency_ms` of scored events (buckets from 1 to 1,000 ms) |
| `fraud_flag_rate` | gauge | share of `is_fraud_pred` = 1 in the rolling window |
| `fallback_rate` | gauge | share of `REVIEW` decisions in the rolling window |
| `score_psi` | gauge | PSI of the scores in the window against the reference |
| `feature_psi{feature}` | gauge | PSI of each drift feature against the reference |
| `scoring_consumer_lag{partition}` | gauge | end offset minus the committed offset of group `fraud-scoring` on `transactions.raw` |
| `reference_bootstrap_used` | gauge | 1 when the drift reference was built from live traffic |
| `window_events` | gauge | events in the rolling window |

`scoring_latency_ms` measures only the time spent in the scoring function; time spent waiting in Kafka appears as consumer lag instead. The flag rate is computed over `ALERT` and `PASS` events, since `REVIEW` events have no prediction; they are counted in the fallback rate.

### 7.2 Drift detection

Drift is measured with the Population Stability Index, PSI = Σ (a − r) · ln(a / r), where r and a are the reference and live proportions per bin, floored at 10⁻⁶ (`monitoring/exporter/drift.py`). The code reads values below 0.1 as stable, 0.1 to 0.2 as a moderate shift and above 0.2 as a significant shift, which is the alert threshold.

The live side is a rolling window of `WINDOW_S` seconds (300 by default, at most 100,000 items). A PSI value is published only when the window holds at least 100 values. Scores use 20 equal-width bins on [0, 1]. Numeric features use up to 10 quantile bins taken from the reference (duplicate edges are merged), with values outside the range counted in the end bins. The categorical feature `card6` uses up to 20 of the most frequent reference categories plus an `OTHER` bucket.

The drift features are the ten highest-ranked model features without `TransactionDT`, a time offset whose PSI would always signal drift; `addr1` (rank 11) takes its place. The features are `C13`, `TransactionAmt`, `card1`, `C1`, `V258`, `card6`, `card2`, `V70`, `C14` and `addr1`. The score PSI is shown on the dashboard and covered by an alert rule; the per-feature values are exported as `feature_psi` and can be inspected in Prometheus or at the exporter's `/metrics` endpoint.

The reference comes from one of two sources:

| Source | When | Content |
|---|---|---|
| `artifacts/reference_distributions.json` | the file exists (not included in the repository) | built by `tools/make_reference.py` from `train_transaction.csv` and `train_identity.csv`: feature distributions over the training window (first 80% by `TransactionDT`) and the distribution of ensemble scores on a 20,000-row sample of the held-out period (seed 42) |
| live traffic | no reference file | the first `BOOTSTRAP_N` scores (5,000 by default) and up to `BOOTSTRAP_N` raw values per drift feature collected until then; `reference_bootstrap_used` is 1, and the PSI panels fill in only after that point |

Building the reference file:

```bash
pip install -r requirements-serve.txt
python tools/make_reference.py --data-dir <folder with train_transaction.csv and train_identity.csv>
docker compose restart metrics-exporter
```

Exporter configuration:

| Variable | Default | Meaning |
|---|---|---|
| `WINDOW_S` | 300 | rolling window in seconds |
| `BOOTSTRAP_N` | 5000 | events used for the live reference when no reference file exists |
| `EXPORTER_PORT` | 8000 | metrics port |
| `SCORING_GROUP` | `fraud-scoring` | consumer group whose lag is reported |
| `ARTIFACTS_DIR` | `/opt/artifacts` | location of the optional reference file |

### 7.3 Dashboards

Grafana provisions two data sources, Prometheus (default) and PostgreSQL, and one dashboard, **Fraud Detection** (UID `fraud-detection`), in the folder "Fraud Detection". It refreshes every 5 seconds over the last 15 minutes and is organized in two rows:

| Row | Data source | Panels |
|---|---|---|
| Transactions | PostgreSQL | counters for scored events (total and last 5 minutes), `ALERT` and `REVIEW` in the last 5 minutes, flag rate in the last 5 minutes and dead letters (total); tables of the latest scored transactions and the latest fraud alerts |
| Model behavior | Prometheus | decisions per second, fraud flag rate, score PSI with the 0.2 line, fallback rate with the bootstrap flag |

Throughput, scoring latency, consumer lag and the Flink job metrics (checkpoints, restarts, backpressure) are not on the dashboard; they remain available in Prometheus and the Flink web UI.

### 7.4 Alert rules

`monitoring/alert_rules.yml` defines six rules in two groups, `system-health` (the first three) and `model-behavior` (the last three):

| Alert | Condition | Duration | Severity |
|---|---|---|---|
| `ScoringLatencyP99Breach` | p99 of `scoring_latency_ms` over 5 minutes above 500 ms | 5 min | page |
| `DlqSpike` | dead-letter rate over 5 minutes above 1 event/s | 2 min | page |
| `ExporterDown` | the exporter scrape target is down | 2 min | warn |
| `FraudFlagRateOutOfBounds` | flag rate below 0.5% or above 15% | 10 min | page |
| `ScorePsiBreach` | score PSI above 0.2 | 10 min | page |
| `FallbackRateSpike` | more than 1% of events in `REVIEW` | 5 min | page |

Active alerts are shown at http://localhost:9090/alerts. No Alertmanager is configured, so no notifications are sent.

## 8. Running the project

### 8.1 Prerequisites

- Docker with Compose v2.
- `test_transaction.csv` and `test_identity.csv` from the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection/data), placed in `data/`.
- Free host ports 3000, 8000, 5432, 8081, 9090 and 29092.
- No existing containers named `kafka`, `jobmanager`, `producer`, `metrics-exporter`, `prometheus`, `grafana`, `postgres` or `sink`, because the compose file uses fixed container names.

### 8.2 Start

```bash
docker compose up --build -d
```

On a host with 8 GB of RAM, add the low-memory profile:

```bash
docker compose -f docker-compose.yml -f docker-compose.lowmem.yml up --build -d
```

| Service | Low-memory setting |
|---|---|
| Kafka | heap `-Xmx384m -Xms256m`, container limit 768 MB |
| Flink JobManager | process size 768 MB, container limit 1 GB |
| Flink TaskManager | process size 1,280 MB (managed memory 64 MB, network memory at most 64 MB), container limit 1,800 MB |
| PostgreSQL, Prometheus, Grafana | container limit 256 MB each |
| Producer | 4 events/s |
| Metrics exporter | live drift reference from the first 1,000 events |

### 8.3 Start order

Compose starts `kafka` and `postgres` first; both have health checks. Once Kafka is healthy, `topics-init` creates the topics and `jobmanager` starts. `taskmanager` and `job-submit` wait for a healthy JobManager, and `job-submit` also waits for `topics-init` to finish. `producer` and `metrics-exporter` wait for `topics-init`, and `sink` waits for `topics-init` and a healthy PostgreSQL. `prometheus` starts after the metrics exporter, and `grafana` after Prometheus. A staged start for hosts with little free memory is described in [operations.md](operations.md).

### 8.4 Service addresses

| Service | Address |
|---|---|
| Grafana | http://localhost:3000 (user `admin`, password `admin`) |
| Flink UI | http://localhost:8081 |
| Prometheus | http://localhost:9090 |
| Metrics exporter | http://localhost:8000/metrics |
| PostgreSQL | `localhost:5432`, database `fraud`, user `fraud`, password `fraud` |
| Kafka | `localhost:29092` |

### 8.5 Checking the stack

```bash
docker compose ps
curl http://localhost:8081/jobs/overview
docker compose logs --tail 3 sink
```

Expected: `kafka` and `postgres` healthy, `topics-init` and `job-submit` exited with code 0, and all other services running. The job `fraud-detection-scoring` should be in state `RUNNING`, and the sink log should show lines such as `[sink] flushed scored=12 dlq=0 (totals 477/0)`.

### 8.6 Updating code or the model

The Flink job code and the artifacts are copied into the serving image. After changing either, rebuild the image and resubmit the job:

```bash
docker compose build jobmanager
docker compose up -d --no-deps --force-recreate jobmanager taskmanager
docker compose up -d --no-deps --force-recreate job-submit
```

The resubmitted job reads `transactions.raw` from the earliest offset (section 5.2). After replacing the artifacts, also run `docker compose restart metrics-exporter` so that the drift reference is rebuilt or reloaded. Other services are rebuilt with `docker compose up -d --build <service>`.

### 8.7 Alert demonstration

Set `MALFORMED_PPM: "500000"` under `producer` in `docker-compose.yml` and run `docker compose up -d producer`. About half of the events then go to `transactions.dlq`, and `DlqSpike` fires once the 5-minute dead-letter rate stays above 1 event/s for 2 minutes. Set the value back to `"200"` and run the same command to return to normal traffic.

### 8.8 Stop

| Task | Command |
|---|---|
| Stop and keep the database | `docker compose down` |
| Stop and delete the database volume | `docker compose down -v` |
| Also remove the locally built images | `docker compose down -v --rmi local` |

## 9. Testing and CI

The suite has 30 `pytest` tests in five modules:

| Module | Tests | Covers |
|---|---|---|
| `test_schema_validation.py` | 7 | JSON parsing, `id-XX` key normalization, type coercion, null handling, dead-letter records, `TransactionID` extraction |
| `test_scoring_udf.py` | 6 | decisions above, below and at the threshold, output fields, base-model order, `REVIEW` on inference errors and on unparseable input |
| `test_preprocessing_parity.py` | 3 | serving path against direct pipeline and model calls to 1e-6 on 50 synthetic events, threshold matches `threshold.json`, unseen categories map to -1 |
| `test_metrics_exporter.py` | 8 | PSI calculation, histogram and category proportions, quantile edges, rolling-window eviction, live reference bootstrap, minimum sample size |
| `test_sink.py` | 6 | mapping of scored and dead-letter messages to database rows, batching by size and by age |

The schema, scoring, exporter and sink tests need no Flink, Kafka or PostgreSQL; the scoring tests use substitute model components. The parity tests load the committed artifacts and are skipped only when `artifacts/manifest.json` is missing. The Flink job wiring (`flink_job.py`), the producer, and the Kafka and PostgreSQL connection code of the sink and exporter are not covered by automated tests.

Run locally:

```bash
pip install -r requirements-serve.txt pytest
python -m pytest tests/
```

The GitHub Actions workflow `.github/workflows/tests.yml` runs on every push and pull request. It uses `ubuntu-latest` with Python 3.12, installs `requirements-serve.txt` and `pytest`, and runs `python -m pytest tests/ -v`.

## 10. Reproducibility

The serving libraries are pinned in `requirements-serve.txt` and match the training run recorded in `manifest.json`, except for pandas:

| Library | Training run (`manifest.json`) | Serving (`requirements-serve.txt`) |
|---|---|---|
| Python | 3.12.13 | 3.12 |
| NumPy | 2.0.2 | 2.0.2 |
| pandas | 2.3.3 | 2.2.3 |
| scikit-learn | 1.6.1 | 1.6.1 |
| SciPy | 1.16.3 | 1.16.3 |
| joblib | 1.5.3 | 1.5.3 |
| XGBoost | 3.2.0 | 3.2.0 |
| LightGBM | 4.6.0 | 4.6.0 |
| CatBoost | 1.2.10 | 1.2.10 |

Other pinned components are the images `flink:2.1.0-java17`, `apache/kafka:3.9.0`, `postgres:17.4`, `prom/prometheus:v3.2.1` and `grafana/grafana:12.2.1`, PyFlink 2.1.0 and the Kafka connector 5.0.0-2.1. The producer, sink and exporter images are based on `python:3.12-slim` and pin their packages:

| Image | Packages |
|---|---|
| producer | `pandas==2.3.3`, `numpy==2.0.2`, `confluent-kafka==2.15.1` |
| sink | `psycopg[binary]==3.2.13`, `confluent-kafka==2.15.1` |
| metrics exporter | `confluent-kafka==2.15.1`, `prometheus-client==0.26.0` |

Not pinned to exact versions: the `python:3.12-slim` base tag, the Python 3.12 patch release in the serving image, `pip` (upgraded at build time and in CI) and `pytest` in CI.

The manifest records the seed (42), the SHA-256 hashes of the training files and the training hardware (Linux x86_64, GPU detected). The producer's malformed-event injection and the reference sample in `tools/make_reference.py` also use seed 42. Every scored event and stored row carries the model version, so each decision can be traced to one artifact bundle. The parity test (section 9) checks that the serving path reproduces direct calls to the pipeline and models.

## 11. Design decisions

| Decision | Rationale | In the implementation |
|---|---|---|
| Apache Flink for serving, rather than Spark Structured Streaming | Fraud scoring is per event and latency-sensitive; Flink processes events one at a time, while Spark Structured Streaming processes micro-batches. | one Python operator scores each event; checkpoints every 30 s |
| Time-ordered evaluation | Fraud patterns change over time, and a deployed model always scores later transactions; a random split would let the model learn from the future. | split on `TransactionDT`; threshold chosen on the last 10% of the training period; ROC-AUC on the most recent 20% |
| Class weights, no resampling of evaluation data | Oversampling before the split can leak synthetic copies of training fraud into the test data and inflate the metrics. | positive-class weights of 15.20 to 17.04; recorded fraud rates of 3.50% (validation) and 3.44% (held-out) |
| Accuracy never reported alone | At about 3.5% fraud, a model that never flags fraud would already be about 96.5% accurate, so accuracy by itself is misleading. | the notebook's holdout evaluation reports accuracy together with ROC-AUC and the confusion matrix; the threshold comes from the precision-recall curve instead of a fixed 0.5 |
| Native model formats and one pickled preprocessing object | Native formats reduce coupling to library versions; the pickle is loaded with the scikit-learn version it was saved with. | XGBoost JSON, LightGBM text, CatBoost binary; scikit-learn 1.6.1 in both the manifest and `requirements-serve.txt` |
| No reimplementation of preprocessing in serving | Reusing the fitted object prevents differences between training-time and serving-time features. | `ScoringService` applies `preprocessing.pkl`; the parity test checks the serving path to 1e-6 |
| Reserved code for unseen categories | New category values at serving time must not cause errors. | ordinal encoding with unknown value -1, covered by a parity test |
| Fail open to human review | An inference error must not drop a transaction or stop the job. | `REVIEW` decision with the error message; malformed input goes to `transactions.dlq`; `FallbackRateSpike` and `DlqSpike` alerts |

## 12. Limitations

| Area | Limitation |
|---|---|
| Labels | The replayed test files are unlabeled, so precision and recall cannot be measured on live traffic; monitoring relies on the flag rate, fallback rate and PSI. |
| Recorded metrics | The manifest records ROC-AUC as its only held-out metric; accuracy comes from the holdout evaluation in the training notebook. Precision, recall and F1 at the deployed threshold are not recorded in the repository. |
| Time feature | `TransactionDT`, a model input, is a time offset that grows over time, so newer transactions carry values outside the range seen in training. For the same reason it is excluded from drift monitoring. |
| Drift reference | Without `reference_distributions.json` (not included), PSI compares live traffic with the first live events of the current exporter run, not with the training data. Building the file requires the Kaggle training files. |
| Repeated scoring | A Flink recovery can publish results again (at-least-once outputs), a resubmitted job re-scores the events retained in `transactions.raw`, and the producer loop re-sends the same transactions. Each re-scoring is stored as a new row, because its `processing_ts` differs. |
| Failed database writes | If a batch insert fails while the database connection stays usable, the sink rolls back and discards that batch. The next successful batch commits the Kafka offsets past it, so those rows are not stored. |
| Latency | Only the time spent in the scoring function is measured. End-to-end latency is not measured; queueing shows up as consumer lag. |
| Scale and durability | One Kafka broker, one partition per topic, replication factor 1 and job parallelism 1. The Flink checkpoint directory (`file:///tmp/flink-checkpoints`) is not on a Docker volume, so checkpoints do not survive recreating the Flink containers. |
| Alerting | No Alertmanager is configured; alerts are shown in Prometheus, but no notifications are sent. |
| Security | Plaintext Kafka listeners and default credentials in `docker-compose.yml` (Grafana `admin`/`admin`, PostgreSQL `fraud`/`fraud`); the setup is intended for local use only. |

## 13. Repository layout

```
.
├── .github/workflows/tests.yml     CI: pytest on every push and pull request
├── artifacts/                      model bundle (section 4)
├── assets/                         dashboard screenshots used by the README
├── data/                           replay CSVs from Kaggle (not tracked; only .gitkeep)
├── docs/
│   ├── operations.md               operating guide
│   └── PROJECT.md                  this document
├── monitoring/
│   ├── exporter/                   metrics exporter: Dockerfile, drift.py, metrics_exporter.py
│   ├── grafana/
│   │   ├── dashboards/             fraud_detection.json (single provisioned dashboard)
│   │   └── provisioning/           data sources and dashboard provider
│   ├── alert_rules.yml             alert rules
│   └── prometheus.yml              scrape and rule configuration
├── producer/                       replay producer: Dockerfile, replay_producer.py
├── serving/                        Flink job: Dockerfile, flink_job.py, scoring_udf.py, schemas.py
├── sink/                           PostgreSQL sink: Dockerfile, consumer_sink.py, schema.sql
├── tests/                          pytest suite (5 modules, 30 tests)
├── tools/make_reference.py         builds the drift reference from the training files
├── .dockerignore
├── .gitignore
├── docker-compose.yml              full stack
├── docker-compose.lowmem.yml       low-memory profile
├── requirements-serve.txt          pinned serving and test dependencies
├── LICENSE                         MIT
└── README.md
```

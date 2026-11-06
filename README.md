# Real-Time Financial Fraud Detection

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![Apache Flink 2.1](https://img.shields.io/badge/Apache%20Flink-2.1-E6526F.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A streaming fraud detection system that scores card transactions one at a time as they arrive. A stacking ensemble of XGBoost, LightGBM and CatBoost, combined by an XGBoost meta-learner, runs inside an Apache Flink job that reads transactions from Kafka and writes decisions back to Kafka. Every decision is stored in PostgreSQL, and system and model metrics are collected by Prometheus and shown in Grafana. The whole stack runs locally with a single Docker Compose command.

## Highlights

- **Per-event scoring** in an Apache Flink 2.1 (PyFlink) job, with Kafka as input and output.
- **Stacking ensemble**: XGBoost, LightGBM and CatBoost base models feeding an XGBoost meta-learner, using 30 features selected by feature importance.
- **ROC-AUC 0.918 and accuracy 0.972** on 118,108 held-out transactions, the most recent 20% of the timeline.
- **Robust input handling**: malformed events are routed to a dead-letter topic, and inference errors produce a `REVIEW` decision instead of a dropped event.
- **Monitoring**: one Grafana dashboard covering live transactions and model behavior (decisions per second, flag rate, score drift, fail-open rate), backed by 6 Prometheus alert rules.
- **Persistence**: every scored transaction and dead letter is stored in PostgreSQL for review queries.
- **Reproducible**: pinned dependency versions, versioned model artifacts with a manifest, and 30 automated tests run by GitHub Actions.

## Architecture

The replay producer streams the IEEE-CIS test transactions to Kafka as JSON events. Inside the Flink job, each event goes through these steps:

1. **Validate**: parse the JSON, normalize identity column names (`id-01` to `id_01`) and coerce types. Malformed events go to `transactions.dlq` with the error reason and the original payload.
2. **Preprocess**: apply the fitted scikit-learn pipeline (median imputation for 401 numeric columns; most-frequent imputation and ordinal encoding for 31 categorical columns, with unseen categories mapped to -1), then select the 30 model features.
3. **Score**: the three base models each produce a fraud probability; the meta-learner combines them (in the order XGBoost, LightGBM, CatBoost) into the final probability.
4. **Decide**: a probability at or above the threshold (0.2883) gives `ALERT`, otherwise `PASS`; any inference error gives `REVIEW`.
5. **Publish**: the result is written to `transactions.scored`.

The job checkpoints every 30 seconds in exactly-once mode. Downstream, a sink service writes every result and dead letter to PostgreSQL, and a metrics exporter reads the topics and feeds Prometheus, which Grafana displays.

## Model

| | |
|---|---|
| Base models | XGBoost (400 trees, depth 10), LightGBM (700 trees, depth 10), CatBoost (600 trees, depth 9) |
| Meta-learner | XGBoost (200 trees, depth 2, learning rate 0.1) |
| Features | top 30 by importance of an XGBoost model trained on all features |
| Tuning | Optuna (TPE sampler), objective averaged over 5 time-series cross-validation folds |
| Class imbalance | positive-class weights of 15.2 to 17.0 in the base models |
| Training data | IEEE-CIS training set, 590,540 transactions |
| Split | time-ordered on `TransactionDT`: first 80% (472,432 rows) for training, last 20% (118,108 rows) held out |
| Decision threshold | 0.2883, the F1-maximizing point on a validation slice (the last 10% of the training period) |
| **Held-out ROC-AUC** | **0.918** |
| **Held-out accuracy** | **0.972** at the decision threshold |
| Model version | `20260921T144658Z` |

Accuracy comes from the holdout evaluation in `notebook/fraud-detection-training.ipynb` (the confusion matrix and ROC curve are plotted there). Because only about 3.5% of transactions are fraud, accuracy is reported together with ROC-AUC, never alone: a model that never flags fraud would already be about 96.5% accurate. Training runs vary slightly in the third decimal: `artifacts/manifest.json` records the ROC-AUC of the committed model, and the notebook's holdout cell reports the metrics of its own most recent run.

The trained model is included in `artifacts/`. Models are stored in their native formats (XGBoost JSON, LightGBM text, CatBoost binary); the preprocessing pipeline is a pickled scikit-learn object, loaded with the same scikit-learn version it was saved with. `artifacts/manifest.json` records the library versions, random seed, SHA-256 hashes of the training files, selected features, hyperparameters and threshold.

## Technology stack

| Component | Version |
|---|---|
| Apache Flink / PyFlink | 2.1.0 (Java 17 image) |
| Flink Kafka connector | 5.0.0-2.1 |
| Apache Kafka (KRaft mode, single broker) | 3.9.0 |
| Python | 3.12 |
| XGBoost / LightGBM / CatBoost | 3.2.0 / 4.6.0 / 1.2.10 |
| scikit-learn / NumPy / pandas | 1.6.1 / 2.0.2 / 2.2.3 |
| PostgreSQL | 17.4 |
| Prometheus | 3.2.1 |
| Grafana | 12.2.1 |
| Orchestration | Docker Compose v2 |

## Quick start

### Prerequisites

- Docker with Compose v2.
- The replay files `test_transaction.csv` and `test_identity.csv` from the [IEEE-CIS Fraud Detection competition on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) (a Kaggle account is needed to download them). Place both files in `data/`.

### Run

```bash
docker compose up --build -d
```

On a machine with 8 GB RAM, use the low-memory profile:

```bash
docker compose -f docker-compose.yml -f docker-compose.lowmem.yml up --build -d
```

Then open Grafana at http://localhost:3000 (user `admin`, password `admin`) and open the **Fraud Detection** dashboard in the "Fraud Detection" folder.

The full operating guide (start order, health checks, configuration, alert demo, troubleshooting) is in [docs/operations.md](docs/operations.md).

### Run the tests

```bash
pip install -r requirements-serve.txt pytest
python -m pytest tests/
```

## Demo

A two-minute walkthrough once the stack is up:

1. **Open the dashboard**: http://localhost:3000/d/fraud-detection/fraud-detection (user `admin`, password `admin`). The **Transactions** row fills with scored transactions within a minute; `ALERT` rows are highlighted in red with their fraud probability.
2. **Model behavior row**: decisions per second, the fraud flag rate settling near the dataset's natural rate, and the score PSI drift panel with its 0.2 alert line.
3. **Trigger an alert**: set `MALFORMED_PPM: "500000"` under `producer` in `docker-compose.yml` and run `docker compose up -d producer`. The **Dead letters** counter in the Transactions row climbs, and the `DlqSpike` alert fires at http://localhost:9090/alerts after a few minutes. Set the value back to `"200"` and rerun the command to recover.
4. **Fail-open path**: inference stays healthy through the bad input — the `REVIEW` counter and the fallback-rate panel stay near zero.

![Transactions — live feed and alerts](assets/dashboard-transactions.png)

![Model behavior — flag rate and score drift](assets/dashboard-model-behavior.png)

## Services and ports

| Service | Address | Purpose |
|---|---|---|
| Grafana | http://localhost:3000 | dashboard |
| Flink UI | http://localhost:8081 | job status, checkpoints, backpressure |
| Prometheus | http://localhost:9090 | metrics and alerts |
| Metrics exporter | http://localhost:8000/metrics | model-behavior metrics |
| PostgreSQL | `localhost:5432` | database `fraud` (user and password `fraud`) |
| Kafka | `localhost:29092` | external listener for local tools |

## Event flow

| Kafka topic | Content |
|---|---|
| `transactions.raw` | one JSON transaction per message, with the raw IEEE-CIS columns |
| `transactions.scored` | one scoring result per transaction |
| `transactions.dlq` | malformed input, with the error reason and the original payload |

Fields of a scored event:

| Field | Description |
|---|---|
| `transaction_id` | `TransactionID` of the input event |
| `fraud_probability` | meta-learner output (null for `REVIEW`) |
| `is_fraud_pred` | 1 when the probability is at or above the threshold |
| `decision` | `ALERT`, `PASS` or `REVIEW` |
| `model_version` | version from `artifacts/manifest.json` |
| `processing_ts` | Unix time when the event was scored |
| `latency_ms` | scoring time inside the Flink operator |
| `error` | error message for `REVIEW` events |
| `warnings` | non-blocking input issues, such as a value that could not be converted to a number |

To read the topics from the host, for example with [kcat](https://github.com/edenhill/kcat):

```bash
kcat -b localhost:29092 -t transactions.scored -C -o -5 -e
kcat -b localhost:29092 -t transactions.dlq -C -o -5 -e
```

## Monitoring

One Grafana dashboard, **Fraud Detection**, is provisioned automatically, with two rows:

- **Transactions** (PostgreSQL): counters for the last 5 minutes (scored, `ALERT`, `REVIEW`, flag rate, dead letters), a live feed of the latest scored transactions and a table of the latest fraud alerts.
- **Model behavior** (Prometheus): decisions per second, fraud flag rate, score PSI against the training reference with the 0.2 alert line, and the fallback (`REVIEW`) rate.

Throughput, scoring latency, consumer lag and Flink checkpoint metrics remain available in Prometheus (http://localhost:9090) and the Flink UI (http://localhost:8081).

Drift is measured with the Population Stability Index (PSI) over a 5-minute rolling window. The score PSI is shown on the dashboard and alerted on; the exporter also computes PSI for ten input features (`C13`, `TransactionAmt`, `card1`, `C1`, `V258`, `card6`, `card2`, `V70`, `C14` and `addr1`), available at `http://localhost:8000/metrics` and in Prometheus as `feature_psi`. The reference distribution comes from `artifacts/reference_distributions.json` when present (built from the training data with `tools/make_reference.py`), or otherwise from the first live events.

Alert rules (`monitoring/alert_rules.yml`):

| Alert | Condition |
|---|---|
| `ScoringLatencyP99Breach` | p99 scoring latency above 500 ms for 5 minutes |
| `DlqSpike` | dead-letter rate above 1 event/s for 2 minutes |
| `ExporterDown` | metrics exporter unreachable for 2 minutes |
| `FraudFlagRateOutOfBounds` | flag rate outside 0.5% to 15% for 10 minutes |
| `ScorePsiBreach` | score PSI above 0.2 for 10 minutes |
| `FallbackRateSpike` | more than 1% of events in `REVIEW` for 5 minutes |

## Persistence

The `sink` service consumes `transactions.scored` and `transactions.dlq` and writes them to the `scored_transactions` and `dlq_events` tables. Kafka offsets are committed only after the database commit, and scored rows use the primary key `(transaction_id, processing_ts)` with `ON CONFLICT DO NOTHING`, so a redelivered message is ignored. Data is kept in the `pgdata` Docker volume across restarts.

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

## Tests

30 tests run with `pytest`, locally and on every push through GitHub Actions (`.github/workflows/tests.yml`).

| Test file | Tests | Covers |
|---|---|---|
| `test_schema_validation.py` | 7 | JSON parsing, `id-XX` column normalization, type coercion, dead-letter records |
| `test_scoring_udf.py` | 6 | threshold application, output fields, base-model order, `REVIEW` on inference errors |
| `test_preprocessing_parity.py` | 3 | the serving path matches direct pipeline and model calls to 1e-6 on 50 synthetic events; unseen categories map to -1 |
| `test_metrics_exporter.py` | 8 | PSI calculation, rolling window, drift reference built from live events |
| `test_sink.py` | 6 | database record mapping and batching |

## Project structure

```
.
├── .github/workflows/tests.yml   CI workflow (pytest)
├── artifacts/                    trained models, preprocessing pipeline, features, threshold, manifest
├── assets/                       dashboard screenshots for this README
├── data/                         replay CSVs from Kaggle (not tracked)
├── docs/                         PROJECT.md (full documentation), operations.md (operating guide)
├── monitoring/
│   ├── exporter/                 model-behavior metrics exporter (PSI, flag rate, lag)
│   ├── grafana/                  dashboard and provisioning
│   ├── alert_rules.yml
│   └── prometheus.yml
├── notebook/                     EDA and model-training notebooks
├── producer/                     replay producer
├── serving/                      PyFlink job, scoring operator, event schema, Dockerfile
├── sink/                         Kafka-to-PostgreSQL consumer and table schema
├── tests/                        pytest suite
├── tools/make_reference.py       builds the drift reference from the training data
├── docker-compose.yml
├── docker-compose.lowmem.yml     low-memory profile
└── requirements-serve.txt        serving and test dependencies
```

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection), Kaggle. The data files are not included in this repository; download them from Kaggle.

## License

MIT. See [LICENSE](LICENSE).

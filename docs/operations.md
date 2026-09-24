# Operations Guide

How to run, inspect, configure and stop the fraud detection stack with Docker Compose.

## 1. Prerequisites

- Docker Desktop or Docker Engine with Compose v2 (`docker compose version` works).
- Replay data in `data/`: `test_transaction.csv` and `test_identity.csv` from the [IEEE-CIS Fraud Detection competition on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data).
- Free host ports: 3000 (Grafana), 8081 (Flink UI), 9090 (Prometheus), 8000 (metrics exporter), 5432 (PostgreSQL), 29092 (Kafka).
- No other containers named `kafka`, `jobmanager`, `producer`, `metrics-exporter`, `prometheus`, `grafana`, `postgres` or `sink` (the compose file uses fixed names). Check with `docker ps -a`, and rename a clash with `docker rename kafka old_kafka`.

## 2. Start

Standard start:

```bash
docker compose up --build -d
```

Low-memory profile for hosts with 8 GB RAM:

```bash
docker compose -f docker-compose.yml -f docker-compose.lowmem.yml up --build -d
```

The low-memory profile sets:

| Service | Setting |
|---|---|
| Kafka | heap 384 MB, container limit 768 MB |
| Flink JobManager | process size 768 MB, container limit 1 GB |
| Flink TaskManager | process size 1280 MB, container limit 1800 MB |
| PostgreSQL, Prometheus, Grafana | container limit 256 MB each |
| Producer | 4 events/s |
| Metrics exporter | drift reference from the first 1,000 events |

When using the profile, every `docker compose up` or `build` command needs both files. To avoid repeating the `-f` options, set `COMPOSE_FILE` once per terminal session:

```powershell
# PowerShell
$env:COMPOSE_FILE = "docker-compose.yml;docker-compose.lowmem.yml"
```

```bash
# bash
export COMPOSE_FILE=docker-compose.yml:docker-compose.lowmem.yml
```

The first build downloads Flink, PyFlink and the machine-learning libraries; later starts reuse the Docker cache.

### Start order

Compose starts the services in dependency order:

1. `kafka` and `postgres` (both health-checked).
2. `topics-init` creates `transactions.raw`, `transactions.scored` and `transactions.dlq`; `jobmanager` starts once Kafka is healthy.
3. `taskmanager` and `job-submit` start once the JobManager is healthy (`job-submit` also waits for the topics).
4. `producer`, `sink` and `metrics-exporter` start once the topics exist.
5. `prometheus`, then `grafana`.

### Staged start

On a host with little free memory, start the stack in stages:

```bash
docker compose up -d kafka postgres topics-init
docker compose up -d jobmanager taskmanager
#   wait until http://localhost:8081/overview shows "taskmanagers":1
docker compose up -d job-submit
#   wait until `docker compose logs job-submit` prints "Job has been submitted"
docker compose up -d producer sink metrics-exporter prometheus grafana
```

## 3. Check that it is running

```bash
docker compose ps
```

Expected: `kafka` and `postgres` healthy; `topics-init` and `job-submit` exited with code 0; all other services up.

```bash
curl http://localhost:8081/jobs/overview
```

Expected: job `fraud-detection-scoring` in state `RUNNING`.

```bash
docker compose logs --tail 3 sink
```

Expected lines such as `[sink] flushed scored=12 dlq=0 (totals 477/0)`.

```bash
docker exec postgres psql -U fraud -d fraud -c "select decision, count(*) from scored_transactions group by 1"
docker exec kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:9092 --topic transactions.scored --max-messages 3
```

In Git Bash, prefix `docker exec` commands with `MSYS_NO_PATHCONV=1`.

## 4. Web interfaces

| Interface | Address |
|---|---|
| Grafana | http://localhost:3000 (admin / admin) |
| Flink UI | http://localhost:8081 |
| Prometheus | http://localhost:9090 (alerts at `/alerts`, targets at `/targets`) |
| Metrics exporter | http://localhost:8000/metrics |
| PostgreSQL | `localhost:5432`, database `fraud`, user `fraud`, password `fraud` |

Grafana dashboards (folder "Fraud Detection"):

| Dashboard | Address | Panels |
|---|---|---|
| Transactions | http://localhost:3000/d/fraud-transactions/transactions?refresh=5s | scored totals, ALERT and REVIEW counts, flag rate, dead letters, live feed of scored transactions, latest alerts, latest REVIEW events, latest dead letters |
| Model Behavior | http://localhost:3000/d/fraud-model-behavior/model-behavior?refresh=5s | flag rate, decisions per second, score quantiles, score PSI, feature PSI, fallback rate |
| System Health | http://localhost:3000/d/fraud-system-health/system-health?refresh=5s | events per second (raw, scored, DLQ), scoring latency, consumer lag, Flink checkpoints, restarts, backpressure |

## 5. Configuration

All settings are environment variables in `docker-compose.yml`. After changing a value, apply it with `docker compose up -d <service>`.

Producer (`producer`):

| Variable | Default | Meaning |
|---|---|---|
| `RATE_EPS` | 50 | events per second |
| `BURST_EVERY_S` | 0 | send a burst every N seconds (0 = off) |
| `BURST_SIZE` | 500 | events per burst |
| `LOOP` | true | restart from the first row when the file ends |
| `MALFORMED_PPM` | 200 | malformed events per million (dead-letter traffic) |
| `DATA_DIR` | /data | folder with the replay CSVs |

Flink job (set on `job-submit`):

| Variable | Default | Meaning |
|---|---|---|
| `PARALLELISM` | 1 | job parallelism (the TaskManager has 2 slots) |
| `CHECKPOINT_MS` | 30000 | checkpoint interval |
| `CONSUMER_GROUP` | fraud-scoring | Kafka consumer group |
| `ARTIFACTS_DIR` | /opt/artifacts | model artifacts inside the image |
| `TOPIC_RAW`, `TOPIC_SCORED`, `TOPIC_DLQ` | transactions.raw / .scored / .dlq | topic names |

Metrics exporter (`metrics-exporter`):

| Variable | Default | Meaning |
|---|---|---|
| `WINDOW_S` | 300 | rolling window for flag rate, fallback rate and PSI |
| `BOOTSTRAP_N` | 5000 | events used to build the drift reference when no reference file exists |
| `EXPORTER_PORT` | 8000 | metrics port |
| `SCORING_GROUP` | fraud-scoring | consumer group whose lag is reported |

Sink (`sink`):

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | postgresql://fraud:fraud@postgres:5432/fraud | PostgreSQL connection |
| `BATCH_ROWS` | 500 | rows per database batch |
| `BATCH_WAIT_S` | 2.0 | maximum seconds before a batch is written |

## 6. Drift reference

The score PSI and feature PSI panels compare live traffic with a reference distribution:

- If `artifacts/reference_distributions.json` exists, the exporter uses it.
- Otherwise the exporter builds the reference from the first `BOOTSTRAP_N` events, and the metric `reference_bootstrap_used` is 1. The PSI panels fill in once that many events have been scored.

To build the reference from the IEEE-CIS training files:

```bash
pip install -r requirements-serve.txt
python tools/make_reference.py --data-dir <folder with train_transaction.csv and train_identity.csv>
docker compose restart metrics-exporter
```

## 7. Trigger an alert

To see the `DlqSpike` alert fire, set `MALFORMED_PPM: "500000"` under `producer` in `docker-compose.yml` and run:

```bash
docker compose up -d producer
```

About half of the events are then sent to `transactions.dlq`. The alert fires in Prometheus (http://localhost:9090/alerts) once the 5-minute dead-letter rate stays above 1 event/s for 2 minutes. Set the value back to `"200"` and run the same command to return to normal traffic.

## 8. Day-to-day commands

| Task | Command |
|---|---|
| Follow logs | `docker compose logs -f producer sink metrics-exporter` |
| Logs of one service | `docker compose logs -f taskmanager` |
| Restart one service | `docker compose restart grafana` |
| Rebuild after a code change | `docker compose up -d --build <service>` |
| Resource usage | `docker stats --no-stream` |

After changing `serving/*.py` (the Flink job), rebuild the image and redeploy the job:

```bash
docker compose build jobmanager
docker compose up -d --no-deps --force-recreate jobmanager taskmanager
docker compose up -d --no-deps --force-recreate job-submit
```

## 9. Stop

| Task | Command |
|---|---|
| Stop, keep the database | `docker compose down` |
| Stop and delete the database volume | `docker compose down -v` |
| Also remove the locally built images | `docker compose down -v --rmi local` |

## 10. Troubleshooting

**Docker commands hang or return "503 Service Unavailable".** The Docker VM is out of memory. Restart Docker Desktop (or run `wsl --shutdown` and start Docker Desktop again), then use the low-memory profile and the staged start.

**"container name ... is already in use".** A container from another project has the same name: `docker rename <name> old_<name>`.

**"exec: C:/Program Files/Git/...: no such file" (Git Bash).** Git Bash rewrites paths; prefix the command with `MSYS_NO_PATHCONV=1`.

**Grafana panels show "No data".**
- Score or feature PSI: the drift reference is not built yet (see section 6).
- Consumer lag: check `docker compose logs metrics-exporter` for `lag poll failed`.
- Transactions dashboard: the PostgreSQL data source did not load; run `docker compose restart grafana`.

**Flink job RESTARTING or FAILED.** Check the TaskManager log for errors:

```bash
docker compose logs taskmanager | grep -iE "error|exception"
```

In PowerShell, use `docker compose logs taskmanager | Select-String -Pattern "error|exception"`. If the TaskManager ran out of memory, raise `taskmanager.memory.process.size` and the container memory limit in `docker-compose.lowmem.yml`.

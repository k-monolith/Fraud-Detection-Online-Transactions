-- Persistent store for scored transactions and dead letters.
-- Applied automatically on first container start (docker-entrypoint-initdb.d).

CREATE TABLE IF NOT EXISTS scored_transactions (
  transaction_id     text NOT NULL,
  processing_ts      double precision NOT NULL,
  event_time         timestamptz NOT NULL,
  fraud_probability  double precision,
  is_fraud_pred      smallint,
  decision           text NOT NULL,
  model_version      text,
  latency_ms         double precision,
  error              text,
  warnings           jsonb,
  inserted_at        timestamptz NOT NULL DEFAULT now(),
  -- (transaction_id, processing_ts) identifies one scoring event; replay loops
  -- re-score the same TransactionID, so the id alone is not unique. The PK plus
  -- ON CONFLICT DO NOTHING makes a redelivered scored message a no-op.
  PRIMARY KEY (transaction_id, processing_ts)
);
CREATE INDEX IF NOT EXISTS idx_scored_decision_time
  ON scored_transactions (decision, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_scored_time
  ON scored_transactions (event_time DESC);

CREATE TABLE IF NOT EXISTS dlq_events (
  id          bigserial PRIMARY KEY,
  error       text,
  raw         text,
  inserted_at timestamptz NOT NULL DEFAULT now()
);

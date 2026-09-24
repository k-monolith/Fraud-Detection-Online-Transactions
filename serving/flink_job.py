"""PyFlink job: transactions.raw -> validate -> score -> transactions.scored
(malformed events -> transactions.dlq).

Configuration comes from environment variables — see ENV below.
Checkpointing every 30 s in exactly-once mode.
"""
from __future__ import annotations

import os

from pyflink.common import Types, WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import (
    CheckpointingMode,
    OutputTag,
    ProcessFunction,
    StreamExecutionEnvironment,
)
from pyflink.datastream.connectors.kafka import (
    DeliveryGuarantee,
    KafkaOffsetsInitializer,
    KafkaRecordSerializationSchema,
    KafkaSink,
    KafkaSource,
)

import schemas
from scoring_udf import ScoreFunction

ENV = {
    "KAFKA_BOOTSTRAP": os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092"),
    "TOPIC_RAW": os.environ.get("TOPIC_RAW", "transactions.raw"),
    "TOPIC_SCORED": os.environ.get("TOPIC_SCORED", "transactions.scored"),
    "TOPIC_DLQ": os.environ.get("TOPIC_DLQ", "transactions.dlq"),
    "CONSUMER_GROUP": os.environ.get("CONSUMER_GROUP", "fraud-scoring"),
    "ARTIFACTS_DIR": os.environ.get("ARTIFACTS_DIR", "/opt/artifacts"),
    "CHECKPOINT_MS": int(os.environ.get("CHECKPOINT_MS", "30000")),
    "PARALLELISM": int(os.environ.get("PARALLELISM", "1")),
}

DLQ_TAG = OutputTag("dlq", Types.STRING())


class ValidateFunction(ProcessFunction):
    """Schema check + normalization. Malformed events go to the DLQ side output
    instead of failing the job."""

    def process_element(self, raw: str, ctx: "ProcessFunction.Context"):
        event, err = schemas.parse_event(raw)
        if event is None:
            yield DLQ_TAG, schemas.make_dlq_record(raw, err)
            return
        if schemas.extract_transaction_id(event) is None:
            yield DLQ_TAG, schemas.make_dlq_record(raw, "missing_transaction_id")
            return
        # Re-serialize the normalized event so downstream sees canonical keys.
        import json

        yield json.dumps(event, allow_nan=False, default=str)


def build_kafka_sink(topic: str) -> KafkaSink:
    return (
        KafkaSink.builder()
        .set_bootstrap_servers(ENV["KAFKA_BOOTSTRAP"])
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(topic)
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        )
        # AT_LEAST_ONCE to the sink: downstream consumers must tolerate rare
        # duplicates on recovery; exactly-once is maintained within Flink state.
        .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
        .build()
    )


def main() -> None:
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(ENV["PARALLELISM"])
    env.enable_checkpointing(ENV["CHECKPOINT_MS"], CheckpointingMode.EXACTLY_ONCE)

    source = (
        KafkaSource.builder()
        .set_bootstrap_servers(ENV["KAFKA_BOOTSTRAP"])
        .set_topics(ENV["TOPIC_RAW"])
        .set_group_id(ENV["CONSUMER_GROUP"])
        .set_starting_offsets(KafkaOffsetsInitializer.earliest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )

    raw_stream = env.from_source(
        source, WatermarkStrategy.no_watermarks(), "transactions.raw"
    )

    validated = raw_stream.process(ValidateFunction(), output_type=Types.STRING())
    dlq_stream = validated.get_side_output(DLQ_TAG)

    scored = validated.map(
        ScoreFunction(ENV["ARTIFACTS_DIR"]), output_type=Types.STRING()
    ).name("stacking-ensemble-scoring")

    scored.sink_to(build_kafka_sink(ENV["TOPIC_SCORED"])).name("sink-scored")
    dlq_stream.sink_to(build_kafka_sink(ENV["TOPIC_DLQ"])).name("sink-dlq")

    env.execute("fraud-detection-scoring")


if __name__ == "__main__":
    main()

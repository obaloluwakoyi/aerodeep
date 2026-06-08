"""
milestone1/stream_a/ingestion.py

Kafka consumer for millisecond-level sensor telemetry.
Writes raw readings to TimescaleDB and forwards preprocessed
windows to the internal processed topic.
"""

from __future__ import annotations

import json
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
import psycopg2.pool
import yaml
from confluent_kafka import Consumer, KafkaError, Producer
from loguru import logger


@dataclass
class SensorReading:
    """Single sensor reading from the telemetry stream."""
    timestamp: datetime
    platform_id: str
    unit_id: str
    sensor_id: str
    value: float
    quality_flag: int = 0   # 0=good, 1=suspect, 2=bad


@dataclass
class TelemetryBatch:
    """A batch of readings for a single compressor unit at one timestep."""
    timestamp: datetime
    platform_id: str
    unit_id: str
    readings: Dict[str, float] = field(default_factory=dict)
    quality_flags: Dict[str, int] = field(default_factory=dict)


class TimescaleDBWriter:
    """Thread-safe writer for raw telemetry into TimescaleDB hypertable."""

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS sensor_telemetry (
        time         TIMESTAMPTZ NOT NULL,
        platform_id  TEXT        NOT NULL,
        unit_id      TEXT        NOT NULL,
        sensor_id    TEXT        NOT NULL,
        value        DOUBLE PRECISION NOT NULL,
        quality_flag SMALLINT    NOT NULL DEFAULT 0
    );

    SELECT create_hypertable(
        'sensor_telemetry', 'time',
        if_not_exists => TRUE,
        chunk_time_interval => INTERVAL '1 day'
    );

    CREATE INDEX IF NOT EXISTS idx_telemetry_unit_time
        ON sensor_telemetry (unit_id, time DESC);
    """

    INSERT_SQL = """
    INSERT INTO sensor_telemetry (time, platform_id, unit_id, sensor_id, value, quality_flag)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT DO NOTHING;
    """

    def __init__(self, dsn: str, pool_size: int = 10):
        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2, maxconn=pool_size, dsn=dsn
        )
        self._ensure_schema()
        logger.info("TimescaleDB writer ready")

    def _ensure_schema(self) -> None:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(self.CREATE_TABLE_SQL)
            conn.commit()
        finally:
            self._pool.putconn(conn)

    def write_batch(self, batch: TelemetryBatch) -> None:
        rows = [
            (
                batch.timestamp,
                batch.platform_id,
                batch.unit_id,
                sensor_id,
                value,
                batch.quality_flags.get(sensor_id, 0),
            )
            for sensor_id, value in batch.readings.items()
        ]
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.executemany(self.INSERT_SQL, rows)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error(f"TimescaleDB write failed: {exc}")
            raise
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        self._pool.closeall()


class QualityChecker:
    """Validates sensor readings against configured normal ranges."""

    def __init__(self, sensor_configs: List[Dict[str, Any]]):
        self._ranges: Dict[str, tuple] = {}
        self._critical: Dict[str, float] = {}
        for s in sensor_configs:
            sid = s["id"]
            self._ranges[sid] = tuple(s.get("normal_range", [-1e9, 1e9]))
            self._critical[sid] = s.get("critical_threshold", float("inf"))

    def check(self, sensor_id: str, value: float) -> int:
        """Returns quality flag: 0=good, 1=suspect, 2=bad/critical."""
        lo, hi = self._ranges.get(sensor_id, (-1e9, 1e9))
        critical = self._critical.get(sensor_id, float("inf"))
        if abs(value) >= critical:
            return 2
        if not (lo <= value <= hi):
            return 1
        return 0


class TelemetryIngestionPipeline:
    """
    Main ingestion pipeline.

    Consumes raw JSON telemetry from Kafka, validates quality,
    writes to TimescaleDB, and forwards to the processed topic
    for downstream windowing/FFT.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)

        kafka_cfg = self._cfg["kafka"]
        ts_cfg = self._cfg["timescaledb"]
        stream_cfg = self._cfg["stream_a"]

        # Kafka consumer
        self._consumer = Consumer(
            {
                "bootstrap.servers": kafka_cfg["bootstrap_servers"],
                "group.id": kafka_cfg["consumer_group"],
                "auto.offset.reset": kafka_cfg["auto_offset_reset"],
                "max.poll.interval.ms": 300000,
                "enable.auto.commit": False,
            }
        )
        self._consumer.subscribe([kafka_cfg["topics"]["telemetry"]])

        # Kafka producer (for processed topic)
        self._producer = Producer(
            {"bootstrap.servers": kafka_cfg["bootstrap_servers"]}
        )
        self._processed_topic = kafka_cfg["topics"]["telemetry_processed"]
        self._alert_topic = kafka_cfg["topics"]["alerts"]

        # TimescaleDB
        dsn = (
            f"host={ts_cfg['host']} port={ts_cfg['port']} "
            f"dbname={ts_cfg['database']} user={ts_cfg['user']} "
            f"password={ts_cfg['password']}"
        )
        self._db = TimescaleDBWriter(dsn, pool_size=ts_cfg["pool_size"])
        self._quality = QualityChecker(stream_cfg["sensors"])

        self._running = False
        self._msgs_processed = 0
        self._msgs_error = 0

        # Register graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Blocking main loop. Runs until SIGTERM/SIGINT."""
        self._running = True
        logger.info("Ingestion pipeline started")
        batch_buffer: List[TelemetryBatch] = []

        while self._running:
            msg = self._consumer.poll(timeout=1.0)

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Kafka error: {msg.error()}")
                self._msgs_error += 1
                continue

            try:
                batch = self._parse_message(msg.value())
                self._apply_quality_flags(batch)
                self._db.write_batch(batch)
                self._forward_processed(batch)
                self._check_alerts(batch)
                self._consumer.commit(asynchronous=False)
                self._msgs_processed += 1

                if self._msgs_processed % 1000 == 0:
                    logger.info(
                        f"Processed {self._msgs_processed} batches | "
                        f"Errors: {self._msgs_error}"
                    )
            except Exception as exc:
                logger.error(f"Failed to process message: {exc}")
                self._msgs_error += 1

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_message(self, raw: bytes) -> TelemetryBatch:
        data = json.loads(raw)
        return TelemetryBatch(
            timestamp=datetime.fromisoformat(data["timestamp"]).replace(
                tzinfo=timezone.utc
            ),
            platform_id=data["platform_id"],
            unit_id=data["unit_id"],
            readings=data["sensors"],
            quality_flags={},
        )

    def _apply_quality_flags(self, batch: TelemetryBatch) -> None:
        for sensor_id, value in batch.readings.items():
            flag = self._quality.check(sensor_id, value)
            if flag > 0:
                batch.quality_flags[sensor_id] = flag

    def _forward_processed(self, batch: TelemetryBatch) -> None:
        payload = json.dumps(
            {
                "timestamp": batch.timestamp.isoformat(),
                "platform_id": batch.platform_id,
                "unit_id": batch.unit_id,
                "sensors": batch.readings,
                "quality_flags": batch.quality_flags,
            }
        ).encode()
        self._producer.produce(self._processed_topic, value=payload)
        self._producer.poll(0)

    def _check_alerts(self, batch: TelemetryBatch) -> None:
        critical = [
            sid
            for sid, flag in batch.quality_flags.items()
            if flag == 2
        ]
        if critical:
            alert = json.dumps(
                {
                    "timestamp": batch.timestamp.isoformat(),
                    "unit_id": batch.unit_id,
                    "type": "sensor_critical",
                    "sensors": critical,
                    "severity": "HIGH",
                }
            ).encode()
            self._producer.produce(self._alert_topic, value=alert)
            logger.warning(
                f"CRITICAL alert — unit {batch.unit_id} — sensors: {critical}"
            )

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self._running = False
        self._consumer.close()
        self._producer.flush(timeout=10)
        self._db.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AeroDeep Stream-A Ingestion")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    pipeline = TelemetryIngestionPipeline(config_path=args.config)
    pipeline.run()

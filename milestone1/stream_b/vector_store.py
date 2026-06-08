"""
milestone1/stream_b/vector_store.py

pgvector-backed vector store for maintenance log embeddings.
Supports upsert, nearest-neighbour search, and metadata filtering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psycopg2
import psycopg2.pool
from psycopg2.extras import execute_values
from loguru import logger

from milestone1.stream_b.embedder import DocumentEmbedding


@dataclass
class SearchResult:
    """A retrieved document with similarity score."""
    source_path: str
    document_type: str
    unit_id: Optional[str]
    date_mentioned: Optional[str]
    component_mentions: List[str]
    chunk_text: Optional[str]
    similarity: float
    content_hash: str


class VectorStore:
    """
    pgvector-backed store for document and chunk embeddings.

    Schema:
      - doc_embeddings: full-document composite embeddings
      - chunk_embeddings: per-chunk (fault/action/component) embeddings
        for fine-grained retrieval

    Supports:
      - Cosine similarity search (via <=> operator)
      - Metadata filtering (unit_id, document_type, date)
      - Hybrid retrieval (combine doc-level and chunk-level scores)
    """

    CREATE_SCHEMA_SQL = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS doc_embeddings (
        id              SERIAL PRIMARY KEY,
        content_hash    TEXT        UNIQUE NOT NULL,
        source_path     TEXT        NOT NULL,
        document_type   TEXT        NOT NULL,
        unit_id         TEXT,
        date_mentioned  TEXT,
        component_mentions  JSONB   DEFAULT '[]',
        embedding       vector({embed_dim})  NOT NULL,
        created_at      TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_doc_emb_unit_id
        ON doc_embeddings (unit_id);
    CREATE INDEX IF NOT EXISTS idx_doc_emb_ivfflat
        ON doc_embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);

    CREATE TABLE IF NOT EXISTS chunk_embeddings (
        id              SERIAL PRIMARY KEY,
        doc_hash        TEXT        NOT NULL REFERENCES doc_embeddings(content_hash),
        chunk_index     INT         NOT NULL,
        chunk_text      TEXT        NOT NULL,
        embedding       vector({embed_dim})  NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_chunk_emb_ivfflat
        ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """

    UPSERT_DOC_SQL = """
    INSERT INTO doc_embeddings
        (content_hash, source_path, document_type, unit_id, date_mentioned,
         component_mentions, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (content_hash) DO UPDATE SET
        source_path = EXCLUDED.source_path,
        document_type = EXCLUDED.document_type,
        unit_id = EXCLUDED.unit_id,
        date_mentioned = EXCLUDED.date_mentioned,
        component_mentions = EXCLUDED.component_mentions,
        embedding = EXCLUDED.embedding;
    """

    UPSERT_CHUNKS_SQL = """
    INSERT INTO chunk_embeddings (doc_hash, chunk_index, chunk_text, embedding)
    VALUES %s
    ON CONFLICT DO NOTHING;
    """

    def __init__(self, dsn: str, embed_dim: int, pool_size: int = 5):
        self._embed_dim = embed_dim
        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2, maxconn=pool_size, dsn=dsn
        )
        self._ensure_schema()
        logger.info(f"VectorStore ready — embed_dim={embed_dim}")

    def _ensure_schema(self) -> None:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                sql = self.CREATE_SCHEMA_SQL.format(embed_dim=self._embed_dim)
                cur.execute(sql)
            conn.commit()
        finally:
            self._pool.putconn(conn)

    def upsert(self, doc_emb: DocumentEmbedding) -> None:
        """Store or update a document embedding and its chunks."""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                # Upsert document-level embedding
                cur.execute(
                    self.UPSERT_DOC_SQL,
                    (
                        doc_emb.content_hash,
                        doc_emb.source_path,
                        doc_emb.document_type,
                        doc_emb.unit_id,
                        doc_emb.date_mentioned,
                        json.dumps(doc_emb.component_mentions),
                        doc_emb.composite_embedding.tolist(),
                    ),
                )

                # Upsert chunk embeddings
                if doc_emb.chunk_embeddings:
                    chunk_rows = [
                        (
                            doc_emb.content_hash,
                            i,
                            doc_emb.chunk_texts[i],
                            emb.tolist(),
                        )
                        for i, emb in enumerate(doc_emb.chunk_embeddings)
                    ]
                    execute_values(cur, self.UPSERT_CHUNKS_SQL, chunk_rows)

            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error(f"VectorStore upsert failed: {exc}")
            raise
        finally:
            self._pool.putconn(conn)

    def search_documents(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        unit_id: Optional[str] = None,
        document_type: Optional[str] = None,
        min_similarity: float = 0.5,
    ) -> List[SearchResult]:
        """
        Nearest-neighbour search over document-level embeddings.
        Optionally filter by unit_id and document_type.
        """
        where_clauses = ["1 - (embedding <=> %s::vector) >= %s"]
        params: List[Any] = [query_embedding.tolist(), min_similarity]

        if unit_id:
            where_clauses.append("unit_id = %s")
            params.append(unit_id)
        if document_type:
            where_clauses.append("document_type = %s")
            params.append(document_type)

        params.append(top_k)
        sql = f"""
        SELECT
            source_path, document_type, unit_id, date_mentioned,
            component_mentions, content_hash,
            1 - (embedding <=> %s::vector) AS similarity
        FROM doc_embeddings
        WHERE {' AND '.join(where_clauses)}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        # Add query embedding again for ORDER BY
        params_full = [query_embedding.tolist()] + params[:-1] + [query_embedding.tolist(), top_k]

        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params_full)
                rows = cur.fetchall()
        finally:
            self._pool.putconn(conn)

        return [
            SearchResult(
                source_path=row[0],
                document_type=row[1],
                unit_id=row[2],
                date_mentioned=row[3],
                component_mentions=json.loads(row[4]) if row[4] else [],
                chunk_text=None,
                similarity=float(row[6]),
                content_hash=row[5],
            )
            for row in rows
        ]

    def search_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        unit_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Fine-grained chunk-level search — best for specific fault queries."""
        params: List[Any] = [query_embedding.tolist()]
        join_filter = ""
        if unit_id:
            join_filter = "AND d.unit_id = %s"
            params.append(unit_id)

        params.extend([query_embedding.tolist(), top_k])

        sql = f"""
        SELECT
            d.source_path, d.document_type, d.unit_id, d.date_mentioned,
            d.component_mentions, d.content_hash, c.chunk_text,
            1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunk_embeddings c
        JOIN doc_embeddings d ON d.content_hash = c.doc_hash
        WHERE 1=1 {join_filter}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s;
        """

        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            self._pool.putconn(conn)

        return [
            SearchResult(
                source_path=row[0],
                document_type=row[1],
                unit_id=row[2],
                date_mentioned=row[3],
                component_mentions=json.loads(row[4]) if row[4] else [],
                content_hash=row[5],
                chunk_text=row[6],
                similarity=float(row[7]),
            )
            for row in rows
        ]

    def close(self) -> None:
        self._pool.closeall()

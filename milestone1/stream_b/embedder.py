"""
milestone1/stream_b/embedder.py

Generates dense semantic embeddings from cleaned maintenance logs and
shift reports using a domain-appropriate sentence-transformer model.

Model selection rationale:
  - "sentence-transformers/all-roberta-large-v1": strong general technical
    text, 1024-dim, good semantic clustering
  - "BAAI/bge-large-en-v1.5": state-of-art retrieval, better for RAG
    (Retrieval Augmented Generation) queries against the log history
  Both are configurable in config.yaml.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger

from milestone1.stream_b.log_cleaner import CleanedDocument


@dataclass
class DocumentEmbedding:
    """
    Semantic embedding for a maintenance document.
    Includes both a full-document embedding and chunk-level embeddings
    for fine-grained retrieval.
    """
    source_path: str
    document_type: str
    unit_id: Optional[str]
    date_mentioned: Optional[str]
    component_mentions: List[str]

    # Full document embedding
    full_embedding: np.ndarray          # shape (embed_dim,)

    # Chunk-level: fault descriptions + actions as separate embeddings
    chunk_embeddings: List[np.ndarray]
    chunk_texts: List[str]

    # Composite embedding: mean of full + salient chunks
    composite_embedding: np.ndarray     # shape (embed_dim,)

    embed_dim: int
    model_name: str
    content_hash: str


class IndustrialEmbedder:
    """
    Sentence-transformer based embedder tuned for industrial maintenance text.

    Batches documents for GPU efficiency. Handles text that exceeds the
    model's max token length by splitting into overlapping chunks and
    mean-pooling the chunk embeddings.
    """

    DEFAULT_MODEL = "sentence-transformers/all-roberta-large-v1"
    CHUNK_OVERLAP_TOKENS = 64

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
        normalize_embeddings: bool = True,
    ):
        device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model '{model_name}' on {device}")

        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length
        self._normalize = normalize_embeddings
        self._embed_dim = self._model.get_sentence_embedding_dimension()

        logger.info(
            f"Embedder ready — dim={self._embed_dim} device={device} "
            f"max_length={max_length}"
        )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed_document(self, doc: CleanedDocument) -> DocumentEmbedding:
        """Embed a single cleaned document into a DocumentEmbedding."""
        content_hash = _content_hash(doc.cleaned_text)

        # Full document: may need chunking if long
        full_emb = self._embed_long_text(doc.cleaned_text)

        # Chunk-level: embed fault descriptions + actions individually
        salient_chunks = []
        chunk_texts = []

        # Build salient text from structured extraction
        if doc.fault_descriptions:
            fault_text = "Fault context: " + "; ".join(doc.fault_descriptions)
            salient_chunks.append(fault_text)
            chunk_texts.append(fault_text)

        if doc.actions_taken:
            action_text = "Actions taken: " + "; ".join(doc.actions_taken)
            salient_chunks.append(action_text)
            chunk_texts.append(action_text)

        if doc.component_mentions:
            comp_text = "Components involved: " + ", ".join(doc.component_mentions)
            salient_chunks.append(comp_text)
            chunk_texts.append(comp_text)

        # Embed all salient chunks in one batch
        if salient_chunks:
            chunk_embs = self._model.encode(
                salient_chunks,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        else:
            chunk_embs = np.empty((0, self._embed_dim), dtype=np.float32)

        # Composite: weighted mean of full doc + salient chunks
        # Salient chunks carry more fault-specific signal — upweight them
        if len(chunk_embs) > 0:
            salient_mean = np.mean(chunk_embs, axis=0)
            composite = 0.4 * full_emb + 0.6 * salient_mean
            if self._normalize:
                composite /= np.linalg.norm(composite) + 1e-8
        else:
            composite = full_emb

        return DocumentEmbedding(
            source_path=doc.source_path,
            document_type=doc.document_type,
            unit_id=doc.unit_id,
            date_mentioned=doc.date_mentioned,
            component_mentions=doc.component_mentions,
            full_embedding=full_emb,
            chunk_embeddings=list(chunk_embs) if len(chunk_embs) > 0 else [],
            chunk_texts=chunk_texts,
            composite_embedding=composite,
            embed_dim=self._embed_dim,
            model_name=self._model_name,
            content_hash=content_hash,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string for similarity search against the vector store."""
        return self._model.encode(
            query,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def embed_documents_batch(
        self, docs: List[CleanedDocument]
    ) -> List[DocumentEmbedding]:
        """Batch-embed a list of documents efficiently."""
        results = []
        for doc in docs:
            results.append(self.embed_document(doc))
        return results

    # ── Private ───────────────────────────────────────────────────────────────

    def _embed_long_text(self, text: str) -> np.ndarray:
        """
        Handle text that may exceed the model's max token length.
        Split into overlapping chunks and mean-pool.
        """
        # Simple word-based chunking (heuristic: ~1.3 tokens/word)
        words = text.split()
        max_words = int(self._max_length / 1.3)
        overlap_words = int(self.CHUNK_OVERLAP_TOKENS / 1.3)

        if len(words) <= max_words:
            return self._model.encode(
                text,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        # Split into overlapping chunks
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunks.append(" ".join(words[start:end]))
            start += max_words - overlap_words

        chunk_embs = self._model.encode(
            chunks,
            batch_size=self._batch_size,
            normalize_embeddings=False,   # normalise after pooling
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        pooled = np.mean(chunk_embs, axis=0)
        if self._normalize:
            pooled /= np.linalg.norm(pooled) + 1e-8
        return pooled


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

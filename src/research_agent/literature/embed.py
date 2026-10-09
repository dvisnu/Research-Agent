"""SPECTER2 embeddings, computed locally on CPU.

Paper vectors use the proximity adapter on "title [SEP] abstract", which is
what Semantic Scholar serves as ``embedding.specter_v2`` (checked: cosine 1.0).
Queries use the ad-hoc query adapter, made for short search queries in the
same space. The model (about 440 MB) downloads on first use.
"""

import os
from typing import Protocol

import numpy as np

BASE_MODEL = "allenai/specter2_base"
PAPER_ADAPTER = "allenai/specter2"
QUERY_ADAPTER = "allenai/specter2_adhoc_query"
BATCH_SIZE = 16


class Embedder(Protocol):
    def papers(self, items: list[tuple[str, str | None]]) -> np.ndarray:
        """Unit vectors for (title, abstract) pairs, one row each."""
        ...

    def query(self, text: str) -> np.ndarray:
        """A unit vector for a search query."""
        ...


class Specter2:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer
        from transformers.utils import logging

        logging.set_verbosity_error()  # silences "adapters available but none activated" at load
        self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoAdapterModel.from_pretrained(BASE_MODEL)
        model.load_adapter(PAPER_ADAPTER, source="hf", load_as="paper", set_active=False)
        model.load_adapter(QUERY_ADAPTER, source="hf", load_as="query", set_active=False)
        model.eval()
        self._model = model

    def papers(self, items: list[tuple[str, str | None]]) -> np.ndarray:
        self._load()
        sep = self._tokenizer.sep_token
        texts = [title + sep + (abstract or "") for title, abstract in items]
        return self._encode(texts, "paper")

    def query(self, text: str) -> np.ndarray:
        self._load()
        return self._encode([text], "query")[0]

    def _encode(self, texts: list[str], adapter: str) -> np.ndarray:
        import torch

        self._model.set_active_adapters(adapter)
        rows = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = self._tokenizer(
                texts[start : start + BATCH_SIZE],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            with torch.no_grad():
                rows.append(self._model(**batch).last_hidden_state[:, 0, :].numpy())
        if not rows:
            return np.zeros((0, 768), dtype=np.float32)
        vectors = np.concatenate(rows)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

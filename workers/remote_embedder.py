"""
remote_embedder.py
------------------
AKL Pipeline — shared embedding backend for Steps 3 and 6.

Computes document embeddings either locally (SentenceTransformer) or on a
remote OpenAI-compatible server (vLLM in embedding mode, /v1/embeddings),
selected by the `embedding_server` section of pipeline_config.yaml.

RemoteEmbedder subclasses bertopic's BaseEmbedder, so it can be passed directly
to BERTopic(embedding_model=...) and BERTopic.load(embedding_model=...), and it
also exposes a SentenceTransformer-style `encode()` used by the workers.
"""

import sys
import time
from typing import List, Optional

import numpy as np
from bertopic.backend import BaseEmbedder


class RemoteEmbedder(BaseEmbedder):
    """Embeds texts through a remote /v1/embeddings endpoint."""

    def __init__(self, base_url: str, model: str, batch_size: int = 64,
                 max_tokens: Optional[int] = 512, timeout: float = 120.0,
                 prompt: Optional[str] = None, max_retries: int = 3,
                 logger=None):
        super().__init__()
        import logging
        import openai
        # httpx logs one INFO line per request (hundreds per run); the progress lines
        # below replace it.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.base_url = base_url
        self.model = model
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.prompt = prompt or ""
        self.max_retries = max_retries
        self.logger = logger
        self.client = openai.OpenAI(api_key="not-needed", base_url=base_url,
                                    timeout=timeout, max_retries=0)

    # ── BaseEmbedder API ──────────────────────────────────────────────────────

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        texts = [self.prompt + str(d) for d in documents]
        total = len(texts)
        vectors = []

        # Progress: a live tqdm bar on a real terminal; otherwise (background runs,
        # logs) one line every 10%, so execution.log stays readable.
        bar = None
        if verbose and sys.stderr.isatty():
            from tqdm import tqdm
            bar = tqdm(total=total, desc="Embedding", unit="doc")
        next_log = 10
        started = time.time()

        for start in range(0, total, self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
            done = min(start + self.batch_size, total)
            if bar is not None:
                bar.update(len(batch))
            elif verbose and self.logger and total and done * 100 // total >= next_log:
                elapsed = time.time() - started
                eta = elapsed / done * (total - done)
                self.logger.info(f"  embedding: {done}/{total} ({done * 100 // total}%) "
                                 f"— {elapsed:.0f}s elapsed, ~{eta:.0f}s left")
                next_log = (done * 100 // total // 10 + 1) * 10
        if bar is not None:
            bar.close()
        return np.asarray(vectors, dtype=np.float32)

    # ── SentenceTransformer-style alias ───────────────────────────────────────

    def encode(self, sentences, show_progress_bar: bool = False, **kwargs) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
        return self.embed(list(sentences), verbose=show_progress_bar)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _embed_batch(self, batch: List[str]) -> list:
        extra = {"truncate_prompt_tokens": self.max_tokens} if self.max_tokens else None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.embeddings.create(
                    model=self.model, input=batch, extra_body=extra)
                return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
            except Exception as e:
                # 4xx (bad request / unknown model) will not succeed on retry
                client_error = getattr(e, "status_code", 500) < 500
                if attempt == self.max_retries or client_error:
                    raise RuntimeError(
                        f"Embedding server {self.base_url} failed "
                        f"(attempt {attempt}/{self.max_retries}): {e}") from e
                if self.logger:
                    self.logger.warning(
                        f"Embedding request failed (attempt {attempt}): {e} — retrying.")
                time.sleep(2 ** attempt)


def is_remote(cfg: dict) -> bool:
    """True when the config selects the remote embedding server."""
    return bool((cfg.get("embedding_server") or {}).get("enabled", False))


def build_embedding_model(config: dict, logger=None, prompt: Optional[str] = None,
                          device: Optional[str] = None):
    """
    Return the embedding model for BERTopic: a RemoteEmbedder when
    `embedding_server.enabled` is true, otherwise a local SentenceTransformer
    on `device` (default: `step3_bertopic.embedding_device`).

    There is no silent fallback: if the remote server is enabled but not
    reachable, a RuntimeError is raised (local and remote vectors differ slightly,
    so mixing them in one run would be inconsistent).
    """
    models_cfg = config.get("models", {})
    name = models_cfg.get("embedding",
                          "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    srv = config.get("embedding_server") or {}

    if srv.get("enabled", False):
        base_url = srv["base_url"]
        remote = RemoteEmbedder(
            base_url=base_url,
            model=srv.get("model") or name,
            batch_size=srv.get("batch_size", 64),
            max_tokens=srv.get("max_tokens", 512),
            timeout=srv.get("timeout", 120),
            prompt=prompt,
            logger=logger,
        )
        if logger:
            logger.info(f"Embedding backend: remote server {base_url} (model {remote.model})")
        try:
            remote.embed(["connectivity check"])
        except Exception as e:
            raise RuntimeError(
                f"Embedding server not reachable at {base_url}. Check the VPN and "
                f"that the container is running, or set embedding_server.enabled: false "
                f"to compute embeddings locally. Details: {e}") from e
        return remote

    from sentence_transformers import SentenceTransformer
    device = device or config.get("step3_bertopic", {}).get("embedding_device", "cpu")
    if logger:
        logger.info(f"Embedding backend: local SentenceTransformer {name} on {device}")
    if prompt:
        return SentenceTransformer(name, prompts={"encode": prompt},
                                   default_prompt_name="encode", device=device)
    return SentenceTransformer(name, device=device)


def embedding_backend_id(config: dict) -> str:
    """Identifier of the embedding backend, for the embeddings cache checksum."""
    srv = config.get("embedding_server") or {}
    if srv.get("enabled", False):
        return f"remote:{srv.get('base_url', '')}:max_tokens={srv.get('max_tokens', 512)}"
    return "local"

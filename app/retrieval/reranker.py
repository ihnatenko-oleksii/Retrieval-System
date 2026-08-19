import logging
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from app.core.config import settings

logger = logging.getLogger(__name__)


DEFAULT_QWEN3_RERANKER_INSTRUCTION = (
    "Given a technical support or engineering question, retrieve passages that directly answer the question "
    "or contain the necessary implementation details."
)


class RerankerUnavailable(RuntimeError):
    """Raised when an explicitly requested reranker cannot load or infer."""


class Qwen3RerankerModel:
    """Small Transformers implementation of the official Qwen3 reranker recipe."""

    def __init__(
        self,
        model_name: str,
        *,
        instruction: str = DEFAULT_QWEN3_RERANKER_INSTRUCTION,
        max_length: int = 8192,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        except Exception as exc:
            raise RerankerUnavailable(
                f"Qwen3 reranker {model_name} failed to load: {type(exc).__name__}: {exc}"
            ) from exc

        self.model_name = model_name
        self.instruction = instruction
        self.max_length = max_length
        self.device = next(self.model.parameters()).device
        self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        if self.true_token_id is None or self.false_token_id is None:
            raise RerankerUnavailable(f"Qwen3 reranker {model_name} has no yes/no scorer tokens")

        prefix = (
            '<|im_start|>system\n'
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            '<|im_start|>user\n'
        )
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    @staticmethod
    def _format_pair(instruction: str, query: str, document: str) -> str:
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"

    def _process_inputs(self, texts: list[str]) -> Any:
        inputs = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for index, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = self.prefix_tokens + input_ids + self.suffix_tokens
        padded = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        return {key: value.to(self.device) for key, value in padded.items()}

    def predict(
        self,
        pairs: list[tuple[str, str]] | list[list[str]],
        *,
        batch_size: int = 16,
    ) -> np.ndarray:
        scores: list[float] = []
        try:
            with self._torch.inference_mode():
                for start in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[start : start + batch_size]
                    texts = [
                        self._format_pair(self.instruction, str(query), str(document))
                        for query, document in batch_pairs
                    ]
                    inputs = self._process_inputs(texts)
                    # Qwen3 exposes logits_to_keep; computing only the final
                    # scorer position avoids materializing logits for every
                    # token in the long candidate sequence.
                    logits = self.model(**inputs, logits_to_keep=1).logits[:, -1, :]
                    selected = logits[:, [self.false_token_id, self.true_token_id]]
                    probabilities = self._torch.log_softmax(selected, dim=1)[:, 1].exp()
                    scores.extend(float(score) for score in probabilities.detach().cpu())
        except Exception as exc:
            raise RerankerUnavailable(
                f"Qwen3 reranker {self.model_name} inference failed: {type(exc).__name__}: {exc}"
            ) from exc
        return np.asarray(scores, dtype=np.float32)


class Reranker:
    def __init__(self, model_name: str | None = None, force_load: bool = False):
        self.model_name = model_name or settings.reranker_model
        self.model = None
        # Load eagerly either when enabled in settings OR when force_load=True
        # (used by the tuning UI which flips the flag per trial).
        if settings.reranker_on or force_load:
            self._load_model()

    def _load_model(self):
        if self.model is not None:
            return
        try:
            if self.model_name.lower().startswith("qwen/qwen3-reranker"):
                self.model = Qwen3RerankerModel(self.model_name)
                logger.info("Loaded Transformers Qwen3 reranker: %s", self.model_name)
            else:
                self.model = CrossEncoder(self.model_name, max_length=512)
                logger.info("Loaded CrossEncoder reranker: %s", self.model_name)
        except Exception as e:
            logger.error("Failed to load reranker %s: %s", self.model_name, e)
            if self.model_name.lower().startswith("qwen/qwen3-reranker"):
                if isinstance(e, RerankerUnavailable):
                    raise
                raise RerankerUnavailable(
                    f"Qwen3 reranker {self.model_name} failed to load: {type(e).__name__}: {e}"
                ) from e

    def rerank(
        self,
        query: str,
        retrieved_chunks: list[tuple[str, dict, float]],
        top_n: int = None,
        enabled: bool | None = None,
    ) -> list[tuple[str, dict, float]]:
        rerank_enabled = settings.reranker_on if enabled is None else bool(enabled)
        if not rerank_enabled or not retrieved_chunks:
            return retrieved_chunks[:top_n] if top_n else retrieved_chunks

        if self.model is None:
            self._load_model()
        if self.model is None:
            if self.model_name.lower().startswith("qwen/qwen3-reranker"):
                raise RerankerUnavailable(f"Qwen3 reranker {self.model_name} is unavailable")
            return retrieved_chunks[:top_n] if top_n else retrieved_chunks

        if top_n is None:
            top_n = settings.rerank_top_n

        contents = [chunk[0] for chunk in retrieved_chunks]
        query_document_pairs = [[query, content] for content in contents]

        try:
            scores = self.model.predict(query_document_pairs)

            scored_chunks = []
            for i, chunk in enumerate(retrieved_chunks):
                content, meta, _ = chunk
                scored_chunks.append((content, meta, float(scores[i])))

            reranked = sorted(scored_chunks, key=lambda x: x[2], reverse=True)
            return reranked[:top_n]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            if self.model_name.lower().startswith("qwen/qwen3-reranker"):
                if isinstance(e, RerankerUnavailable):
                    raise
                raise RerankerUnavailable(
                    f"Qwen3 reranker {self.model_name} inference failed: {type(e).__name__}: {e}"
                ) from e
            return retrieved_chunks[:top_n] if top_n else retrieved_chunks

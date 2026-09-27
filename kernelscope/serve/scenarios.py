"""Deterministic arrivals with synthetic tokens or locally resolved raw text.

These traces replay the same batch composition across policies. They are not an
open-loop, wall-clock arrival benchmark or a natural-language quality evaluation.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from numbers import Integral
import time

import numpy as np
import yaml


@dataclass(frozen=True)
class Request:
    rid: int
    prompt_len: int | None
    max_new_tokens: int
    arrival_step: int = 0
    prompt_text: str | None = None
    token_ids: tuple[int, ...] | None = None
    prompt_recipe: str | None = None
    corpus: tuple[str, ...] = ()
    prompt_suffix: str = ""
    expected_answer: str | None = None
    prompt_kind: str | None = None
    prompt_sha256: str | None = None

    def __post_init__(self):
        for name in ("rid", "prompt_len", "max_new_tokens", "arrival_step"):
            value = getattr(self, name)
            if name == "prompt_len" and value is None and (self.prompt_text is not None or self.token_ids is not None):
                continue
            minimum = 1 if name in ("prompt_len", "max_new_tokens") else 0
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
        modes = sum(value is not None for value in (self.prompt_text, self.token_ids, self.prompt_recipe))
        if modes > 1:
            raise ValueError("provide only one of prompt_text, token_ids, or prompt_recipe")
        if self.prompt_text is not None and (not isinstance(self.prompt_text, str) or not self.prompt_text.strip()):
            raise ValueError("prompt_text must contain non-whitespace text")
        if self.token_ids is not None:
            if not self.token_ids or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in self.token_ids):
                raise ValueError("token_ids must contain nonnegative integers")
            object.__setattr__(self, "token_ids", tuple(int(x) for x in self.token_ids))
            if self.prompt_len is None:
                object.__setattr__(self, "prompt_len", len(self.token_ids))
            if self.prompt_len != len(self.token_ids):
                raise ValueError("prompt_len must equal the number of explicit token_ids")
        if self.prompt_recipe is not None:
            if self.prompt_recipe != "corpus_repeat" or not self.corpus:
                raise ValueError("prompt_recipe must be corpus_repeat with a nonempty corpus")
            if any(not isinstance(text, str) or not text.strip() for text in self.corpus):
                raise ValueError("corpus must contain nonempty original text paragraphs")
        if not isinstance(self.prompt_suffix, str) or (self.prompt_suffix and self.prompt_recipe is None):
            raise ValueError("prompt_suffix is supported only by corpus_repeat recipes")
        if self.expected_answer is not None and (not isinstance(self.expected_answer, str) or not self.expected_answer.strip()):
            raise ValueError("expected_answer must be nonempty text")


def load(path) -> list[Request]:
    return load_scenario(path)[0]


def load_scenario(path) -> tuple[list[Request], dict]:
    with open(path) as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict) or not isinstance(spec.get("requests"), list) or not spec["requests"]:
        raise ValueError("scenario must contain a non-empty requests list")
    dataset = spec.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a metadata mapping")
    corpus = spec.get("corpus", [])
    if not isinstance(corpus, list) or any(not isinstance(text, str) or not text.strip() for text in corpus):
        raise ValueError("corpus must be a list of nonempty text paragraphs")
    out = []
    for group in spec["requests"]:
        if not isinstance(group, dict):
            raise ValueError("each requests entry must be a mapping")
        unknown = set(group) - {"count", "prompt_len", "max_new_tokens", "arrival_step", "prompt_text", "token_ids",
                                "prompt_recipe", "prompt_suffix", "expected_answer"}
        if unknown or "max_new_tokens" not in group:
            raise ValueError(f"invalid request group fields: {group}")
        count = group.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("request count must be a positive integer")
        for _ in range(count):
            out.append(Request(len(out), group.get("prompt_len"), group["max_new_tokens"], group.get("arrival_step", 0),
                               prompt_text=group.get("prompt_text"), token_ids=group.get("token_ids"),
                               prompt_recipe=group.get("prompt_recipe"), corpus=tuple(corpus) if group.get("prompt_recipe") else (),
                               prompt_suffix=group.get("prompt_suffix", ""), expected_answer=group.get("expected_answer")))
    return out, dataset


def prompt_ids(req: Request, vocab: int, seed: int = 0) -> list[int]:
    if isinstance(vocab, bool) or not isinstance(vocab, Integral) or vocab < 2:
        raise ValueError("vocab must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if req.token_ids is not None:
        if any(token >= vocab for token in req.token_ids):
            raise ValueError(f"request {req.rid} has a token outside model vocabulary [0, {vocab})")
        return list(req.token_ids)
    if req.prompt_text is not None or req.prompt_recipe is not None:
        raise ValueError("text requests must be resolved with the local tokenizer before Engine.run")
    rng = np.random.default_rng(np.random.SeedSequence([seed, req.rid]))
    lo, hi = (1000, vocab - 1000) if vocab > 2000 else (0, vocab)
    return rng.integers(lo, hi, size=req.prompt_len).tolist()


def _hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def needs_tokenizer(requests) -> bool:
    return any(request.prompt_text is not None or request.prompt_recipe is not None for request in requests)


def resolve_requests(requests, vocab, tokenizer=None, seed=0, dataset=None):
    """Prepare every prompt outside GPU timing, retaining source and token identities.

    Text is tokenized with add_special_tokens=False; no chat template is assumed.
    corpus_repeat repeats a seeded paragraph order, truncates its token stream to
    the requested length minus the suffix, and appends the intact suffix tokens.
    """
    started = time.perf_counter()
    requests = list(requests)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if needs_tokenizer(requests) and tokenizer is None:
        raise ValueError("natural-text scenarios require the local model tokenizer")
    if len({request.rid for request in requests}) != len(requests):
        raise ValueError("request IDs must be unique")
    resolved, descriptions = [], []
    for request in requests:
        source = asdict(request)
        if request.prompt_recipe == "corpus_repeat":
            rng = np.random.default_rng(np.random.SeedSequence([seed, request.rid]))
            paragraphs = [request.corpus[i] for i in rng.permutation(len(request.corpus))]
            body = tokenizer.encode("\n\n".join(paragraphs) + "\n\n", add_special_tokens=False).ids
            suffix = tokenizer.encode(request.prompt_suffix, add_special_tokens=False).ids if request.prompt_suffix else []
            target = request.prompt_len - len(suffix)
            if not body or target < 1:
                raise ValueError("corpus must tokenize nonempty and prompt_len must leave room beyond prompt_suffix")
            ids = (body * ((target + len(body) - 1) // len(body)))[:target] + suffix
            kind = "constructed_natural_text"
        elif request.prompt_text is not None:
            ids = tokenizer.encode(request.prompt_text, add_special_tokens=False).ids
            if not ids:
                raise ValueError("prompt_text tokenized to an empty sequence")
            if request.prompt_len is not None and request.prompt_len != len(ids):
                raise ValueError(f"request {request.rid}: prompt_text has {len(ids)} tokens, not prompt_len={request.prompt_len}; omit prompt_len or use corpus_repeat")
            kind = "constructed_natural_text"
        else:
            ids = prompt_ids(request, vocab, seed)
            kind = request.prompt_kind or ("explicit_token_ids" if request.token_ids is not None else "seeded_synthetic_token_ids")
        resolved_request = replace(request, prompt_len=len(ids), token_ids=tuple(ids), prompt_text=None,
                                   prompt_recipe=None, corpus=(), prompt_suffix="", prompt_kind=kind,
                                   prompt_sha256=_hash(ids))
        prompt_ids(resolved_request, vocab, seed)  # Vocabulary validation before any measured run.
        resolved.append(resolved_request)
        descriptions.append({"rid": request.rid, "prompt_kind": kind, "source_sha256": _hash(source),
                             "prompt_sha256": resolved_request.prompt_sha256,
                             "requested_prompt_len": request.prompt_len, "resolved_prompt_len": len(ids),
                             "expected_answer": request.expected_answer})
    kinds = {request.prompt_kind for request in resolved}
    metadata = {"dataset": dict(dataset or {}), "dataset_id": (dataset or {}).get("id", "inline_requests"),
                "evaluation_split": (dataset or {}).get("evaluation_split", "unspecified"),
                "scenario_family": (dataset or {}).get("scenario_family", "unspecified"),
                "dataset_sha256": _hash({"dataset": dataset or {}, "requests": [asdict(r) for r in requests]}),
                "resolved_prompts_sha256": _hash([{k: item[k] for k in ("rid", "prompt_sha256")} for item in descriptions]),
                "prompt_kind": next(iter(kinds)) if len(kinds) == 1 else "mixed",
                "prompt_mode": "raw_text_continuation_no_chat_template" if "constructed_natural_text" in kinds else "token_ids",
                "prompt_resolution": "prepared_once_before_warmup_and_measurement", "prompts": descriptions,
                "tokenization_setup_us": (time.perf_counter() - started) * 1e6}
    return resolved, metadata

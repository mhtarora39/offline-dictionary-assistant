"""Term extraction modules using local Ollama models and rule-based heuristics."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from db import get_db_connection, get_known_forms, is_known_form, normalize_term

INTENT_WORDS = {
    "tell",
    "me",
    "what",
    "is",
    "does",
    "mean",
    "meaning",
    "meanings",
    "define",
    "definition",
    "definitions",
    "of",
    "the",
    "word",
    "words",
    "phrase",
    "term",
    "a",
    "an",
    "please",
    "can",
    "you",
    "could",
    "give",
    "about",
    "for",
    "in",
    "to",
    "do",
    "are",
    "s",
    "explain",
    # Common typos and contractions of intent words
    "ix",
    "iz",
    "si",
    "wat",
    "waht",
    "wht",
    "whot",
    "wats",
    "whats",
    "what's",
    "defin",
    "defne",
    "dafine",
    "mening",
    "meen",
    "meanin",
    "meanning",
    "teh",
    "dus",
    "dose",
    "pls",
    "plz",
    "abt",
    "whos",
    "who's",
    "who",
}

STRUCTURED_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "term": {"type": ["string", "null"]}
    },
    "required": ["term"],
}

FEW_SHOT_MESSAGES = [
    {
        "role": "system",
        "content": (
            'Return the word or phrase to define copied exactly as typed (keep typos), '
            'or null if there is no word to define. '
            'Output JSON only: {"term": "<exact text from the query>"} or {"term": null}. '
            'Do not invent any term.'
        ),
    },
    {"role": "user", "content": "tell me what is black mean"},
    {"role": "assistant", "content": '{"term": "black"}'},
    {"role": "user", "content": "what does kick the bucket mean"},
    {"role": "assistant", "content": '{"term": "kick the bucket"}'},
    {"role": "user", "content": "what's the weather today"},
    {"role": "assistant", "content": '{"term": null}'},
    {"role": "user", "content": "define serendipity please"},
    {"role": "assistant", "content": '{"term": "serendipity"}'},
    {"role": "user", "content": "how are you feeling today"},
    {"role": "assistant", "content": '{"term": null}'},
    {"role": "user", "content": "meaning of elefant please"},
    {"role": "assistant", "content": '{"term": "elefant"}'},
]


@dataclass
class ExtractResult:
    """Result of term extraction."""

    term: Optional[str]
    raw_output: Any
    meta: Dict[str, Any] = field(default_factory=dict)


def validate_term(term: Optional[str], query: str) -> Tuple[bool, Optional[str]]:
    """Validate that the returned term (lowercased) is a substring of the lowercased query.

    Returns (is_valid, normalized_term_or_none). Never accepts invented terms.
    """
    if term is None:
        return True, None

    cleaned = term.strip().strip("'\"`“”‘’.,!?:;()[]{}*~_#")
    if not cleaned:
        return True, None

    # Substring check on lowercased strings
    if cleaned.lower() in query.lower():
        return True, cleaned
    return False, None


class RuleExtractor:
    """Rule-based extractor baseline requiring no model.

    Algorithm:
    1. Find the longest n-gram (max 4 tokens) of the query that is a form in the DB
       (skipping n-grams composed entirely of intent words).
    2. Otherwise, strip intent words and use the remainder.
    3. If several tokens remain, prefer ones found in the DB, last content word first.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path

    def _tokenize(self, text: str) -> List[str]:
        # Split into alphanumeric tokens, preserving contractions/hyphens if appropriate
        tokens = re.findall(r"[A-Za-z0-9'-]+", text)
        cleaned = [t.strip("'-").lower() for t in tokens if t.strip("'-")]
        return cleaned

    def extract(self, query: str) -> ExtractResult:
        tokens = self._tokenize(query)
        if not tokens:
            return ExtractResult(term=None, raw_output=None, meta={"extractor": "rules", "validation_ok": True})

        with get_db_connection(self.db_path) as conn:
            # 1. Longest n-gram (max 4 tokens) that is a headword/form in the DB
            n_tokens = len(tokens)
            for n in range(min(4, n_tokens), 0, -1):
                for i in range(n_tokens - n + 1):
                    ngram_tokens = tokens[i : i + n]
                    # Skip if the n-gram consists entirely of intent words
                    if all(t in INTENT_WORDS for t in ngram_tokens):
                        continue
                    ngram_candidate = " ".join(ngram_tokens)
                    if is_known_form(ngram_candidate, conn=conn):
                        return ExtractResult(
                            term=ngram_candidate,
                            raw_output=ngram_candidate,
                            meta={"extractor": "rules", "step": 1, "validation_ok": True},
                        )

            # 2. Otherwise strip intent words and use remainder
            remainder = [t for t in tokens if t not in INTENT_WORDS]
            if not remainder:
                return ExtractResult(
                    term=None,
                    raw_output=None,
                    meta={"extractor": "rules", "step": 2, "validation_ok": True},
                )

            if len(remainder) == 1:
                return ExtractResult(
                    term=remainder[0],
                    raw_output=remainder[0],
                    meta={"extractor": "rules", "step": 2, "validation_ok": True},
                )

            # 3. If several tokens remain, prefer ones found in the DB, last content word first
            known = get_known_forms(remainder, conn=conn)
            for token in reversed(remainder):
                if token in known:
                    return ExtractResult(
                        term=token,
                        raw_output=token,
                        meta={"extractor": "rules", "step": 3, "validation_ok": True},
                    )

            # Fallback to last content word if none in DB
            chosen = remainder[-1]
            return ExtractResult(
                term=chosen,
                raw_output=chosen,
                meta={"extractor": "rules", "step": 3, "fallback_last": True, "validation_ok": True},
            )


class OllamaExtractor:
    """LLM-based extractor calling Ollama chat API with structured output."""

    def __init__(
        self,
        model_name: str,
        ollama_tag: str,
        options: Optional[Dict[str, Any]] = None,
        think: Optional[bool] = None,
        ollama_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.ollama_tag = ollama_tag
        self.options = options or {
            "temperature": 0,
            "seed": 0,
            "num_ctx": 1024,
            "num_predict": 32,
        }
        self.think = think
        self.ollama_url = ollama_url.rstrip("/")

    def warmup(self) -> None:
        """Warm up the model to preload weights before timing queries."""
        try:
            self._call_chat("ping")
        except Exception:
            pass

    def _call_chat(self, query: str) -> Dict[str, Any]:
        messages = list(FEW_SHOT_MESSAGES)
        messages.append({"role": "user", "content": query})

        payload: Dict[str, Any] = {
            "model": self.ollama_tag,
            "messages": messages,
            "stream": False,
            "format": STRUCTURED_FORMAT_SCHEMA,
            "options": self.options,
            "keep_alive": "10m",
        }
        # Only send think when explicitly specified
        if self.think is not None:
            payload["think"] = self.think

        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def extract(self, query: str) -> ExtractResult:
        t0 = time.perf_counter()
        raw_response = self._call_chat(query)
        t_extract_ms = round((time.perf_counter() - t0) * 1000, 2)

        content = raw_response.get("message", {}).get("content", "")
        raw_term: Optional[str] = None
        parsed_ok = False

        if content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "term" in parsed:
                    raw_term = parsed["term"]
                    parsed_ok = True
            except Exception:
                pass

        # Validate substring constraint
        is_valid, validated_term = validate_term(raw_term, query)

        # Durations in Ollama are nanoseconds; convert to ms
        ollama_stats = {
            "load_duration": round(raw_response.get("load_duration", 0) / 1e6, 2),
            "prompt_eval_count": raw_response.get("prompt_eval_count", 0),
            "prompt_eval_duration": round(raw_response.get("prompt_eval_duration", 0) / 1e6, 2),
            "eval_count": raw_response.get("eval_count", 0),
            "eval_duration": round(raw_response.get("eval_duration", 0) / 1e6, 2),
        }

        meta = {
            "extractor": "llm",
            "model": self.model_name,
            "raw_term": raw_term,
            "is_null": (raw_term is None) and parsed_ok,
            "validation_ok": is_valid and (raw_term is None or validated_term is not None),
            "t_extract_ms": t_extract_ms,
            "ollama": ollama_stats,
        }

        final_term = validated_term if meta["validation_ok"] else None
        return ExtractResult(term=final_term, raw_output=content, meta=meta)


class PipelineExtractor:
    """Composite pipeline that coordinates Ollama and Rule extractor based on strategy."""

    def __init__(
        self,
        strategy: str = "llm+rules",
        ollama_extractor: Optional[OllamaExtractor] = None,
        rule_extractor: Optional[RuleExtractor] = None,
    ):
        self.strategy = strategy
        self.ollama_extractor = ollama_extractor
        self.rule_extractor = rule_extractor or RuleExtractor()

    def extract(self, query: str) -> ExtractResult:
        if self.strategy == "rules":
            res = self.rule_extractor.extract(query)
            res.meta["extractor_used"] = "rules"
            return res

        if self.strategy == "llm":
            if not self.ollama_extractor:
                raise ValueError("OllamaExtractor required for strategy='llm'")
            res = self.ollama_extractor.extract(query)
            res.meta["extractor_used"] = "llm"
            return res

        # Strategy 'llm+rules'
        if not self.ollama_extractor:
            res = self.rule_extractor.extract(query)
            res.meta["extractor_used"] = "rules"
            return res

        llm_res = self.ollama_extractor.extract(query)
        # If term extracted and valid, return it
        if llm_res.term is not None and llm_res.meta.get("validation_ok", False):
            llm_res.meta["extractor_used"] = "llm"
            return llm_res

        # Fallback to RuleExtractor on null or validation failure
        rule_res = self.rule_extractor.extract(query)
        rule_res.meta["extractor_used"] = "rules"
        rule_res.meta["fallback_from_llm"] = True
        rule_res.meta["llm_raw_output"] = llm_res.raw_output
        rule_res.meta["llm_meta"] = llm_res.meta
        return rule_res


class OnnxNERExtractor:
    """Lightweight on-device token classification extractor using quantized INT8 ONNX."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_dir: Optional[Path] = None,
    ):
        base_dir = Path(__file__).resolve().parent / "ner_trainer" / "model"
        self.model_path = model_path or (base_dir / "term_extractor_int8.onnx")
        self.model_dir = model_dir or base_dir
        self.label_list = ["O", "B-TERM", "I-TERM"]

        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.session = ort.InferenceSession(str(self.model_path))

    def extract(self, query: str) -> ExtractResult:
        import numpy as np

        t0 = time.perf_counter()
        if not query or not query.strip():
            t_extract_ms = round((time.perf_counter() - t0) * 1000, 2)
            meta = {
                "extractor": "onnx_ner",
                "model": "term_extractor_int8.onnx",
                "raw_term": None,
                "is_null": True,
                "validation_ok": True,
                "t_extract_ms": t_extract_ms,
            }
            return ExtractResult(term=None, raw_output=None, meta=meta)

        encoding = self.tokenizer(query, return_offsets_mapping=True)
        input_ids = np.array([encoding["input_ids"]], dtype=np.int64)
        attention_mask = np.array([encoding["attention_mask"]], dtype=np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        outputs = self.session.run(["logits"], ort_inputs)
        logits = outputs[0][0]
        pred_ids = np.argmax(logits, axis=-1)
        t_extract_ms = round((time.perf_counter() - t0) * 1000, 2)

        word_ids = encoding.word_ids(0)
        offsets = encoding["offset_mapping"]

        # Group wordpieces by word id. Take each word's label from its FIRST subtoken only.
        words = []
        seen_words = set()
        for idx, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            start_char, end_char = offsets[idx]
            if w_id not in seen_words:
                seen_words.add(w_id)
                label = self.label_list[pred_ids[idx]]
                words.append({
                    "word_id": w_id,
                    "label": label,
                    "start": start_char,
                    "end": end_char,
                })
            else:
                words[-1]["end"] = max(words[-1]["end"], end_char)
                words[-1]["start"] = min(words[-1]["start"], start_char)

        extracted_raw: Optional[str] = None
        if words:
            # The term is the first contiguous run of words labeled B-TERM then I-TERM
            # (if there is an I-TERM with no B-TERM before it, start there).
            start_idx = None
            for i, w in enumerate(words):
                if w["label"] in ("B-TERM", "I-TERM"):
                    start_idx = i
                    break

            if start_idx is not None:
                end_idx = start_idx
                while end_idx + 1 < len(words) and words[end_idx + 1]["label"] in ("B-TERM", "I-TERM"):
                    end_idx += 1

                term_words = words[start_idx : end_idx + 1]
                raw_slice = query[term_words[0]["start"] : term_words[-1]["end"]]
                punct_and_quotes = "'\"`“”‘’.,!?:;()[]{}*~_#"
                cleaned = raw_slice.strip().strip(punct_and_quotes).strip()
                if cleaned:
                    extracted_raw = cleaned

        is_valid, validated_term = validate_term(extracted_raw, query)

        meta = {
            "extractor": "onnx_ner",
            "model": "term_extractor_int8.onnx",
            "raw_term": extracted_raw,
            "is_null": extracted_raw is None,
            "validation_ok": is_valid,
            "t_extract_ms": t_extract_ms,
        }

        final_term = validated_term if is_valid else None
        return ExtractResult(term=final_term, raw_output=extracted_raw, meta=meta)


# Extractor registry for extensible dispatch
EXTRACTOR_REGISTRY: Dict[str, Any] = {
    "rules": RuleExtractor,
    "llm": OllamaExtractor,
    "llm+rules": PipelineExtractor,
    "onnx": OnnxNERExtractor,
}


def register_extractor(name: str, factory: Any) -> None:
    """Register a new extractor factory or class by name."""
    EXTRACTOR_REGISTRY[name] = factory


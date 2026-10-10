"""Main application entry point for the dictionary question-answering assistant.

Supports:
- CLI REPL (default)
- Minimal Gradio Web UI (--ui web)
- Configurable models and extractors (llm, rules, llm+rules)
- Read-only SQLite dictionary lookup and structured logging
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
import yaml

from db import DEFAULT_DB_PATH, find_closest_terms, fuzzy_lookup, get_db_connection, lookup
from extractors import (
    ExtractResult,
    OllamaExtractor,
    OnnxNERExtractor,
    PipelineExtractor,
    RuleExtractor,
)
from render import (
    NO_TERM_MESSAGE,
    render_closest_definition,
    render_definition,
    render_fuzzy_definition,
    render_no_term,
    render_not_found,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "models.yaml"
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "logs" / "queries.jsonl"


def load_model_config(
    model_name: Optional[str] = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Tuple[str, Dict[str, Any]]:
    """Load model configuration from models.yaml.

    Returns (selected_name, model_entry_dict).
    """
    resolved_path = Path(config_path).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    models_dict = data.get("models", {})
    if not models_dict:
        raise ValueError(f"No models configured in {resolved_path}")

    # Determine default model if not passed
    target = model_name or data.get("default_model") or next(iter(models_dict.keys()))

    # Find matching model by key, name, or ollama_tag
    selected_cfg = None
    selected_key = None

    if target in models_dict:
        selected_key = target
        selected_cfg = models_dict[target]
    else:
        for k, cfg in models_dict.items():
            if cfg.get("name") == target or cfg.get("ollama_tag") == target:
                selected_key = k
                selected_cfg = cfg
                break

    if selected_cfg is None:
        available = list(models_dict.keys())
        raise KeyError(
            f"Model '{target}' not found in {resolved_path}. Available models: {available}"
        )

    # Ensure name and ollama_tag are present
    cfg = dict(selected_cfg)
    if "name" not in cfg:
        cfg["name"] = selected_key
    if "ollama_tag" not in cfg:
        cfg["ollama_tag"] = cfg["name"]

    return selected_key, cfg


def preflight_ollama(ollama_tag: str, ollama_url: str = "http://localhost:11434") -> None:
    """Preflight check: confirm Ollama is reachable and model tag is pulled."""
    tags_url = f"{ollama_url.rstrip('/')}/api/tags"
    try:
        resp = requests.get(tags_url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, Exception):
        print(
            f"Error: Ollama server is unreachable at {ollama_url}.\n"
            f"Please start Ollama by launching the Ollama application or running:\n"
            f"    ollama serve\n",
            file=sys.stderr,
        )
        sys.exit(1)

    installed_tags = {m.get("name") for m in data.get("models", []) if m.get("name")}
    # Also check tags without :latest suffix or exact match
    tag_matches = (
        ollama_tag in installed_tags
        or f"{ollama_tag}:latest" in installed_tags
        or any(t.startswith(ollama_tag) for t in installed_tags)
    )

    if not tag_matches:
        print(
            f"Error: Required model '{ollama_tag}' is not installed in Ollama.\n"
            f"To pull this model, run:\n"
            f"    ollama pull {ollama_tag}\n",
            file=sys.stderr,
        )
        sys.exit(1)


def log_query(
    log_path: Path,
    record: Dict[str, Any],
) -> None:
    """Append one JSON line per query to the queries log file."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Warning: Failed to log query: {e}", file=sys.stderr)


class DictionaryApp:
    """Core application pipeline connecting extractor, DB lookup, renderer, and logger."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        extractor_type: str = "llm+rules",
        db_path: Path = DEFAULT_DB_PATH,
        config_path: Path = DEFAULT_CONFIG_PATH,
        log_path: Path = DEFAULT_LOG_PATH,
        ollama_url: str = "http://localhost:11434",
    ):
        self.db_path = Path(db_path).resolve()
        self.log_path = Path(log_path).resolve()
        self.config_path = Path(config_path).resolve()
        self.extractor_type = extractor_type
        self.ollama_url = ollama_url
        self.model_cfg: Optional[Dict[str, Any]] = None
        self.model_name = model_name

        # Load lookup config
        self.max_distance = 2
        self.require_unique_best = True
        if self.config_path.is_file():
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}
            lookup_cfg = cfg_data.get("lookup", {})
            self.max_distance = lookup_cfg.get("max_distance", 2)
            self.require_unique_best = lookup_cfg.get("require_unique_best", True)

        self.db_conn = get_db_connection(self.db_path)

        # Build extractors
        self.rule_extractor = RuleExtractor(db_path=self.db_path)
        self.ollama_extractor: Optional[OllamaExtractor] = None
        self.onnx_extractor: Optional[OnnxNERExtractor] = None

        if self.extractor_type == "onnx":
            self.model_name = "term_extractor_int8.onnx"
            self.onnx_extractor = OnnxNERExtractor()
            self.pipeline = self.onnx_extractor
        elif self.extractor_type in ("llm", "llm+rules"):
            selected_key, cfg = load_model_config(self.model_name, config_path)
            self.model_name = selected_key
            self.model_cfg = cfg

            # Preflight check
            preflight_ollama(cfg["ollama_tag"], self.ollama_url)

            self.ollama_extractor = OllamaExtractor(
                model_name=cfg["name"],
                ollama_tag=cfg["ollama_tag"],
                options=cfg.get("options"),
                think=cfg.get("think"),
                ollama_url=self.ollama_url,
            )
            # Warm up model to ensure accurate subsequent timings
            self.ollama_extractor.warmup()

            self.pipeline = PipelineExtractor(
                strategy=self.extractor_type,
                ollama_extractor=self.ollama_extractor,
                rule_extractor=self.rule_extractor,
            )
        else:
            self.pipeline = PipelineExtractor(
                strategy="rules",
                ollama_extractor=None,
                rule_extractor=self.rule_extractor,
            )

    def process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Process a user query end-to-end: extract -> lookup -> render -> log."""
        t_start = time.perf_counter()

        # 1. Extraction step
        t0_extract = time.perf_counter()
        extract_res: ExtractResult = self.pipeline.extract(query)
        t_extract_ms = round((time.perf_counter() - t0_extract) * 1000, 2)

        extracted_term = extract_res.term
        meta = extract_res.meta
        extractor_used = meta.get("extractor_used", meta.get("extractor", "rules"))
        validation_ok = meta.get("validation_ok", True)
        raw_output = extract_res.raw_output

        # Extract Ollama stats if available
        ollama_stats = meta.get("ollama")
        if not ollama_stats and "llm_meta" in meta:
            ollama_stats = meta["llm_meta"].get("ollama")
        if not ollama_stats:
            ollama_stats = {
                "load_duration": 0.0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0.0,
                "eval_count": 0,
                "eval_duration": 0.0,
            }

        # 2. Lookup step
        t0_lookup = time.perf_counter()
        rows = []
        resolution = "none"
        resolved_headwords: list[str] = []

        if extracted_term:
            resolution, resolved_headwords, rows = fuzzy_lookup(
                extracted_term,
                max_distance=self.max_distance,
                require_unique_best=self.require_unique_best,
                limit=5,
                conn=self.db_conn,
                db_path=self.db_path,
            )
        t_lookup_ms = round((time.perf_counter() - t0_lookup) * 1000, 2)

        # 3. Render step
        if not extracted_term:
            answer = render_no_term()
        elif resolution == "exact":
            answer = render_definition(rows)
        elif resolution == "fuzzy" and resolved_headwords:
            answer = render_fuzzy_definition(
                term=extracted_term,
                headword=resolved_headwords[0],
                rows=rows,
            )
        else:
            answer = render_not_found(extracted_term)

        t_total_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Prepare log record
        log_record = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": self.model_name if self.model_name else "rules",
            "query": query,
            "raw_model_output": raw_output,
            "extracted_term": extracted_term,
            "validation_ok": validation_ok,
            "extractor_used": extractor_used,
            "resolution": resolution,
            "resolved_headwords": resolved_headwords,
            "rows_found": len(rows),
            "t_extract_ms": t_extract_ms,
            "t_lookup_ms": t_lookup_ms,
            "t_total_ms": t_total_ms,
            "ollama": ollama_stats,
        }
        log_query(self.log_path, log_record)

        debug_info = {
            "extracted_term": extracted_term,
            "resolution": resolution,
            "resolved_headwords": resolved_headwords,
            "extractor_used": extractor_used,
            "validation_ok": validation_ok,
            "rows_found": len(rows),
            "t_extract_ms": t_extract_ms,
            "t_lookup_ms": t_lookup_ms,
            "t_total_ms": t_total_ms,
            "ollama": ollama_stats,
        }

        return answer, debug_info

    def run_repl(self) -> None:
        """Run interactive CLI REPL."""
        model_display = self.model_name if self.model_name else "rules"
        print(f"=== Dictionary Assistant (Model: {model_display}, Extractor: {self.extractor_type}) ===")
        print("Type your question (e.g., 'tell me what is black mean') or 'exit'/'quit' to leave.\n")

        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not query:
                continue
            if query.lower() in ("exit", "quit", ":q"):
                print("Goodbye!")
                break

            answer, _debug = self.process_query(query)
            print(answer)
            print()

    def run_web(self, host: str = "127.0.0.1", port: int = 7860) -> None:
        """Launch Gradio Web Interface with a clean UI and collapsible debug panel."""
        import gradio as gr

        model_display = self.model_name if self.model_name else "rules"

        def ui_handler(user_query: str):
            if not user_query or not user_query.strip():
                return "", "Please enter a question or word to look up."
            ans, debug = self.process_query(user_query.strip())
            debug_pretty = json.dumps(debug, indent=2)
            return ans, debug_pretty

        with gr.Blocks(title="Dictionary Assistant") as demo:
            gr.Markdown(
                f"# 📖 Offline Dictionary Assistant\n"
                f"Powered by **WordNet 3.0** & Local LLM (`{model_display}`) via Ollama."
            )
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="Ask a question",
                        placeholder="e.g. tell me what is black mean, what does kick the bucket mean...",
                        lines=2,
                    )
                    submit_btn = gr.Button("Look Up", variant="primary")
                with gr.Column():
                    output_text = gr.Textbox(
                        label="Definition",
                        lines=8,
                        interactive=False,
                    )
                    with gr.Accordion("Debug Details", open=False):
                        debug_output = gr.Code(
                            label="Extraction & Latency Info",
                            language="json",
                            interactive=False,
                        )

            submit_btn.click(
                fn=ui_handler,
                inputs=[query_input],
                outputs=[output_text, debug_output],
            )
            query_input.submit(
                fn=ui_handler,
                inputs=[query_input],
                outputs=[output_text, debug_output],
            )

        print(f"Launching Gradio Web UI at http://{host}:{port}...")
        demo.launch(server_name=host, server_port=port, inbrowser=False)


def main():
    parser = argparse.ArgumentParser(description="Dictionary Question-Answering App")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name from models.yaml (e.g. sam860/lfm2.5:350m or openbmb/minicpm5:latest)",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        choices=["llm", "rules", "llm+rules", "onnx"],
        default="llm+rules",
        help="Extraction strategy: llm, rules, llm+rules, or onnx (default: llm+rules)",
    )
    parser.add_argument(
        "--ui",
        type=str,
        choices=["cli", "web"],
        default="cli",
        help="Interface mode: cli or web (default: cli)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to dictionary SQLite DB (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to models.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Path to log file (default: {DEFAULT_LOG_PATH})",
    )

    args = parser.parse_args()

    app = DictionaryApp(
        model_name=args.model,
        extractor_type=args.extractor,
        db_path=args.db,
        config_path=args.config,
        log_path=args.log_file,
    )

    if args.ui == "web":
        app.run_web()
    else:
        app.run_repl()


if __name__ == "__main__":
    main()

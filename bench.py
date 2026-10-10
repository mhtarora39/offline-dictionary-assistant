"""Benchmark term extraction and dictionary resolution against gold test sets.

Usage:
  python bench.py --split dev --models sam860/lfm2.5:350m,openbmb/minicpm5:latest \
                  --extractors rules,llm,llm+rules --db dictionary_eval.db
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Ensure immediate unbuffered output on Windows
sys.stdout.reconfigure(line_buffering=True)

from app import load_model_config, preflight_ollama
from db import DEFAULT_DB_PATH, fuzzy_lookup
from extractors import (
    EXTRACTOR_REGISTRY,
    ExtractResult,
    OllamaExtractor,
    PipelineExtractor,
    RuleExtractor,
)


def percentile(data: List[float], p: float) -> float:
    """Calculate the p-th percentile of data (0 <= p <= 100)."""
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    d0 = data[int(f)] * (c - k)
    d1 = data[int(c)] * (k - f)
    return d0 + d1


def load_dataset(split: str, root_dir: Path) -> List[Dict[str, Any]]:
    """Load queries_<split>.jsonl and gold_<split>.jsonl joined by id."""
    queries_path = root_dir / "tests" / f"queries_{split}.jsonl"
    gold_path = root_dir / "tests" / f"gold_{split}.jsonl"

    if not queries_path.is_file():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    if not gold_path.is_file():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    queries_by_id: Dict[str, str] = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            queries_by_id[str(item["id"])] = item["query"]

    joined: List[Dict[str, Any]] = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gold_item = json.loads(line)
            gid = str(gold_item["id"])
            if gid not in queries_by_id:
                raise KeyError(f"Gold ID '{gid}' missing from queries file")
            joined.append({
                "id": gid,
                "query": queries_by_id[gid],
                "gold": gold_item,
            })

    return joined


def evaluate_run(
    label: str,
    extractor: Any,
    dataset: List[Dict[str, Any]],
    db_path: Path,
    max_distance: int = 2,
    require_unique_best: bool = True,
    cached_llm_results: Optional[Dict[str, Tuple[ExtractResult, float]]] = None,
    save_llm_cache: Optional[Dict[str, Tuple[ExtractResult, float]]] = None,
    rule_extractor: Optional[RuleExtractor] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate an extractor on the dataset, returning metrics dictionary and failure items."""
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    total = len(dataset)

    for idx, item in enumerate(dataset):
        if (idx + 1) % 25 == 0 or (idx + 1) == total:
            print(f"  [{idx + 1}/{total}] processing queries for {label}...", flush=True)

        qid = item["id"]
        query = item["query"]
        gold = item["gold"]

        # If cached LLM result is available for llm+rules
        if cached_llm_results is not None and query in cached_llm_results:
            cached_res, cached_lat = cached_llm_results[query]
            if cached_res.term is not None and cached_res.meta.get("validation_ok", False):
                res = cached_res
                t_extract_ms = cached_lat
            else:
                t0_rule = time.perf_counter()
                res = rule_extractor.extract(query) if rule_extractor else extractor.extract(query)
                t_extract_ms = cached_lat + (time.perf_counter() - t0_rule) * 1000.0
        else:
            t0 = time.perf_counter()
            res = extractor.extract(query)
            t_extract_ms = (time.perf_counter() - t0) * 1000.0
            if save_llm_cache is not None:
                save_llm_cache[query] = (res, t_extract_ms)

        term = res.term
        meta = res.meta or {}
        val_ok = meta.get("validation_ok", True)

        # Lookup in dictionary
        resolution, resolved_headwords, rows = fuzzy_lookup(
            term,
            max_distance=max_distance,
            require_unique_best=require_unique_best,
            db_path=db_path,
        )

        # Evaluate metrics
        expect = gold.get("expect")
        gold_surface = gold.get("gold_surface")
        gold_hws = [h.lower() for h in gold.get("gold_headwords", [])]

        # 1. extraction_exact: term == gold_surface, case-insensitive; for no_term, correct = None
        if expect == "no_term" or gold_surface is None:
            is_extraction_exact = term is None
        else:
            is_extraction_exact = term is not None and term.strip().lower() == gold_surface.strip().lower()

        # 2. invented_term: LLM output failing substring validation
        is_invented = not val_ok

        # 3. resolution_correct: for expect=found, resolved headword is in gold_headwords
        if expect == "found":
            is_resolution_correct = any(h.lower() in gold_hws for h in resolved_headwords)
        else:
            is_resolution_correct = None

        # 4. false_answer_rate: for expect=not_found or no_term, any definition was returned
        if expect in ("not_found", "no_term"):
            is_false_answer = len(rows) > 0 or len(resolved_headwords) > 0
        else:
            is_false_answer = None

        item_eval = {
            "id": qid,
            "type": gold.get("type", "unknown"),
            "expect": expect,
            "latency_ms": t_extract_ms,
            "extraction_exact": is_extraction_exact,
            "invented": is_invented,
            "resolution_correct": is_resolution_correct,
            "false_answer": is_false_answer,
            "extracted_term": term,
            "resolution": resolution,
            "resolved_headwords": resolved_headwords,
        }
        results.append(item_eval)

        # Check if failure
        has_failure = (
            not is_extraction_exact
            or (expect == "found" and not is_resolution_correct)
            or (expect in ("not_found", "no_term") and is_false_answer)
        )
        if has_failure:
            failures.append({
                "run": label,
                "id": qid,
                "query": query,
                "gold": {
                    "surface": gold_surface,
                    "headwords": gold.get("gold_headwords", []),
                    "expect": expect,
                    "type": gold.get("type"),
                },
                "predicted": {
                    "term": term,
                    "resolution": resolution,
                    "resolved_headwords": resolved_headwords,
                    "validation_ok": val_ok,
                },
            })

    # Aggregate metrics overall and per type
    types = sorted({r["type"] for r in results})
    groups = ["Overall"] + types

    metrics: Dict[str, Dict[str, Any]] = {}
    for g in groups:
        subset = results if g == "Overall" else [r for r in results if r["type"] == g]
        if not subset:
            continue

        n_total = len(subset)
        exact_acc = sum(1 for r in subset if r["extraction_exact"]) / n_total
        inv_rate = sum(1 for r in subset if r["invented"]) / n_total

        found_subset = [r for r in subset if r["expect"] == "found"]
        res_corr_rate = (
            (sum(1 for r in found_subset if r["resolution_correct"]) / len(found_subset))
            if found_subset
            else None
        )

        not_found_subset = [r for r in subset if r["expect"] in ("not_found", "no_term")]
        false_ans_rate = (
            (sum(1 for r in not_found_subset if r["false_answer"]) / len(not_found_subset))
            if not_found_subset
            else None
        )

        lats = sorted(r["latency_ms"] for r in subset)
        p50 = percentile(lats, 50.0)
        p95 = percentile(lats, 95.0)

        metrics[g] = {
            "count": n_total,
            "extraction_exact": exact_acc,
            "invented_term_rate": inv_rate,
            "resolution_correct": res_corr_rate,
            "false_answer_rate": false_ans_rate,
            "p50_ms": p50,
            "p95_ms": p95,
        }

    return {"label": label, "groups": metrics}, failures


def print_side_by_side_table(
    metric_title: str,
    metric_key: str,
    runs_metrics: List[Dict[str, Any]],
    is_percentage: bool = True,
    is_latency: bool = False,
) -> None:
    """Print side-by-side table for a single metric across runs."""
    labels = [r["label"] for r in runs_metrics]
    all_groups: List[str] = ["Overall"]
    for r in runs_metrics:
        for g in r["groups"].keys():
            if g not in all_groups:
                all_groups.append(g)

    # Determine column widths
    max_label_w = max(18, max(len(l) for l in labels))
    col_w = max(14, max_label_w)
    grp_w = max(18, max(len(g) for g in all_groups))

    sep = "-" * (grp_w + 3 + (col_w + 3) * len(labels))
    print("\n" + sep, flush=True)
    print(f" METRIC: {metric_title}", flush=True)
    print(sep, flush=True)

    header_cols = [f"{'type':<{grp_w}}"] + [f"{l:>{col_w}}" for l in labels]
    print(" | ".join(header_cols), flush=True)
    print(sep, flush=True)

    for g in all_groups:
        row_cols = [f"{g:<{grp_w}}"]
        for r in runs_metrics:
            g_data = r["groups"].get(g)
            if not g_data:
                row_cols.append(f"{'N/A':>{col_w}}")
                continue
            val = g_data.get(metric_key)
            if val is None:
                row_cols.append(f"{'N/A':>{col_w}}")
            elif is_percentage:
                row_cols.append(f"{val * 100:>13.1f}%")
            elif is_latency:
                row_cols.append(f"{val:>11.2f} ms")
            else:
                row_cols.append(f"{val:>{col_w}}")
        print(" | ".join(row_cols), flush=True)

    print(sep, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Term extraction and lookup benchmark")
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "test"],
        default="dev",
        help="Evaluation split (dev or test)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="sam860/lfm2.5:350m,openbmb/minicpm5:latest",
        help="Comma-separated model names from models.yaml",
    )
    parser.add_argument(
        "--extractors",
        type=str,
        default="rules,llm,llm+rules",
        help="Comma-separated extractor strategies: rules,llm,llm+rules",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("dictionary_eval.db"),
        help="Database file to query against (default: dictionary_eval.db)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("models.yaml"),
        help="Path to models.yaml",
    )
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="Required confirmation flag to run on the test split",
    )

    args = parser.parse_args()

    # Split safety check
    if args.split == "test" and not args.confirm_test:
        print(
            "WARNING: Test set must be run only once, after all tuning on dev.\n"
            "Pass --confirm-test to proceed with running the test split.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    project_root = Path(__file__).resolve().parent
    db_path = Path(args.db).resolve()
    if not db_path.is_file():
        print(f"Error: Database file not found: {db_path}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Load lookup config from models.yaml
    max_distance = 2
    require_unique_best = True
    config_path = Path(args.config).resolve()
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}
        lookup_cfg = cfg_data.get("lookup", {})
        max_distance = lookup_cfg.get("max_distance", 2)
        require_unique_best = lookup_cfg.get("require_unique_best", True)

    # Load dataset
    print(f"Loading {args.split} dataset...", flush=True)
    dataset = load_dataset(args.split, project_root)
    print(f"Loaded {len(dataset)} items from {args.split} split.", flush=True)

    # Parse requested models and extractors
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    extractor_names = [e.strip() for e in args.extractors.split(",") if e.strip()]

    # Build evaluation schedule:
    runs_to_execute = []
    if "rules" in extractor_names:
        runs_to_execute.append({"label": "rules", "strategy": "rules", "model": None})
    if "onnx" in extractor_names:
        runs_to_execute.append({"label": "onnx", "strategy": "onnx", "model": None})

    for m_name in model_names:
        if "llm" in extractor_names:
            runs_to_execute.append({"label": f"{m_name} (llm)", "strategy": "llm", "model": m_name})
        if "llm+rules" in extractor_names:
            runs_to_execute.append({"label": f"{m_name} (llm+rules)", "strategy": "llm+rules", "model": m_name})

    rule_extractor = RuleExtractor(db_path=db_path)
    all_runs_metrics: List[Dict[str, Any]] = []
    all_failures: List[Dict[str, Any]] = []

    # Model LLM cache to avoid duplicate Ollama calls between 'llm' and 'llm+rules'
    llm_caches: Dict[str, Dict[str, Tuple[ExtractResult, float]]] = {}

    for r_spec in runs_to_execute:
        label = r_spec["label"]
        strat = r_spec["strategy"]
        m_name = r_spec["model"]

        print(f"\n==================================================", flush=True)
        print(f"Running benchmark for: {label}", flush=True)
        print(f"==================================================", flush=True)

        cached_results = None
        save_cache = None

        if strat == "rules":
            ext = rule_extractor
        elif strat == "onnx":
            onnx_cls = EXTRACTOR_REGISTRY.get("onnx")
            ext = onnx_cls()
        elif strat == "llm":
            _, cfg = load_model_config(m_name, config_path=config_path)
            preflight_ollama(cfg["ollama_tag"])

            ollama_ext = OllamaExtractor(
                model_name=cfg["name"],
                ollama_tag=cfg["ollama_tag"],
                options=cfg.get("options"),
                think=cfg.get("think"),
            )
            print(f"Warming up {cfg['name']}...", flush=True)
            ollama_ext.warmup()
            ext = ollama_ext

            llm_caches[m_name] = {}
            save_cache = llm_caches[m_name]
        elif strat == "llm+rules":
            if m_name in llm_caches:
                cached_results = llm_caches[m_name]
                ext = None
            else:
                _, cfg = load_model_config(m_name, config_path=config_path)
                preflight_ollama(cfg["ollama_tag"])

                ollama_ext = OllamaExtractor(
                    model_name=cfg["name"],
                    ollama_tag=cfg["ollama_tag"],
                    options=cfg.get("options"),
                    think=cfg.get("think"),
                )
                print(f"Warming up {cfg['name']}...", flush=True)
                ollama_ext.warmup()

                ext = PipelineExtractor(
                    strategy=strat,
                    ollama_extractor=ollama_ext,
                    rule_extractor=rule_extractor,
                )

        print(f"Evaluating {len(dataset)} queries for {label}...", flush=True)
        run_metric, run_failures = evaluate_run(
            label=label,
            extractor=ext,
            dataset=dataset,
            db_path=db_path,
            max_distance=max_distance,
            require_unique_best=require_unique_best,
            cached_llm_results=cached_results,
            save_llm_cache=save_cache,
            rule_extractor=rule_extractor,
        )
        all_runs_metrics.append(run_metric)
        all_failures.extend(run_failures)

    # Print side-by-side comparison tables per metric
    print("\n" + "=" * 80, flush=True)
    print("BENCHMARK COMPARISON TABLES", flush=True)
    print("=" * 80, flush=True)

    print_side_by_side_table("Extraction Exact-Match Accuracy", "extraction_exact", all_runs_metrics, is_percentage=True)
    print_side_by_side_table("Invented-Term Rate", "invented_term_rate", all_runs_metrics, is_percentage=True)
    print_side_by_side_table("Resolution Correct Rate (expect=found)", "resolution_correct", all_runs_metrics, is_percentage=True)
    print_side_by_side_table("False-Answer Rate (expect=not_found/no_term)", "false_answer_rate", all_runs_metrics, is_percentage=True)
    print_side_by_side_table("Extraction Latency p50", "p50_ms", all_runs_metrics, is_percentage=False, is_latency=True)
    print_side_by_side_table("Extraction Latency p95", "p95_ms", all_runs_metrics, is_percentage=False, is_latency=True)

    # Save output results
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. JSON output
    json_path = results_dir / f"{args.split}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "timestamp": timestamp,
                "db": str(db_path),
                "metrics": all_runs_metrics,
            },
            f,
            indent=2,
        )
    print(f"\nSaved benchmark metrics to {json_path}", flush=True)

    # 2. CSV output
    csv_path = results_dir / f"{args.split}_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Group", "Count", "Extraction_Exact", "Invented_Rate", "Resolution_Correct", "False_Answer_Rate", "p50_ms", "p95_ms"])
        for rm in all_runs_metrics:
            r_label = rm["label"]
            for grp, gm in rm["groups"].items():
                writer.writerow([
                    r_label,
                    grp,
                    gm["count"],
                    f"{gm['extraction_exact']:.4f}",
                    f"{gm['invented_term_rate']:.4f}",
                    f"{gm['resolution_correct']:.4f}" if gm["resolution_correct"] is not None else "",
                    f"{gm['false_answer_rate']:.4f}" if gm["false_answer_rate"] is not None else "",
                    f"{gm['p50_ms']:.2f}",
                    f"{gm['p95_ms']:.2f}",
                ])
    print(f"Saved benchmark CSV to {csv_path}", flush=True)

    # 3. Failures JSONL
    failures_path = results_dir / f"{args.split}_failures.jsonl"
    with open(failures_path, "w", encoding="utf-8") as f:
        for fail in all_failures:
            f.write(json.dumps(fail, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_failures)} failure records to {failures_path}", flush=True)


if __name__ == "__main__":
    main()

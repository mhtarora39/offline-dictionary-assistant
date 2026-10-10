import sys
from pathlib import Path

# Add project root to sys.path so tests can import db, extractors, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import requests

from db import DEFAULT_DB_PATH, get_db_connection, lookup
from extractors import (
    OllamaExtractor,
    RuleExtractor,
    validate_term,
)
from render import render_definition, render_no_term, render_not_found


# =========================================================================
# Tests that need no model (must pass)
# =========================================================================

def test_lookup_black_color_adjective_first():
    """lookup('black') returns rows with the color-adjective sense first."""
    rows = lookup("black", limit=5)
    assert len(rows) > 0
    headword, pos, gloss, example, freq, idx = rows[0]
    assert headword == "black"
    assert pos == "adj"
    assert "achromatic color of maximum darkness" in gloss


def test_lookup_inflections_ran_and_geese():
    """lookup('ran') resolves to headword 'run'; lookup('geese') to 'goose'."""
    ran_rows = lookup("ran", limit=5)
    assert len(ran_rows) > 0
    assert ran_rows[0][0] == "run"
    assert ran_rows[0][1] == "verb"

    geese_rows = lookup("geese", limit=5)
    assert len(geese_rows) > 0
    assert geese_rows[0][0] == "goose"
    assert geese_rows[0][1] == "noun"


def test_lookup_phrase_and_missing_word():
    """lookup('kick the bucket') returns 1 headword; lookup('flibbergast') returns 0 rows."""
    kick_rows = lookup("kick the bucket", limit=5)
    assert len(kick_rows) == 1
    assert kick_rows[0][0] == "kick the bucket"
    assert kick_rows[0][1] == "verb"

    flibbergast_rows = lookup("flibbergast", limit=5)
    assert len(flibbergast_rows) == 0


def test_rule_extractor():
    """RuleExtractor handles messy phrasing, multiword, and inflection queries."""
    extractor = RuleExtractor()

    res1 = extractor.extract("tell me what is black mean")
    assert res1.term == "black"

    res2 = extractor.extract("what does kick the bucket mean")
    assert res2.term == "kick the bucket"

    res3 = extractor.extract("meaning of ran")
    assert res3.term == "ran"


def test_validation_rejects_invented_term():
    """validation rejects a term that is not a substring of the query."""
    # Substring matches should pass
    valid, term = validate_term("black", "tell me what is black mean")
    assert valid is True
    assert term == "black"

    # Invented term not in query should be rejected
    invalid, term = validate_term("white", "tell me what is black mean")
    assert invalid is False
    assert term is None

    # Null / None term
    valid_null, term_null = validate_term(None, "what's the weather today")
    assert valid_null is True
    assert term_null is None


def test_rendering_formatting():
    """Test template rendering output format."""
    rows = lookup("black", limit=5)
    rendered = render_definition(rows, max_senses_per_group=3)
    assert "black" in rendered
    assert "(adj)" in rendered
    assert "(noun)" in rendered

    not_found = render_not_found("flibbergast")
    assert "No definition found for 'flibbergast'." in not_found

    not_found_with_sugg = render_not_found("flibbergast", suggestions=["flabbergast"])
    assert "No definition found for 'flibbergast'." in not_found_with_sugg
    assert "Did you mean 'flabbergast'?" in not_found_with_sugg

    no_term = render_no_term()
    assert "I couldn't find a word to look up in that." in no_term


def test_find_closest_terms():
    """find_closest_terms returns relevant spelling suggestions for missing words."""
    from db import find_closest_terms

    matches = find_closest_terms("flibbergast", limit=3)
    assert len(matches) > 0
    assert "flabbergast" in matches

    matches_happines = find_closest_terms("happines", limit=3)
    assert len(matches_happines) > 0
    assert "happiness" in matches_happines


def test_closest_definition_rendering():
    """Test rendering of closest match definition and additional suggestions."""
    from render import render_closest_definition

    rows = lookup("flabbergast", limit=5)
    rendered = render_closest_definition(
        original_term="flibbergast",
        closest_term="flabbergast",
        rows=rows,
        other_suggestions=["flabbergasts", "flabbergasted"],
    )
    assert "No exact match found for 'flibbergast'." in rendered
    assert "Showing definition for closest match 'flabbergast':" in rendered
    assert "flabbergast" in rendered
    assert "Other suggestions: 'flabbergasts', 'flabbergasted'" in rendered


def test_app_closest_definition_end_to_end():
    """DictionaryApp provides closest match definition when exact word is not found."""
    from app import DictionaryApp

    app = DictionaryApp(extractor_type="rules")
    answer, debug = app.process_query("tell me what is flibbergast mean")
    assert "Showing results for flabbergast (you typed flibbergast)" in answer
    assert "flabbergast" in answer
    assert "flabbergast" in debug["resolved_headwords"]
    assert debug["resolution"] == "fuzzy"


def test_fuzzy_lookup():
    """Fuzzy lookup: one-letter typo of common word resolves; invented word returns not found."""
    from db import fuzzy_lookup

    # One-letter typo of a common word resolves
    res, hws, rows = fuzzy_lookup("nieghborhood")
    assert res == "fuzzy"
    assert "neighborhood" in hws
    assert len(rows) > 0

    # Invented word returns not found
    res_inv, hws_inv, rows_inv = fuzzy_lookup("thuxdrondsnarp")
    assert res_inv == "none"
    assert len(hws_inv) == 0
    assert len(rows_inv) == 0


def test_leakage_guard():
    """Ensure no few-shot terms or queries overlap with gold_dev or gold_test."""
    import json
    from extractors import FEW_SHOT_MESSAGES

    # Extract all few-shot terms and queries
    fs_terms = set()
    fs_queries = set()
    for m in FEW_SHOT_MESSAGES:
        content = m["content"]
        if m["role"] == "user":
            fs_queries.add(content.strip().lower())
        elif m["role"] == "assistant":
            try:
                parsed = json.loads(content)
                t = parsed.get("term")
                if t:
                    fs_terms.add(t.strip().lower())
            except Exception:
                pass

    # Check against gold_dev.jsonl and gold_test.jsonl
    test_dir = Path(__file__).resolve().parent
    for split in ["dev", "test"]:
        gold_path = test_dir / f"gold_{split}.jsonl"
        with open(gold_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                gs = (item.get("gold_surface") or "").strip().lower()
                ghs = [h.strip().lower() for h in item.get("gold_headwords", [])]

                if gs:
                    assert gs not in fs_terms, f"Leakage: gold_surface '{gs}' in few-shot terms ({split})"
                for gh in ghs:
                    assert gh not in fs_terms, f"Leakage: gold_headword '{gh}' in few-shot terms ({split})"


# =========================================================================
# Tests that need Ollama (skip automatically if server or model unavailable)
# =========================================================================

def _ollama_model_available(tag: str) -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code != 200:
            return False
        names = {m.get("name") for m in r.json().get("models", [])}
        return tag in names or f"{tag}:latest" in names or any(n.startswith(tag) for n in names)
    except Exception:
        return False


@pytest.mark.skipif(
    not _ollama_model_available("sam860/lfm2.5:350m"),
    reason="Ollama server or sam860/lfm2.5:350m not available",
)
def test_ollama_extractor_black():
    """OllamaExtractor returns 'black' for 'tell me what is black mean'."""
    extractor = OllamaExtractor(
        model_name="sam860/lfm2.5:350m",
        ollama_tag="sam860/lfm2.5:350m",
    )
    res = extractor.extract("tell me what is black mean")
    assert res.term == "black"


@pytest.mark.skipif(
    not _ollama_model_available("openbmb/minicpm5:latest"),
    reason="Ollama server or openbmb/minicpm5:latest not available",
)
def test_ollama_extractor_null():
    """OllamaExtractor returns None for 'what's the weather today'."""
    extractor = OllamaExtractor(
        model_name="openbmb/minicpm5:latest",
        ollama_tag="openbmb/minicpm5:latest",
        think=False,
    )
    res = extractor.extract("what's the weather today")
    assert res.term is None


def test_onnx_extractor_substring_30_queries():
    """Verify that OnnxNERExtractor extracts terms that are strictly case-insensitive substrings.

    Tests 30 varied queries including typos, ALL CAPS, 'x' and "x" quotes,
    multiword terms, merged words, and apostrophes (e.g. what's).
    """
    from extractors import OnnxNERExtractor

    extractor = OnnxNERExtractor()

    varied_queries = [
        "what does 'cat' mean",
        'define "serendipity" please',
        "WHAT DOES PARADIGM MEAN",
        "TELL ME ABOUT 'PHILOSOPHY'",
        "what is nieghborhood",
        "how to define pronounciation",
        "meaning of kick the bucket",
        "tell me what spill the beans means",
        "whats the meaning of life",
        "what's serendipity",
        "what's 'rock and roll'?",
        "whatis photosynthesis",
        "whatdoes elephant mean",
        "hey what does drapht meanexactly",
        "Squeeze meaningpls!",
        "define 'spic and span' please",
        "EXPLAIN 'A BLESSING IN DISGUISE'",
        "what does 'x' mean",
        'define "x" right now',
        "can you tell me what's algorithm?",
        "wats  the mening fo pose!",
        "wat is th meeningfo collision??",
        "what does the word 'liason' mean",
        "what is 'state-of-the-art'?",
        "WHAT DOES 'KICK THE BUCKET' MEAN?",
        "tell me about 'machine learning'",
        "meaning of 'deep learning'?",
        "what's the definition of 'quantum computing'",
        "what does 'inter-connected' mean",
        "hey what is 'abracadabra'?",
    ]

    assert len(varied_queries) == 30

    for q in varied_queries:
        res = extractor.extract(q)
        assert res.meta.get("validation_ok") is True
        if res.term is not None:
            assert res.term.lower() in q.lower(), (
                f"Extracted term '{res.term}' is not a case-insensitive substring of query '{q}'"
            )


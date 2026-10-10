"""Database operations for dictionary lookup using WordNet 3.0 SQLite database."""

from __future__ import annotations

import sqlite3
import unicodedata
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "dictionary.db"

REFERENCE_LOOKUP_QUERY = """
SELECT e.headword, e.pos, s.gloss, s.example, s.freq, s.idx
FROM forms f JOIN entries e ON e.id=f.entry_id JOIN senses s ON s.entry_id=e.id
WHERE f.form=? ORDER BY s.freq DESC, s.idx ASC LIMIT ?
"""


def normalize_term(term: str) -> str:
    """Normalize term for dictionary lookup: lowercase, NFC, strip quotes/punctuation.

    Preserves internal spaces and hyphens for multi-word phrases.
    """
    if not term:
        return ""
    # Strip whitespace, quotes, and punctuation from outer edges
    stripped = term.strip().strip("'\"`“”‘’.,!?:;()[]{}*~_#")
    # Normalize unicode to NFC and lowercase
    normalized = unicodedata.normalize("NFC", stripped).lower()
    # Replace multiple internal spaces or underscores with a single space
    return " ".join(normalized.replace("_", " ").split())


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a read-only SQLite database connection using URI mode=ro."""
    path = db_path if db_path is not None else DEFAULT_DB_PATH
    resolved_path = Path(path).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Dictionary database not found at {resolved_path}")

    # Use URI filename with mode=ro to enforce read-only access
    uri = f"{resolved_path.as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def lookup(
    term: str,
    limit: int = 5,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> List[Tuple[str, str, str, Optional[str], int, int]]:
    """Look up a normalized term in the database and return ranked senses.

    Returns a list of tuples: (headword, pos, gloss, example, freq, idx).
    """
    normalized = normalize_term(term)
    if not normalized:
        return []

    should_close = False
    if conn is None:
        conn = get_db_connection(db_path)
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(REFERENCE_LOOKUP_QUERY, (normalized, limit))
        rows = [
            (
                row["headword"],
                row["pos"],
                row["gloss"],
                row["example"],
                row["freq"],
                row["idx"],
            )
            for row in cursor.fetchall()
        ]
        return rows
    finally:
        if should_close:
            conn.close()


# Cached forms by length and form -> headwords map per database path
_FORMS_CACHE: dict[str, tuple[dict[int, list[str]], dict[str, list[str]]]] = {}


def _get_forms_cache(
    conn: sqlite3.Connection, db_path_str: str
) -> tuple[dict[int, list[str]], dict[str, list[str]]]:
    """Retrieve or build in-memory cache of (forms_by_len, form_to_headwords)."""
    global _FORMS_CACHE
    if db_path_str in _FORMS_CACHE:
        return _FORMS_CACHE[db_path_str]

    cursor = conn.cursor()
    cursor.execute("SELECT f.form, e.headword FROM forms f JOIN entries e ON e.id = f.entry_id")
    rows = cursor.fetchall()

    from collections import defaultdict

    by_len: dict[int, list[str]] = defaultdict(list)
    form_to_hw: dict[str, list[str]] = {}

    for row in rows:
        f = row[0]
        hw = row[1]
        if f not in form_to_hw:
            form_to_hw[f] = []
            by_len[len(f)].append(f)
        if hw not in form_to_hw[f]:
            form_to_hw[f].append(hw)

    _FORMS_CACHE[db_path_str] = (by_len, form_to_hw)
    return _FORMS_CACHE[db_path_str]


def fuzzy_lookup(
    term: str,
    max_distance: int = 2,
    require_unique_best: bool = True,
    limit: int = 5,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> Tuple[str, List[str], List[Tuple[str, str, str, Optional[str], int, int]]]:
    """Look up term with exact match on forms, falling back to rapidfuzz over all forms.

    Returns (resolution, resolved_headwords, rows):
    - resolution: "exact" | "fuzzy" | "none"
    - resolved_headwords: list of matching headword names (e.g. ["pronunciation"])
    - rows: sense rows ranked by freq DESC, idx ASC (empty if resolution == "none")
    """
    from rapidfuzz.distance import Levenshtein

    normalized = normalize_term(term)
    if not normalized:
        return "none", [], []

    should_close = False
    if conn is None:
        conn = get_db_connection(db_path)
        should_close = True

    try:
        # 1. Exact lookup on forms
        exact_rows = lookup(normalized, limit=limit, conn=conn)
        if exact_rows:
            headwords = list(dict.fromkeys(r[0] for r in exact_rows))
            return "exact", headwords, exact_rows

        # 2. Fuzzy fallback with rapidfuzz over all forms
        resolved_db_path = str(Path(db_path or DEFAULT_DB_PATH).resolve())
        by_len, form_to_hw = _get_forms_cache(conn, resolved_db_path)

        L = len(normalized)
        from collections import defaultdict

        by_dist: dict[int, list[str]] = defaultdict(list)
        min_len = max(1, L - max_distance)
        max_len = L + max_distance

        for l in range(min_len, max_len + 1):
            for candidate_form in by_len.get(l, []):
                d = Levenshtein.distance(normalized, candidate_form, score_cutoff=max_distance)
                if d <= max_distance:
                    by_dist[d].append(candidate_form)

        if not by_dist:
            return "none", [], []

        min_dist = min(by_dist.keys())
        cand_forms_at_min = by_dist[min_dist]

        # Map candidate forms at min_dist to unique headwords
        candidate_headwords = list(
            dict.fromkeys(hw for cf in cand_forms_at_min for hw in form_to_hw.get(cf, []))
        )

        # Uniqueness rule: accept only if edit distance <= max_distance AND no other headword at the same distance
        if require_unique_best and len(candidate_headwords) != 1:
            return "none", [], []

        if not candidate_headwords:
            return "none", [], []

        best_headword = candidate_headwords[0]
        fuzzy_rows = lookup(best_headword, limit=limit, conn=conn)
        return "fuzzy", [best_headword], fuzzy_rows

    finally:
        if should_close:
            conn.close()


def is_known_form(
    form: str,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """Check if a normalized form exists in the forms table."""
    normalized = normalize_term(form)
    if not normalized:
        return False

    should_close = False
    if conn is None:
        conn = get_db_connection(db_path)
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM forms WHERE form = ? LIMIT 1", (normalized,))
        return cursor.fetchone() is not None
    finally:
        if should_close:
            conn.close()


def get_known_forms(
    candidates: Sequence[str],
    conn: sqlite3.Connection,
) -> Set[str]:
    """Given a sequence of candidates, return the set of forms present in the DB."""
    if not candidates:
        return set()

    normalized_candidates = {normalize_term(c) for c in candidates if normalize_term(c)}
    if not normalized_candidates:
        return set()

    # Query in batches if large
    cand_list = list(normalized_candidates)
    batch_size = 500
    known = set()
    cursor = conn.cursor()

    for i in range(0, len(cand_list), batch_size):
        batch = cand_list[i : i + batch_size]
        placeholders = ",".join("?" for _ in batch)
        query = f"SELECT form FROM forms WHERE form IN ({placeholders})"
        cursor.execute(query, batch)
        for row in cursor.fetchall():
            known.add(row["form"])

    return known


def find_closest_terms(
    term: str,
    limit: int = 3,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> List[str]:
    """Find the closest existing forms/words in the dictionary using fuzzy matching.

    Supports:
    1. Multi-word phrase correction (e.g., 'kick the buket' -> 'kick the bucket')
    2. Corpus frequency-weighted ranking so common words are preferred over obscure lemmas
    3. Cross-prefix fallback for misspelled initial letters (e.g. 'psycology' -> 'psychology')
    """
    import difflib
    import math

    normalized = normalize_term(term)
    if not normalized or len(normalized) < 2:
        return []

    should_close = False
    if conn is None:
        conn = get_db_connection(db_path)
        should_close = True

    try:
        cursor = conn.cursor()

        # 1. Multi-word phrase support
        if " " in normalized:
            words = [w for w in normalized.split() if len(w) >= 2]

            # 1a. Check if any constituent content word is an exact match in the DB with senses
            exact_matches = []
            for w in words:
                cursor.execute(
                    """
                    SELECT f.form, MAX(s.freq) as max_freq
                    FROM forms f
                    JOIN entries e ON e.id = f.entry_id
                    JOIN senses s ON s.entry_id = e.id
                    WHERE f.form = ?
                    GROUP BY f.form
                    """,
                    (w,),
                )
                row = cursor.fetchone()
                if row and row[1] is not None and row[1] > 0:
                    exact_matches.append((row[0], row[1]))
            if exact_matches:
                exact_matches.sort(key=lambda x: x[1], reverse=True)
                return [m[0] for m in exact_matches][:limit]

            # 1b. Check multi-word phrase prefix
            first_word = words[0] if words else ""
            cursor.execute(
                """
                SELECT DISTINCT form FROM forms
                WHERE form LIKE ? AND length(form) BETWEEN ? AND ?
                LIMIT 1500
                """,
                (f"{first_word}%", max(1, len(normalized) - 4), len(normalized) + 4),
            )
            candidates = [row[0] for row in cursor.fetchall()]
            matches = difflib.get_close_matches(normalized, candidates, n=limit, cutoff=0.6)
            if matches:
                return matches

            # 1c. If multi-word phrase didn't match directly, try finding closest matches for each content word
            for w in sorted(words, key=len, reverse=True):
                if len(w) >= 3:
                    sub_matches = find_closest_terms(w, limit=limit, conn=conn)
                    if sub_matches:
                        return sub_matches

        # 2. Single-word / general candidate search with frequency weighting
        first_char = normalized[0]
        min_len = max(1, len(normalized) - 3)
        max_len = len(normalized) + 3

        query_primary = """
            SELECT f.form, MAX(s.freq) as max_freq
            FROM forms f
            JOIN entries e ON e.id = f.entry_id
            JOIN senses s ON s.entry_id = e.id
            WHERE f.form >= ? AND f.form < ?
              AND length(f.form) BETWEEN ? AND ?
            GROUP BY f.form
        """
        cursor.execute(query_primary, (first_char, chr(ord(first_char) + 1), min_len, max_len))
        candidates = cursor.fetchall()
        freq_map = dict(candidates)
        form_names = list(freq_map.keys())
        matches = difflib.get_close_matches(normalized, form_names, n=min(15, limit * 4), cutoff=0.6)

        # 3. Fallback: if no match or low-similarity match sharing initial letter (e.g. 'katerpillar' -> 'caterpillar', 'psycology' -> 'psychology')
        top_sim = difflib.SequenceMatcher(None, normalized, matches[0]).ratio() if matches else 0.0
        if not matches or top_sim < 0.75:
            query_fallback = """
                SELECT f.form, MAX(s.freq) as max_freq
                FROM forms f
                JOIN entries e ON e.id = f.entry_id
                JOIN senses s ON s.entry_id = e.id
                WHERE length(f.form) BETWEEN ? AND ?
                GROUP BY f.form
                LIMIT 25000
            """
            cursor.execute(query_fallback, (min_len, max_len))
            fallback_candidates = cursor.fetchall()
            fallback_map = dict(fallback_candidates)
            freq_map.update(fallback_map)
            form_names = list(freq_map.keys())
            matches = difflib.get_close_matches(normalized, form_names, n=min(15, limit * 4), cutoff=0.6)

        if not matches:
            return []

        # Ranking score: string similarity (dominant) + frequency bonus (tie-breaker)
        def score(cand: str) -> float:
            sim = difflib.SequenceMatcher(None, normalized, cand).ratio()
            freq = freq_map.get(cand, 0)
            return (sim * 10.0) + (math.log1p(freq) * 0.3)

        return sorted(matches, key=score, reverse=True)[:limit]
    finally:
        if should_close:
            conn.close()


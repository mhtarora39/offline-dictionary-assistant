#!/usr/bin/env python3
"""
Build dictionary.db (SQLite) from WordNet 3.0 via NLTK.

Tables
  entries(id, headword, pos)                one row per (headword, part of speech)
  senses(id, entry_id, idx, gloss, example, freq)
                                            idx  = WordNet sense order (0-based)
                                            freq = tagged-corpus count for this lemma sense
  forms(form, entry_id)                     every surface form that maps to an entry
                                            (lemma, irregulars, regular inflections)
  meta(key, value)

Usage
  pip install nltk
  python build_dictionary_db.py --out dictionary.db
  # eval variant: hide 5% of headwords to create real NOT_FOUND queries
  python build_dictionary_db.py --out dictionary_eval.db --holdout 0.05 --seed 42
"""
import argparse
import os
import random
import sqlite3
import time
import unicodedata

import nltk
from nltk.corpus import wordnet as wn

SCHEMA_VERSION = "1"
POS_MAP = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}  # 's' (satellite) is folded into 'a'


def norm(text: str) -> str:
    """Lowercase, NFC-normalise, underscores -> spaces."""
    return unicodedata.normalize("NFC", text.replace("_", " ").lower()).strip()


def ensure_wordnet():
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        wn.ensure_loaded()


# ---------------------------------------------------------------- inflections
def regular_candidates(lemma: str, pos: str):
    """Rule-based surface forms. Filtered later by a WordNet round trip."""
    c = set()
    last = lemma.split(" ")[-1] if " " in lemma else lemma  # inflect last word of phrases
    head = lemma[: len(lemma) - len(last)]

    def add(suffix_form):
        c.add(head + suffix_form)

    if pos == "n":
        if last.endswith(("s", "x", "z", "ch", "sh")):
            add(last + "es")
        elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
            add(last[:-1] + "ies")
        else:
            add(last + "s")
        if last.endswith("man"):
            add(last[:-3] + "men")
    elif pos == "v":
        # 3rd person
        if last.endswith(("s", "x", "z", "ch", "sh", "o")):
            add(last + "es")
        elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
            add(last[:-1] + "ies")
        else:
            add(last + "s")
        # past / -ing, several spellings; the round trip removes wrong ones
        stem_e = last[:-1] if last.endswith("e") else last
        add(stem_e + "ed")
        add(stem_e + "ing")
        add(last + "ed")
        add(last + "ing")
        if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
            add(last[:-1] + "ied")
        if len(last) >= 2:  # consonant doubling: stop -> stopped
            add(last + last[-1] + "ed")
            add(last + last[-1] + "ing")
    elif pos == "a":
        stem_e = last[:-1] if last.endswith("e") else last
        add(stem_e + "er")
        add(stem_e + "est")
        add(last + "er")
        add(last + "est")
        if last.endswith("y") and len(last) > 1:
            add(last[:-1] + "ier")
            add(last[:-1] + "iest")
        if len(last) >= 2:
            add(last + last[-1] + "er")
            add(last + last[-1] + "est")
    return c


def load_irregulars(pos_letter: str):
    """Read WordNet exception lists (noun.exc, verb.exc, adj.exc, adv.exc).
    Line format: <inflected> <lemma1> [<lemma2> ...]"""
    name = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}[pos_letter] + ".exc"
    out = []
    try:
        with wn.open(name) as f:  # works whether WordNet is zipped or extracted
            for line in f:
                if isinstance(line, bytes):
                    line = line.decode("utf8")
                parts = line.strip().split()
                if len(parts) >= 2:
                    infl, lemmas = parts[0], parts[1:]
                    out.append((norm(infl), [norm(l) for l in lemmas]))
    except Exception as e:  # keep going without irregulars rather than fail the build
        print(f"  warning: could not read {name}: {e}")
    return out


# ---------------------------------------------------------------------- build
def build(out_path: str, holdout: float, seed: int):
    ensure_wordnet()
    t0 = time.time()

    if os.path.exists(out_path):
        os.remove(out_path)
    db = sqlite3.connect(out_path)
    db.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE entries (id INTEGER PRIMARY KEY, headword TEXT NOT NULL, pos TEXT NOT NULL);
        CREATE TABLE senses  (id INTEGER PRIMARY KEY, entry_id INTEGER NOT NULL,
                              idx INTEGER NOT NULL, gloss TEXT NOT NULL,
                              example TEXT, freq INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE forms   (form TEXT NOT NULL, entry_id INTEGER NOT NULL);
        CREATE TABLE meta    (key TEXT PRIMARY KEY, value TEXT);
        """
    )

    # 1. choose holdout headwords (for NOT_FOUND eval queries)
    all_names = sorted({norm(n) for n in wn.all_lemma_names()})
    held = set()
    if holdout > 0:
        rng = random.Random(seed)
        held = set(rng.sample(all_names, int(len(all_names) * holdout)))
        print(f"holdout: {len(held)} headwords hidden")

    # 2. entries + senses
    entry_id = 0
    sense_id = 0
    entries = {}  # (headword, pos) -> id
    n_senses = 0
    for pos in "nvar":
        for raw in sorted(wn.all_lemma_names(pos=pos)):
            head = norm(raw)
            if head in held:
                continue
            key = (head, pos)
            synsets = wn.synsets(raw, pos=pos)  # already in WordNet sense order
            if not synsets:
                continue
            if key not in entries:
                entry_id += 1
                entries[key] = entry_id
                db.execute("INSERT INTO entries VALUES (?,?,?)", (entry_id, head, POS_MAP[pos]))
            eid = entries[key]
            for idx, ss in enumerate(synsets):
                freq = 0
                for lem in ss.lemmas():
                    if lem.name().lower() == raw.lower():
                        freq = lem.count()
                        break
                ex = ss.examples()[0] if ss.examples() else None
                sense_id += 1
                db.execute(
                    "INSERT INTO senses VALUES (?,?,?,?,?,?)",
                    (sense_id, eid, idx, ss.definition(), ex, freq),
                )
                n_senses += 1
    print(f"entries={len(entries)} senses={n_senses}  ({time.time()-t0:.0f}s)")

    # 3. forms: lemma itself + irregulars + round-trip-validated regular inflections
    forms = set()
    for (head, pos), eid in entries.items():
        forms.add((head, eid))
    by_head = {}
    for (head, pos), eid in entries.items():
        by_head.setdefault((head, pos), eid)

    for pos in "nvar":
        for infl, lemmas in load_irregulars(pos):
            for lem in lemmas:
                eid = by_head.get((lem, pos))
                if eid:
                    forms.add((infl, eid))

    for (head, pos), eid in entries.items():
        if pos == "r":
            continue
        for cand in regular_candidates(head, pos):
            cand_n = norm(cand)
            if cand_n == head:
                continue
            # keep only if WordNet's own lemmatiser maps the candidate back to this headword
            back = wn.morphy(cand_n.replace(" ", "_"), pos)
            if back and norm(back) == head:
                forms.add((cand_n, eid))
    db.executemany("INSERT INTO forms VALUES (?,?)", sorted(forms))
    if held:
        # a hidden word can still resolve as an inflected form of another headword
        # (e.g. "saw" -> see); those are not valid NOT_FOUND queries, so drop them
        form_set = {f for f, _ in forms}
        true_missing = sorted(w for w in held if w not in form_set)
        holdout_file = os.path.splitext(out_path)[0] + "_holdout_words.txt"
        with open(holdout_file, "w", encoding="utf8") as f:
            f.write("\n".join(true_missing))
        print(f"holdout: {len(true_missing)} of {len(held)} are truly NOT_FOUND -> {holdout_file}")
    print(f"forms={len(forms)}  ({time.time()-t0:.0f}s)")

    # 4. indexes + meta
    db.executescript(
        """
        CREATE INDEX idx_forms_form ON forms(form);
        CREATE INDEX idx_senses_entry ON senses(entry_id, freq DESC, idx);
        CREATE INDEX idx_entries_head ON entries(headword);
        """
    )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source": f"WordNet {wn.get_version()} via NLTK",
        "holdout_fraction": str(holdout),
        "license_note": "WordNet is (c) Princeton University; include its license text in your app.",
    }
    db.executemany("INSERT INTO meta VALUES (?,?)", meta.items())
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"wrote {out_path}  {os.path.getsize(out_path)/1e6:.1f} MB  total {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------- check
def lookup(db_path: str, term: str, top_k: int = 3):
    """Reference lookup: exact form match, senses ranked by corpus frequency."""
    db = sqlite3.connect(db_path)
    rows = db.execute(
        """
        SELECT e.headword, e.pos, s.gloss, s.freq, s.idx
        FROM forms f
        JOIN entries e ON e.id = f.entry_id
        JOIN senses  s ON s.entry_id = e.id
        WHERE f.form = ?
        ORDER BY s.freq DESC, s.idx ASC
        LIMIT ?
        """,
        (norm(term), top_k),
    ).fetchall()
    db.close()
    return rows


def smoke_test(db_path: str):
    print("\nsmoke test")
    for q in ["black", "ran", "geese", "kick the bucket", "flibbergast"]:
        rows = lookup(db_path, q)
        print(f"  {q!r}: {len(rows)} rows")
        for head, pos, gloss, freq, idx in rows:
            print(f"     {head} ({pos}) freq={freq}: {gloss[:70]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dictionary.db")
    ap.add_argument("--holdout", type=float, default=0.0, help="fraction of headwords to hide, e.g. 0.05")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.out, args.holdout, args.seed)
    smoke_test(args.out)
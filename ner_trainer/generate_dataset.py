"""Synthetic dataset generator for training a lightweight NER term extractor.

Generates training and validation data in BIO format using real words
and multi-word idioms from WordNet (dictionary.db).
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

DB_PATH = Path(__file__).resolve().parent.parent / "dictionary.db"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

TEMPLATES = [
    # Clean direct questions
    "what does {TERM} mean",
    "what does '{TERM}' mean",
    'what does "{TERM}" mean',
    "what is the meaning of {TERM}",
    "what is the meaning of '{TERM}'",
    'what is the meaning of "{TERM}"',
    "what does the word {TERM} mean",
    "what does the word '{TERM}' mean",
    'what does the word "{TERM}" mean',
    "what's the definition of {TERM}",
    "what's the definition of '{TERM}'",
    "tell me what is {TERM} mean",
    "tell me what is '{TERM}' mean",
    "tell me the meaning of {TERM}",
    "tell me the meaning of '{TERM}'",
    "can you define {TERM} for me",
    "can you define '{TERM}' for me",
    "define {TERM} please",
    "define '{TERM}' please",
    'define "{TERM}" please',
    "definition of {TERM}",
    "definition of '{TERM}'",
    "meaning of {TERM}",
    "meaning of '{TERM}'",
    'meaning of "{TERM}"',
    "what is {TERM}",
    "what is '{TERM}'",
    'what is "{TERM}"',
    "what is the word '{TERM}'",
    "what is the word {TERM}",
    "tell me about the word '{TERM}'",
    "tell me about the word {TERM}",

    # Casual / conversational queries
    "hey can you explain what {TERM} means",
    "hey can you explain what '{TERM}' means",
    "i was reading a book and saw '{TERM}' what does that mean",
    "my friend used the word '{TERM}' yesterday what is that",
    "i don't understand what '{TERM}' means",
    "could you tell me what {TERM} is supposed to mean",
    "could you tell me what '{TERM}' is supposed to mean",
    "can you explain the term {TERM}",
    "can you explain the term '{TERM}'",
    "what's the idea behind {TERM}",
    "what's the idea behind '{TERM}'",
    "i heard the phrase '{TERM}' today what does it mean",
    "help me understand what '{TERM}' means",
    "give me the dictionary definition of '{TERM}'",
    "explain {TERM} in simple words",
    "explain '{TERM}' in simple words",

    # Noisy / typo templates (real user query variations)
    "whta does {TERM} mean",
    "whta does '{TERM}' mean",
    "whta does the word '{TERM}' mean",
    "wats the mening of {TERM}",
    "wats the mening of '{TERM}'",
    "wats the mening fo {TERM}",
    "wats the mening fo '{TERM}'",
    "wat is the meaning of {TERM}",
    "wat is the meaning of '{TERM}'",
    "wat is th meening of {TERM}",
    "wat is th meening of '{TERM}'",
    "wat is th meeningfo {TERM}",
    "wat is th meeningfo '{TERM}'",
    "whats the meeing of {TERM}",
    "whats the meeing of '{TERM}'",
    "tell me about the wrd {TERM}",
    "tell me about the wrd '{TERM}'",
    "what is teh defenition of {TERM}",
    "what is teh defenition of '{TERM}'",
    "hey what does {TERM} mean eactly",
    "hey what does '{TERM}' mean eactly",
    "tell em about the word {TERM}",
    "tell em about the word '{TERM}'",
    "can u tell me what {TERM} is",
    "can u tell me what '{TERM}' is",

    # Minimalist / spoken inputs
    "{TERM}",
    "'{TERM}'",
    '"{TERM}"',
    "the word {TERM}",
    "the word '{TERM}'",
    "word {TERM}",
    "word '{TERM}'",
    "{TERM} meaning",
    "'{TERM}' meaning",
    "{TERM} meaning please",
    "{TERM} meaning pls",
    "{TERM} meaningpls",
    "{TERM} meanin pls",
    "{TERM} definition",
    "'{TERM}' definition",
    "what is {TERM}?",
    "what is '{TERM}'?",
    "define {TERM}",
    "define '{TERM}'",
]

NEGATIVE_QUERIES = [
    # Assistant / utility commands
    "send a message to dad",
    "send a message to mom",
    "send a text message to john",
    "call mom right now",
    "call dad please",
    "set a timer for five minutes",
    "set an alarm for seven am",
    "remind me to buy groceries",
    "remind me to take my medicine",
    "open google maps",
    "navigate to the nearest hospital",
    "play some relaxing music",
    "play jazz music on spotify",
    "turn off the lights in the bedroom",
    "turn on the air conditioner",
    "what time is it right now",
    "what day is today",
    "what is the date today",
    "what is the weather like today",
    "what's the weather in new york",
    "will it rain tomorrow morning",

    # Chit-chat / General knowledge
    "hello how are you doing today",
    "good morning how are you",
    "good afternoon my friend",
    "hey what is your name",
    "tell me a funny joke please",
    "who was the first president of the united states",
    "who won the football match yesterday",
    "how far is the moon from earth",
    "how do i bake a chocolate cake",
    "what is two plus two",
    "where is the nearest coffee shop",
    "what are you capable of doing",
    "tell me about yourself",
    "how do you calculate the area of a circle",
    "can you help me with my math homework",
    "how old is the universe",
    "why is the sky blue",
    "what is the capital of australia",
    "how much does an elephant weigh",
    "where did i leave my keys",
]


def load_candidate_terms(db_path: Path) -> Tuple[List[str], List[str]]:
    """Load single-word and multi-word forms from dictionary.db."""
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT form FROM forms WHERE length(form) >= 3")
    all_forms = [r[0] for r in cur.fetchall()]
    conn.close()

    single_words = [f for f in all_forms if " " not in f and f.isalpha()]
    multi_words = [f for f in all_forms if " " in f and not any(ch.isdigit() for ch in f)]

    return single_words, multi_words


def tokenize_with_bio(text: str, term: str | None) -> Tuple[List[str], List[str]]:
    """Tokenize query and generate BIO tags for the exact term."""
    # Split text into word tokens preserving punctuation separation (e.g. quotes as separate tokens)
    raw_tokens = re.findall(r"\w+(?:'\w+)?|[^\s\w]", text)
    tokens = [t for t in raw_tokens if t.strip()]

    if not term:
        return tokens, ["O"] * len(tokens)

    term_tokens = [t.lower() for t in re.findall(r"\w+(?:'\w+)?|[^\s\w]", term) if t.strip()]
    k = len(term_tokens)
    tags = ["O"] * len(tokens)

    # Search for term subsequence in tokens (case-insensitive)
    lower_tokens = [t.lower() for t in tokens]
    match_idx = -1
    for i in range(len(lower_tokens) - k + 1):
        if lower_tokens[i : i + k] == term_tokens:
            match_idx = i
            break

    if match_idx != -1:
        tags[match_idx] = "B-TERM"
        for j in range(1, k):
            tags[match_idx + j] = "I-TERM"

    return tokens, tags


def generate_dataset(
    single_words: List[str],
    multi_words: List[str],
    num_samples: int = 10000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate balanced dataset of single-word, multi-word, and negative queries."""
    random.seed(seed)
    dataset: List[Dict[str, Any]] = []

    # 1. Positive queries (~85%)
    num_positives = int(num_samples * 0.85)
    for _ in range(num_positives):
        template = random.choice(TEMPLATES)
        # 70% single words, 30% multi-word phrases
        if random.random() < 0.70 and single_words:
            term = random.choice(single_words)
        else:
            term = random.choice(multi_words) if multi_words else random.choice(single_words)

        sentence = template.format(TERM=term)

        # Randomly append punctuation occasionally
        if random.random() < 0.3 and not sentence.endswith(("?", "!", ".")):
            sentence += random.choice(["?", "!", ".", "??", "!!"])

        # Casing augmentation: 15% ALL CAPS, 10% Title Case
        case_roll = random.random()
        if case_roll < 0.15:
            sentence = sentence.upper()
        elif case_roll < 0.25:
            sentence = sentence.title()

        tokens, tags = tokenize_with_bio(sentence, term)
        dataset.append({
            "text": sentence,
            "tokens": tokens,
            "tags": tags,
            "target_term": term,
        })

    # 2. Negative queries (~15%)
    num_negatives = num_samples - num_positives
    for _ in range(num_negatives):
        query = random.choice(NEGATIVE_QUERIES)
        # Occasionally punctuate or uppercase negatives too
        if random.random() < 0.15:
            query = query.upper()
        tokens, tags = tokenize_with_bio(query, term=None)
        dataset.append({
            "text": query,
            "tokens": tokens,
            "tags": tags,
            "target_term": None,
        })

    random.shuffle(dataset)
    return dataset


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading dictionary forms from {DB_PATH}...")
    single_words, multi_words = load_candidate_terms(DB_PATH)
    print(f"Loaded {len(single_words)} single words and {len(multi_words)} multi-word idioms.")

    print("Generating 10,000 synthetic queries...")
    data = generate_dataset(single_words, multi_words, num_samples=10000, seed=42)

    # 85% train, 15% validation split
    split_idx = int(len(data) * 0.85)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for row in train_data:
            f.write(json.dumps(row) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for row in val_data:
            f.write(json.dumps(row) + "\n")

    print(f"Saved {len(train_data)} train samples to {train_path}")
    print(f"Saved {len(val_data)} validation samples to {val_path}")


if __name__ == "__main__":
    main()

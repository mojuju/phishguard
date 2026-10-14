"""Loading the dataset and splitting it into train / test sets (standard library only)."""
from __future__ import annotations

import csv
import random
from pathlib import Path

TEXT_CANDIDATES = ("email text", "text", "body", "email", "message", "content")
LABEL_CANDIDATES = ("email type", "label", "type", "class", "category", "is_phishing")
PHISHING_VALUES = {"1", "phishing", "phishing email", "spam", "malicious", "fraud"}

csv.field_size_limit(2**31 - 1)   # some emails are very long


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    lookup = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    for cand in candidates:                        # fall back to partial match
        for low, original in lookup.items():
            if cand in low:
                return original
    raise ValueError(f"Could not find a column like {candidates} in {columns}")


def load_dataset(path: str | Path) -> tuple[list[str], list[int]]:
    """
    Read a CSV of emails and return (texts, labels) where label 1 = phishing, 0 = safe.

    Column names are detected automatically, so most public phishing datasets
    (e.g. Kaggle "Phishing Email Detection") work without edits.
    """
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        text_col = _pick_column(reader.fieldnames or [], TEXT_CANDIDATES)
        label_col = _pick_column(reader.fieldnames or [], LABEL_CANDIDATES)

        texts, labels, seen = [], [], set()
        for row in reader:
            text = (row.get(text_col) or "").strip()
            if not text or text.lower() in {"nan", "empty"} or text in seen:
                continue                            # skip blanks and duplicates
            seen.add(text)
            texts.append(text)
            labels.append(int((row.get(label_col) or "").strip().lower() in PHISHING_VALUES))

    if len(set(labels)) < 2:
        raise ValueError("Dataset must contain both phishing and safe emails.")
    return texts, labels


def train_test_split(texts, labels, test_size=0.2, seed=42):
    """Stratified split: keeps the same phishing/safe ratio in both sets."""
    rng = random.Random(seed)
    train_idx, test_idx = [], []
    for cls in (0, 1):
        idx = [i for i, y in enumerate(labels) if y == cls]
        rng.shuffle(idx)
        cut = int(len(idx) * test_size)
        test_idx += idx[:cut]
        train_idx += idx[cut:]
    rng.shuffle(train_idx)
    pick = lambda ids, seq: [seq[i] for i in ids]
    return (pick(train_idx, texts), pick(test_idx, texts),
            pick(train_idx, labels), pick(test_idx, labels))

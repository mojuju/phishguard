"""
Multinomial Naive Bayes classifier, written from scratch in pure Python.

Idea: for every word we learn how often it appears in phishing emails and how
often in safe emails. To classify a new email we combine those word
probabilities (using Bayes' theorem) and pick the more likely class.

    log P(class | email) ∝ log P(class) + Σ log P(word | class)

Laplace smoothing (alpha) stops unseen words from giving a probability of 0.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import rules
from .text import tokenize

ML_WEIGHT = 0.75          # how much the final verdict trusts the model vs. the rules
PHISHING_THRESHOLD = 0.6
SUSPICIOUS_THRESHOLD = 0.35


class NaiveBayes:
    def __init__(self, alpha: float = 1.0, min_df: int = 2, max_features: int = 50_000):
        self.alpha = alpha
        self.min_df = min_df
        self.max_features = max_features
        self.log_prior: list[float] = [0.0, 0.0]
        self.log_likelihood: dict[str, tuple[float, float]] = {}
        self.log_unseen: tuple[float, float] = (0.0, 0.0)

    # ---------- training ----------
    def fit(self, texts: list[str], labels: list[int]) -> "NaiveBayes":
        docs = [tokenize(t) for t in texts]

        # 1. Build the vocabulary: words seen in at least `min_df` emails, most common first.
        doc_freq = Counter()
        for tokens in docs:
            doc_freq.update(set(tokens))
        vocab = [w for w, c in doc_freq.most_common(self.max_features) if c >= self.min_df]
        vocab_set = set(vocab)

        # 2. Count every word separately for safe (0) and phishing (1) emails.
        counts = (Counter(), Counter())
        for tokens, y in zip(docs, labels):
            counts[y].update(t for t in tokens if t in vocab_set)

        # 3. Class priors: how common each class is.
        n = len(labels)
        n_phish = sum(labels)
        self.log_prior = [math.log((n - n_phish) / n), math.log(n_phish / n)]

        # 4. Word likelihoods with Laplace smoothing.
        totals = [sum(counts[c].values()) + self.alpha * len(vocab) for c in (0, 1)]
        self.log_likelihood = {
            w: (math.log((counts[0][w] + self.alpha) / totals[0]),
                math.log((counts[1][w] + self.alpha) / totals[1]))
            for w in vocab
        }
        self.log_unseen = (math.log(self.alpha / totals[0]), math.log(self.alpha / totals[1]))
        return self

    # ---------- prediction ----------
    def _log_scores(self, text: str) -> tuple[float, float]:
        safe, phish = self.log_prior
        for tok in tokenize(text):
            ll = self.log_likelihood.get(tok)
            if ll is not None:
                safe += ll[0]
                phish += ll[1]
        return safe, phish

    def predict_proba(self, text: str) -> float:
        """Probability (0..1) that the email is phishing."""
        safe, phish = self._log_scores(text)
        diff = safe - phish                   # numerically stable sigmoid
        if diff > 700:
            return 0.0
        return 1.0 / (1.0 + math.exp(diff))

    def predict(self, text: str) -> int:
        return int(self.predict_proba(text) >= 0.5)

    # ---------- explanations ----------
    def term_weight(self, term: str) -> float:
        """> 0 means the word points to phishing, < 0 points to safe."""
        ll = self.log_likelihood.get(term)
        return 0.0 if ll is None else ll[1] - ll[0]

    def top_terms(self, text: str, k: int = 8) -> list[tuple[str, float]]:
        seen = {t: self.term_weight(t) for t in set(tokenize(text))}
        ranked = sorted(((t, w) for t, w in seen.items() if w > 0), key=lambda x: x[1], reverse=True)
        return [(t, w) for t, w in ranked if "token" not in t][:k]

    def most_phishy_terms(self, k: int = 20) -> list[str]:
        ranked = sorted(self.log_likelihood, key=self.term_weight, reverse=True)
        return [t for t in ranked if "token" not in t][:k]

    # ---------- save / load ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha, "min_df": self.min_df, "max_features": self.max_features,
            "log_prior": self.log_prior, "log_unseen": self.log_unseen,
            "log_likelihood": self.log_likelihood,
        }
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "NaiveBayes":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            p = json.load(f)
        m = cls(p["alpha"], p["min_df"], p["max_features"])
        m.log_prior = p["log_prior"]
        m.log_unseen = tuple(p["log_unseen"])
        m.log_likelihood = {w: tuple(v) for w, v in p["log_likelihood"].items()}
        return m


@dataclass
class Verdict:
    label: str                 # "Phishing" | "Suspicious" | "Safe"
    score: float               # combined risk 0..1
    ml_probability: float
    rule_score: float
    flags: list[rules.Flag] = field(default_factory=list)
    top_terms: list[tuple[str, float]] = field(default_factory=list)


def analyze_email(model: NaiveBayes, text: str) -> Verdict:
    """Full analysis of one email: model probability + rule checks."""
    ml_prob = model.predict_proba(text)
    flags = rules.analyze(text)
    r_score = rules.rule_score(flags)
    score = ML_WEIGHT * ml_prob + (1 - ML_WEIGHT) * r_score

    if score >= PHISHING_THRESHOLD:
        label = "Phishing"
    elif score >= SUSPICIOUS_THRESHOLD:
        label = "Suspicious"
    else:
        label = "Safe"
    return Verdict(label, score, ml_prob, r_score, flags, model.top_terms(text))

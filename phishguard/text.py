"""Text preprocessing: turn raw email text into a list of tokens (features)."""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z][a-z0-9']+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
MONEY_RE = re.compile(r"[$€£]\s?\d")

STOP_WORDS = frozenset("""
a about above after again all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself just me
more most my myself no nor not now of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then there these they
this those through to too under until up very was we were what when where which while who whom
why will with would you your yours yourself yourselves
""".split())

MAX_TOKENS_PER_EMAIL = 3000


def normalize(text: str) -> str:
    """Lowercase and replace links, emails, money and numbers with placeholder words.

    The placeholders let the model learn "this email contains a link" instead of
    memorising one specific URL.
    """
    text = text.lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = NUMBER_RE.sub(" numtoken ", text)
    return text


def tokenize(text: str) -> list[str]:
    """Words (minus stop words) plus two-word phrases (bigrams)."""
    words = [w for w in TOKEN_RE.findall(normalize(text)) if w not in STOP_WORDS]
    words = words[:MAX_TOKENS_PER_EMAIL]
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
    return words + bigrams

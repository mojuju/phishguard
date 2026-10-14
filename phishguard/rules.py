"""
Rule-based phishing indicators.

These are the checks a SOC analyst runs by hand when triaging a suspicious
email. They complement the ML model: the model learns word patterns, the
rules catch technical tricks (raw-IP links, URL shorteners, risky files).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

URGENCY_TERMS = (
    "urgent", "immediately", "act now", "within 24 hours", "account suspended",
    "account locked", "verify your account", "confirm your", "unusual activity",
    "final notice", "limited time",
)
CREDENTIAL_TERMS = (
    "password", "login", "log in", "sign in", "ssn", "social security",
    "bank account", "credit card", "pin",
)
REWARD_TERMS = ("winner", "you have won", "prize", "lottery", "free gift", "claim your")

URL_SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "rb.gy")
RISKY_EXTENSIONS = ("exe", "scr", "js", "vbs", "bat", "cmd", "jar", "iso", "zip", "rar", "docm", "xlsm")

URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)
IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
FILE_RE = re.compile(r"\b[\w\-]+\.(" + "|".join(RISKY_EXTENSIONS) + r")\b", re.IGNORECASE)


@dataclass(frozen=True)
class Flag:
    """One red flag found in an email."""
    rule: str
    severity: str          # "low" | "medium" | "high"
    detail: str


SEVERITY_POINTS = {"low": 1, "medium": 2, "high": 3}


def extract_urls(text: str) -> list[str]:
    return URL_RE.findall(text)


def _find_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    low = text.lower()
    return [t for t in terms if re.search(r"\b" + re.escape(t) + r"\b", low)]


def check_url(url: str) -> list[Flag]:
    flags: list[Flag] = []
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if IP_HOST_RE.match(host):
        flags.append(Flag("ip_address_link", "high", f"Link points to a raw IP address: {url}"))
    if "@" in parsed.netloc:
        flags.append(Flag("obfuscated_link", "high", f"'@' in link hides the real destination: {url}"))
    if any(host == s or host.endswith("." + s) for s in URL_SHORTENERS):
        flags.append(Flag("url_shortener", "medium", f"Shortened link hides the destination: {url}"))
    if parsed.scheme == "http":
        flags.append(Flag("no_https", "low", f"Link is not encrypted (HTTP): {url}"))
    if host.count("-") >= 3 or host.count(".") >= 4:
        flags.append(Flag("suspicious_domain", "medium", f"Unusually complex domain name: {host}"))
    if "xn--" in host:
        flags.append(Flag("punycode_domain", "high", f"Look-alike (punycode) domain: {host}"))
    return flags


def analyze(text: str) -> list[Flag]:
    """Run every rule against an email and return the red flags found."""
    flags: list[Flag] = []

    if found := _find_terms(text, URGENCY_TERMS):
        flags.append(Flag("urgency", "medium", "Pressure language: " + ", ".join(found)))
    if found := _find_terms(text, CREDENTIAL_TERMS):
        flags.append(Flag("credential_request", "medium", "Asks about credentials: " + ", ".join(found)))
    if found := _find_terms(text, REWARD_TERMS):
        flags.append(Flag("reward_bait", "medium", "Too-good-to-be-true offer: " + ", ".join(found)))

    for url in extract_urls(text):
        flags.extend(check_url(url))

    files = sorted({m.group(0) for m in FILE_RE.finditer(text)})
    if files:
        flags.append(Flag("risky_attachment", "high", "Risky file type mentioned: " + ", ".join(files)))

    return flags


def rule_score(flags: list[Flag]) -> float:
    """Turn flags into a 0..1 risk score (8+ points = maximum)."""
    points = sum(SEVERITY_POINTS[f.severity] for f in flags)
    return min(points / 8.0, 1.0)

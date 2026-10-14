import csv

from phishguard import data, metrics
from phishguard.model import NaiveBayes, analyze_email
from phishguard.text import tokenize

PHISH = [
    "Your account is suspended, verify your password immediately at http://10.0.0.5/login",
    "You have won a prize! Claim your free gift now, enter your credit card",
    "Unusual activity detected. Confirm your bank account login within 24 hours",
    "Final notice: verify your account or it will be locked, click https://bit.ly/x",
] * 10
SAFE = [
    "Hi team, the meeting is moved to Thursday at 10am",
    "Attached are the slides from today's presentation, thanks everyone",
    "Can we have lunch tomorrow? Let me know what time works",
    "The quarterly report is ready for review in the shared folder",
] * 10


def make_csv(tmp_path):
    path = tmp_path / "emails.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "Email Text", "Email Type"])
        for i, t in enumerate(PHISH):
            w.writerow([i, f"{t} #{i}", "Phishing Email"])
        for i, t in enumerate(SAFE):
            w.writerow([i, f"{t} #{i}", "Safe Email"])
    return path


def trained(tmp_path):
    texts, labels = data.load_dataset(make_csv(tmp_path))
    return NaiveBayes().fit(texts, labels)


def test_tokenize_replaces_links_and_numbers():
    tokens = tokenize("Visit http://x.com now, pay $500")
    assert "urltoken" in tokens and "moneytoken" in tokens
    assert "visit urltoken" in tokens          # bigrams included


def test_load_dataset_detects_columns(tmp_path):
    texts, labels = data.load_dataset(make_csv(tmp_path))
    assert len(texts) == len(PHISH) + len(SAFE)
    assert sum(labels) == len(PHISH)


def test_split_is_stratified():
    texts = [str(i) for i in range(100)]
    labels = [1] * 30 + [0] * 70
    _, X_test, _, y_test = data.train_test_split(texts, labels, test_size=0.2)
    assert len(X_test) == 20 and sum(y_test) == 6


def test_model_separates_classes(tmp_path):
    model = trained(tmp_path)
    phish = analyze_email(model, "Verify your password immediately at http://10.1.1.1/login")
    safe = analyze_email(model, "Hi team, the meeting is on Thursday")
    assert phish.score > safe.score
    assert phish.label in {"Phishing", "Suspicious"}
    assert safe.label == "Safe"
    assert phish.top_terms


def test_save_and_load(tmp_path):
    model = trained(tmp_path)
    model.save(tmp_path / "m.json.gz")
    loaded = NaiveBayes.load(tmp_path / "m.json.gz")
    text = "verify your account password"
    assert abs(loaded.predict_proba(text) - model.predict_proba(text)) < 1e-9


def test_metrics():
    r = metrics.report([1, 1, 0, 0], [1, 0, 0, 1], [0.9, 0.4, 0.2, 0.6])
    assert r["accuracy"] == 0.5 and r["recall"] == 0.5 and r["roc_auc"] == 0.75

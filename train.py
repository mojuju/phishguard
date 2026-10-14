"""
Train and evaluate the PhishGuard model.

Usage:
    python train.py                          # uses data/emails.csv
    python train.py --data path/to/file.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from phishguard import data, metrics
from phishguard.model import NaiveBayes


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PhishGuard model.")
    parser.add_argument("--data", default="data/emails.csv", help="CSV with email text and labels")
    parser.add_argument("--out", default="models/phishguard.json.gz", help="Where to save the model")
    parser.add_argument("--reports", default="reports", help="Folder for metrics and charts")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"Dataset not found: {args.data}\nSee data/README.md for how to download it.")

    start = time.time()
    texts, labels = data.load_dataset(args.data)
    print(f"Loaded {len(texts):,} emails  |  phishing: {sum(labels):,}  safe: {len(labels) - sum(labels):,}")

    X_train, X_test, y_train, y_test = data.train_test_split(texts, labels, args.test_size)
    print(f"Training on {len(X_train):,} emails, testing on {len(X_test):,}...")

    model = NaiveBayes().fit(X_train, y_train)
    print(f"Vocabulary: {len(model.log_likelihood):,} words and phrases")

    scores = [model.predict_proba(t) for t in X_test]
    preds = [int(s >= 0.5) for s in scores]
    result = metrics.report(y_test, preds, scores)
    result.update({"emails_total": len(texts), "emails_test": len(X_test),
                   "top_phishing_terms": model.most_phishy_terms(20)})

    print()
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{k:>10}: {result[k]:.2%}")
    print(f"   missed phishing: {result['phishing_missed']}   |   false alarms: {result['false_alarms']}")
    print("\nTop phishing indicators learned:", ", ".join(result["top_phishing_terms"][:12]))

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (reports / "confusion_matrix.svg").write_text(metrics.confusion_matrix_svg(result), encoding="utf-8")

    model.save(args.out)
    print(f"\nSaved model  -> {args.out}")
    print(f"Saved report -> {reports / 'metrics.json'}, {reports / 'confusion_matrix.svg'}")
    print(f"Done in {time.time() - start:.0f}s. Now run:  python app.py")


if __name__ == "__main__":
    main()

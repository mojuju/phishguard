"""Evaluation metrics (standard library only)."""
from __future__ import annotations


def confusion(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def roc_auc(y_true: list[int], scores: list[float]) -> float:
    """Probability that a random phishing email scores higher than a random safe one."""
    ranked = sorted(zip(scores, y_true))
    ranks, i = [0.0] * len(ranked), 0
    while i < len(ranked):                       # average ranks for ties
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    pos_rank_sum = sum(r for r, (_, y) in zip(ranks, ranked) if y == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def report(y_true: list[int], y_pred: list[int], scores: list[float]) -> dict[str, float]:
    c = confusion(y_true, y_pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": round((c["tp"] + c["tn"]) / len(y_true), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc(y_true, scores), 4),
        "phishing_missed": c["fn"],
        "false_alarms": c["fp"],
        **c,
    }


def confusion_matrix_svg(c: dict[str, int]) -> str:
    """A small, self-contained SVG chart of the confusion matrix."""
    cells = [("Safe", "Safe", c["tn"], True), ("Safe", "Phishing", c["fp"], False),
             ("Phishing", "Safe", c["fn"], False), ("Phishing", "Phishing", c["tp"], True)]
    peak = max(v for *_, v, _ in cells) or 1
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="420" height="380" font-family="Segoe UI, Arial, sans-serif">',
        '<rect width="420" height="380" fill="#ffffff"/>',
        '<text x="250" y="30" text-anchor="middle" font-size="16" font-weight="600" fill="#1f2937">Confusion matrix (test set)</text>',
        '<text x="250" y="60" text-anchor="middle" font-size="13" fill="#4b5563">Predicted</text>',
        '<text x="30" y="210" text-anchor="middle" font-size="13" fill="#4b5563" transform="rotate(-90 30 210)">Actual</text>',
    ]
    for i, name in enumerate(("Safe", "Phishing")):
        parts.append(f'<text x="{185 + i * 130}" y="85" text-anchor="middle" font-size="13" fill="#374151">{name}</text>')
        parts.append(f'<text x="105" y="{155 + i * 130}" text-anchor="end" font-size="13" fill="#374151">{name}</text>')
    for idx, (_, _, value, correct) in enumerate(cells):
        r, col = divmod(idx, 2)
        x, y = 120 + col * 130, 95 + r * 130
        shade = 0.15 + 0.85 * value / peak
        color = f"rgba(37,99,235,{shade:.2f})" if correct else f"rgba(220,38,38,{shade:.2f})"
        text_color = "#ffffff" if shade > 0.55 else "#111827"
        parts.append(f'<rect x="{x}" y="{y}" width="126" height="126" rx="8" fill="{color}"/>')
        parts.append(f'<text x="{x + 63}" y="{y + 70}" text-anchor="middle" font-size="24" font-weight="700" fill="{text_color}">{value:,}</text>')
    parts.append("</svg>")
    return "\n".join(parts)

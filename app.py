"""
PhishGuard web app (standard library only, no installs needed).

Run:  python app.py      then open http://localhost:8501
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from phishguard.model import NaiveBayes, analyze_email

MODEL_PATH = Path("models/phishguard.json.gz")
HOST, PORT = "127.0.0.1", 8501
MAX_BODY = 1_000_000       # 1 MB limit on pasted emails

RULE_NAMES = {
    "ip_address_link": "IP address link", "obfuscated_link": "Obfuscated link",
    "url_shortener": "URL shortener", "no_https": "No HTTPS",
    "suspicious_domain": "Suspicious domain", "punycode_domain": "Look-alike domain",
    "urgency": "Urgency", "credential_request": "Credential request",
    "reward_bait": "Reward bait", "risky_attachment": "Risky attachment",
}

PAGE = (Path(__file__).parent / "phishguard" / "static" / "index.html").read_text(encoding="utf-8")


def make_handler(model: NaiveBayes):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"Not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/analyze":
                return self._send(404, b"Not found", "text/plain")
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, b'{"error":"Email too large"}', "application/json")
            try:
                email = json.loads(self.rfile.read(length) or b"{}").get("email", "")
            except json.JSONDecodeError:
                return self._send(400, b'{"error":"Bad request"}', "application/json")

            v = analyze_email(model, str(email))
            result = {
                "label": v.label,
                "score": round(v.score, 4),
                "ml_probability": round(v.ml_probability, 4),
                "flags": [{"name": RULE_NAMES.get(f.rule, f.rule), "severity": f.severity, "detail": f.detail}
                          for f in sorted(v.flags, key=lambda f: ["high", "medium", "low"].index(f.severity))],
                "top_terms": [t for t, _ in v.top_terms],
            }
            self._send(200, json.dumps(result).encode("utf-8"), "application/json")

        def log_message(self, *args):     # keep the terminal quiet
            pass

    return Handler


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run `python train.py` first.")
    model = NaiveBayes.load(MODEL_PATH)
    server = ThreadingHTTPServer((HOST, PORT), make_handler(model))
    url = f"http://localhost:{PORT}"
    print(f"PhishGuard running at {url}   (press Ctrl+C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

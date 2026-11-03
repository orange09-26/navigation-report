# -*- coding: utf-8 -*-
"""Threaded deterministic qwen relay with bounded output length."""

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY_PATH = "/home/orange/.dashscope_key"
API = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = os.getenv("MAPGPT_MODEL", "qwen-plus")
LOG_PATH = os.getenv("MAPGPT_RELAY_LOG", "/home/orange/relay_v5.jsonl")
MAX_OUTPUT_TOKENS = int(os.getenv("MAPGPT_MAX_OUTPUT_TOKENS", "500"))
_log_lock = threading.Lock()


def call_llm(prompt):
    with open(KEY_PATH, encoding="utf-8") as fh:
        key = fh.read().strip()
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }).encode()
    req = urllib.request.Request(
        API,
        data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    response = urllib.request.urlopen(req, timeout=180)
    obj = json.loads(response.read().decode())
    content = obj["choices"][0]["message"]["content"]
    meta = {
        "request_id": response.headers.get("X-DashScope-Request-Id")
        or response.headers.get("X-Request-Id") or "",
        "model": obj.get("model", MODEL),
        "usage": obj.get("usage", {}),
        "created": obj.get("created"),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return content, meta


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n).decode())
        try:
            content, meta = call_llm(data["prompt"])
            payload = json.dumps({"result": content, **meta}).encode()
            code = 200
        except urllib.error.HTTPError as exc:
            payload = json.dumps({
                "error": "HTTP %s %s" % (exc.code, exc.read().decode()[:300])
            }).encode()
            code = 500
        except Exception as exc:
            payload = json.dumps({
                "error": "%s: %s" % (type(exc).__name__, exc)
            }).encode()
            code = 500
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(
        f"relay_v5 listening on 127.0.0.1:{port} model={MODEL} "
        f"max_output_tokens={MAX_OUTPUT_TOKENS}",
        flush=True,
    )
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()

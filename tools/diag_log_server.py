#!/usr/bin/env python3
"""Tiny local server that stores vid2model diagnostics into NDJSON file."""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_handler(output_path: Path):  # type: ignore[override]
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "Vid2ModelDiagServer/1.0"

        def _send_cors_headers(self) -> None:
            origin = self.headers.get("Origin", "")
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Credentials", "true")
                self.send_header("Vary", "Origin")
            else:
                self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._send_cors_headers()
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/diag":
                self.send_response(404)
                self._send_cors_headers()
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b""
            text = raw.decode("utf-8", errors="replace")
            try:
                payload = json.loads(text) if text else {}
            except json.JSONDecodeError:
                payload = {"raw": text, "parseError": True}

            record = {
                "receivedAt": iso_now(),
                "client": self.client_address[0],
                "payload": payload,
            }
            line = json.dumps(record, ensure_ascii=False)

            with lock:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")

            self.send_response(204)
            self._send_cors_headers()
            self.end_headers()

        def log_message(self, fmt: str, *args) -> None:
            # Keep server stdout clean; file is the main output.
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive browser diagnostics and write NDJSON logs.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "output" / "retarget-diag.ndjson"),
        help="Output NDJSON file path",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    handler_cls = make_handler(output_path)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"[diag-log-server] listening on http://{args.host}:{args.port}/diag")
    print(f"[diag-log-server] writing to: {output_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("[diag-log-server] stopped")


if __name__ == "__main__":
    main()

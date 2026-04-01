import argparse
import json
import mimetypes
import os
import ssl
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
STATIC_ROOT = PROJECT_ROOT / "static"
MODELS_ROOT = Path(os.environ.get("CHRONICLE_MODEL_ROOT", PROJECT_ROOT / "models")).resolve()
GOOGLE_NEWS_SEARCH_URL = "https://news.google.com/rss/search"


def main():
    parser = argparse.ArgumentParser(description="Chronicle browser-only static server")
    parser.add_argument("--host", default=os.environ.get("CHRONICLE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CHRONICLE_PORT", "8000")))
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ChronicleStaticHandler)
    print(f"Chronicle running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nChronicle stopped.")
    finally:
        server.server_close()


class ChronicleStaticHandler(BaseHTTPRequestHandler):
    server_version = "ChronicleStatic/1.0"

    def do_GET(self):
        return self.dispatch_request(send_body=True)

    def do_HEAD(self):
        return self.dispatch_request(send_body=False)

    def do_POST(self):
        return self.send_not_found(send_body=True)

    def log_message(self, format_string, *args):
        return

    def dispatch_request(self, send_body):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            return self.serve_file(
                FRONTEND_ROOT / "index.html",
                content_type="text/html; charset=utf-8",
                send_body=send_body,
            )

        if path.startswith("/static/"):
            relative_path = path.removeprefix("/static/")
            return self.serve_safe_path(
                STATIC_ROOT,
                relative_path,
                send_body=send_body,
                cache_header="no-store",
            )

        if path.startswith("/models/"):
            relative_path = path.removeprefix("/models/")
            return self.serve_safe_path(
                MODELS_ROOT,
                relative_path,
                send_body=send_body,
                cache_header="public, max-age=31536000, immutable",
            )

        if path == "/search/google":
            return self.serve_google_search(parsed.query, send_body=send_body)

        return self.send_not_found(send_body=send_body)

    def serve_safe_path(self, base_root, relative_path, send_body, cache_header):
        candidate_path = (base_root / relative_path).resolve()
        base_root = base_root.resolve()
        if base_root not in candidate_path.parents and candidate_path != base_root:
            return self.send_not_found(send_body=send_body)
        if not candidate_path.exists() or not candidate_path.is_file():
            return self.send_not_found(send_body=send_body)
        return self.serve_file(candidate_path, send_body=send_body, cache_header=cache_header)

    def serve_file(self, file_path, content_type=None, send_body=True, cache_header="no-store"):
        try:
            content = file_path.read_bytes()
        except FileNotFoundError:
            return self.send_not_found(send_body=send_body)

        guessed_type = content_type
        if guessed_type is None:
            guessed_type, _ = mimetypes.guess_type(str(file_path))
            guessed_type = guessed_type or "application/octet-stream"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", guessed_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", cache_header)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.end_headers()
        if send_body:
            self.wfile.write(content)

    def serve_google_search(self, query_string, send_body):
        params = parse_qs(query_string)
        query = str(params.get("q", [""])[0]).strip()
        if not query:
            return self.send_json({"error": "Missing query"}, status=HTTPStatus.BAD_REQUEST, send_body=send_body)

        upstream_params = {
            "q": query,
            "hl": str(params.get("hl", ["en-US"])[0]).strip() or "en-US",
            "gl": str(params.get("gl", ["US"])[0]).strip() or "US",
            "ceid": str(params.get("ceid", ["US:en"])[0]).strip() or "US:en",
        }
        upstream_url = f"{GOOGLE_NEWS_SEARCH_URL}?{urlencode(upstream_params)}"
        request = Request(
            upstream_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
                "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
            },
        )

        try:
            with urlopen(request, timeout=8, context=build_ssl_context()) as response:
                content = response.read()
                content_type = response.headers.get_content_type() or "application/rss+xml"
        except Exception as exc:  # noqa: BLE001
            return self.send_json(
                {"error": f"Google relay failed: {exc}"},
                status=HTTPStatus.BAD_GATEWAY,
                send_body=send_body,
            )

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            self.wfile.write(content)

    def send_json(self, payload, status=HTTPStatus.OK, send_body=True):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def send_not_found(self, send_body):
        body = b"Not found"
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.end_headers()
        if send_body:
            self.wfile.write(body)


def build_ssl_context():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl._create_unverified_context()


if __name__ == "__main__":
    main()

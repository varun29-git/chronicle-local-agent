import argparse
import json
import mimetypes
import os
import platform
import sqlite3
import threading
import traceback
import uuid
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import newsletter_agent
from newsletter_schema import DB_PATH, initialize_database

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
STATIC_ROOT = PROJECT_ROOT / "static"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "newsletters"

JOBS = {}
JOBS_LOCK = threading.Lock()

ALLOWED_DEPTHS = {"low", "medium", "high"}
ALLOWED_STYLES = {"concise", "feynman", "soc", "custom"}
ALLOWED_DEVICE_CLASSES = {
    "",
    "macbook",
    "midrange_laptop",
    "gaming_laptop",
    "midrange_phone",
    "flagship_phone",
}


def main():
    parser = argparse.ArgumentParser(description="Chronicle local-first newsletter studio")
    parser.add_argument("--host", default=os.environ.get("CHRONICLE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CHRONICLE_PORT", "8000")))
    args = parser.parse_args()

    initialize_database()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), ChronicleHandler)
    print(f"Chronicle running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nChronicle stopped.")
    finally:
        server.server_close()


class ChronicleHandler(BaseHTTPRequestHandler):
    server_version = "ChronicleHTTP/1.0"

    def do_GET(self):
        return self.dispatch_request(send_body=True)

    def do_HEAD(self):
        return self.dispatch_request(send_body=False)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/generate":
            payload = self.read_json_body()
            if payload is None:
                return self.send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)

            try:
                normalized_payload = normalize_generation_payload(payload)
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

            try:
                job = create_generation_job(normalized_payload)
            except RuntimeError as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)

            return self.send_json({"job": job}, status=HTTPStatus.ACCEPTED)

        if parsed.path == "/api/research":
            payload = self.read_json_body()
            if payload is None:
                return self.send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)

            try:
                normalized_payload = normalize_generation_payload(payload)
                research_bundle = collect_research_bundle(normalized_payload)
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

            return self.send_json({"research": research_bundle}, status=HTTPStatus.OK)

        if parsed.path == "/api/runs/save":
            payload = self.read_json_body()
            if payload is None:
                return self.send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)

            try:
                run_record = save_browser_issue(payload)
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

            return self.send_json({"run": run_record}, status=HTTPStatus.CREATED)

        return self.send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format_string, *args):
        return

    def dispatch_request(self, send_body):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            return self.serve_file(
                FRONTEND_ROOT / "index.html",
                "text/html; charset=utf-8",
                send_body=send_body,
            )

        if path.startswith("/static/"):
            relative_path = path.removeprefix("/static/")
            return self.serve_safe_path(STATIC_ROOT, relative_path, send_body=send_body)

        if path.startswith("/newsletters/"):
            relative_path = path.removeprefix("/newsletters/")
            return self.serve_safe_path(OUTPUT_ROOT, relative_path, send_body=send_body)

        if path == "/api/status":
            return self.send_json(build_status_payload(), send_body=send_body)

        if path == "/api/runs":
            query = parse_qs(parsed.query)
            try:
                limit = max(1, min(int(query.get("limit", ["24"])[0]), 60))
            except ValueError:
                limit = 24
            return self.send_json({"runs": fetch_runs(limit=limit)}, send_body=send_body)

        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/").strip()
            job = get_job(job_id)
            if not job:
                return self.send_json(
                    {"error": "Job not found"},
                    status=HTTPStatus.NOT_FOUND,
                    send_body=send_body,
                )
            return self.send_json({"job": job}, send_body=send_body)

        return self.send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND, send_body=send_body)

    def read_json_body(self):
        content_length = self.headers.get("Content-Length", "0").strip()
        try:
            total_bytes = int(content_length)
        except ValueError:
            return None

        if total_bytes <= 0:
            return {}

        raw_body = self.rfile.read(total_bytes)
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def send_json(self, payload, status=HTTPStatus.OK, send_body=True):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def serve_safe_path(self, base_root, relative_path, send_body=True):
        candidate_path = (base_root / relative_path).resolve()
        base_root = base_root.resolve()
        if base_root not in candidate_path.parents and candidate_path != base_root:
            return self.send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND, send_body=send_body)
        if not candidate_path.exists() or not candidate_path.is_file():
            return self.send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND, send_body=send_body)
        return self.serve_file(candidate_path, send_body=send_body)

    def serve_file(self, file_path, content_type=None, send_body=True):
        try:
            content = file_path.read_bytes()
        except FileNotFoundError:
            return self.send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND, send_body=send_body)

        guessed_type = content_type
        if guessed_type is None:
            guessed_type, _ = mimetypes.guess_type(str(file_path))
            guessed_type = guessed_type or "application/octet-stream"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", guessed_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if send_body:
            self.wfile.write(content)


def normalize_generation_payload(payload):
    brief = str(payload.get("brief", "")).strip()
    if not brief:
        raise ValueError("A newsletter brief is required.")

    depth = str(payload.get("depth", "medium")).strip().lower()
    if depth not in ALLOWED_DEPTHS:
        raise ValueError("Depth must be low, medium, or high.")

    explanation_style = str(payload.get("explanation_style", "concise")).strip().lower()
    if explanation_style not in ALLOWED_STYLES:
        raise ValueError("Explanation style must be concise, feynman, soc, or custom.")

    style_instructions = str(payload.get("style_instructions", "")).strip()
    if explanation_style == "custom" and not style_instructions:
        raise ValueError("Custom style instructions are required when explanation style is custom.")

    days = coerce_int(payload.get("days", newsletter_agent.DEFAULT_DAYS), minimum=1, maximum=30)
    queries = coerce_optional_int(payload.get("queries"), minimum=1, maximum=8)
    results_per_query = coerce_optional_int(payload.get("results_per_query"), minimum=1, maximum=8)

    device_class = str(payload.get("device_class", "")).strip().lower()
    if device_class not in ALLOWED_DEVICE_CLASSES:
        raise ValueError("Unsupported device class override.")

    return {
        "brief": brief,
        "days": days,
        "depth": depth,
        "explanation_style": explanation_style,
        "style_instructions": style_instructions,
        "queries": queries,
        "results_per_query": results_per_query,
        "device_class": device_class or None,
    }


def coerce_int(value, minimum, maximum):
    try:
        parsed_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected a numeric value.") from exc
    return max(minimum, min(parsed_value, maximum))


def coerce_optional_int(value, minimum, maximum):
    if value in {None, "", "null"}:
        return None
    return coerce_int(value, minimum, maximum)


def build_status_payload():
    runtime = build_runtime_snapshot()
    return {
        "product": {
            "name": "Chronicle",
            "tagline": "Local-first newsletter studio",
            "runs_on_device": True,
        },
        "runtime": runtime,
        "browser_ai": build_browser_ai_snapshot(),
        "active_job": find_active_job(),
        "run_count": count_runs(),
        "issue_count": count_issues(),
    }


def build_runtime_snapshot():
    system_info = newsletter_agent.detect_system_info()
    runtime_profile = newsletter_agent.choose_model_profile(system_info)
    model_path = newsletter_agent.normalize_model_reference(runtime_profile["model_path"])
    local_model = newsletter_agent.is_local_model_reference(model_path)
    model_ready = os.path.exists(model_path) if local_model else False
    dependencies_ready, dependency_message = detect_runtime_dependencies(runtime_profile["runtime_backend"])

    return {
        "hostname": platform.node() or "This device",
        "system_name": system_info.get("system_name") or platform.system(),
        "platform": system_info.get("platform") or platform.platform(),
        "machine": system_info.get("machine") or platform.machine(),
        "hardware_model": system_info.get("hardware_model") or "",
        "chip": system_info.get("chip") or "",
        "gpu_model": system_info.get("gpu_model") or "",
        "memory_total_gb": round(system_info.get("memory_total_gb", 0.0), 1),
        "memory_available_gb": round(system_info.get("memory_available_gb", 0.0), 1),
        "device_class": runtime_profile["device_class"],
        "runtime_backend": runtime_profile["runtime_backend"],
        "slice_label": runtime_profile["slice_label"],
        "model_path": model_path,
        "model_local": local_model,
        "model_ready": model_ready,
        "dependencies_ready": dependencies_ready,
        "dependency_message": dependency_message,
        "local_model_root": str(Path(newsletter_agent.DEFAULT_MODEL_ROOT).resolve()),
        "output_directory": str(OUTPUT_ROOT.resolve()),
    }


def build_browser_ai_snapshot():
    return {
        "provider": "@huggingface/transformers",
        "preferred_backend": "webgpu",
        "fallback_backend": "wasm",
        "high_tier_model": "onnx-community/Qwen2-0.5B-Instruct-ONNX",
        "low_tier_model": "onnx-community/SmolLM2-135M-Instruct-ONNX-MHA",
        "cpu_fallback_model": "onnx-community/SmolLM2-135M-Instruct-ONNX",
    }


def detect_runtime_dependencies(runtime_backend):
    try:
        if runtime_backend == "mlx":
            import mlx_lm  # noqa: F401
            return True, "mlx_lm available"
        if runtime_backend == "transformers":
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True, "torch and transformers available"
    except ModuleNotFoundError as exc:
        return False, f"Missing dependency: {exc.name}"
    return False, "Unsupported runtime backend"


def collect_research_bundle(payload):
    settings = newsletter_agent.build_research_settings(
        payload["depth"],
        payload["queries"],
        payload["results_per_query"],
    )
    plan = newsletter_agent.build_fallback_research_plan(
        payload["brief"],
        payload["days"],
        settings["query_limit"],
        payload["depth"],
    )
    market_snapshot = newsletter_agent.fetch_market_snapshot(payload["brief"])

    collected_sources = []
    for query in plan["queries"]:
        results = newsletter_agent.search_web(query, settings["results_per_query"])
        for rank_index, result in enumerate(results, start=1):
            article_text = newsletter_agent.fetch_article_text(result["url"], settings["article_chars"])
            source_text = newsletter_agent.build_source_text(result, article_text)
            if not source_text:
                continue

            collected_sources.append(
                {
                    "query": query,
                    "rank_index": rank_index,
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": result["snippet"],
                    "article_text": article_text,
                    "source_text": source_text,
                }
            )

    return {
        "brief": payload["brief"],
        "days": payload["days"],
        "depth": payload["depth"],
        "explanation_style": payload["explanation_style"],
        "style_instructions": payload["style_instructions"],
        "plan": plan,
        "market_snapshot": market_snapshot,
        "sources": collected_sources,
    }


def save_browser_issue(payload):
    brief = str(payload.get("brief", "")).strip()
    title = str(payload.get("title", "")).strip()
    markdown = str(payload.get("markdown", "")).strip()
    depth = str(payload.get("depth", "medium")).strip().lower() or "medium"
    explanation_style = str(payload.get("explanation_style", "concise")).strip().lower() or "concise"
    style_instructions = str(payload.get("style_instructions", "")).strip()

    if not brief:
        raise ValueError("A newsletter brief is required.")
    if not title:
        raise ValueError("A generated title is required.")
    if not markdown:
        raise ValueError("Generated markdown is required.")
    if depth not in ALLOWED_DEPTHS:
        raise ValueError("Depth must be low, medium, or high.")
    if explanation_style not in ALLOWED_STYLES:
        raise ValueError("Explanation style must be concise, feynman, soc, or custom.")

    queries = payload.get("queries") or []
    sections = payload.get("sections") or []
    if not isinstance(queries, list):
        raise ValueError("Queries must be an array.")
    if not isinstance(sections, list):
        raise ValueError("Sections must be an array.")

    plan = {
        "title": title,
        "audience": str(payload.get("audience", "")).strip() or "General readers",
        "tone": str(payload.get("tone", "")).strip() or newsletter_agent.DEFAULT_WRITING_STYLE,
        "queries": [str(item).strip() for item in queries if str(item).strip()],
        "sections": [str(item).strip() for item in sections if str(item).strip()],
    }
    if not plan["sections"]:
        plan["sections"] = ["What happened", "Why it matters", "What to watch next"]

    run_id = newsletter_agent.save_run(
        plan,
        brief,
        depth,
        explanation_style,
        style_instructions,
    )

    for source in payload.get("sources") or []:
        if not source.get("url"):
            continue
        newsletter_agent.save_source(
            run_id,
            {
                "query": str(source.get("query", "")).strip(),
                "rank_index": int(source.get("rank_index", 0) or 0),
                "title": str(source.get("title", "")).strip() or "Untitled source",
                "url": str(source.get("url", "")).strip(),
                "snippet": str(source.get("snippet", "")).strip(),
                "article_text": str(source.get("article_text", "")).strip(),
                "source_summary": str(source.get("source_summary", "")).strip(),
                "relevance_score": float(source.get("relevance_score", 0) or 0),
            },
        )

    output_files = newsletter_agent.write_newsletter_files(str(OUTPUT_ROOT), title, markdown)
    newsletter_agent.update_run_output_path(run_id, output_files["html_path"])
    return fetch_run_by_id(run_id)


def create_generation_job(payload):
    with JOBS_LOCK:
        active_job = find_active_job_locked()
        if active_job is not None:
            raise RuntimeError("A Chronicle issue is already generating on this device.")

        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "status": "queued",
            "message": "Queued on this device",
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "params": payload,
            "result": None,
            "error": None,
        }
        JOBS[job_id] = job

    worker = threading.Thread(target=run_generation_job, args=(job_id, payload), daemon=True)
    worker.start()
    return snapshot_job(job_id)


def run_generation_job(job_id, payload):
    try:
        update_job(job_id, status="running", message="Warming the local Chronicle runtime")
        newsletter_agent.initialize_model_runtime(payload.get("device_class"))

        settings = newsletter_agent.build_research_settings(
            payload["depth"],
            payload["queries"],
            payload["results_per_query"],
        )

        update_job(job_id, status="running", message="Researching and drafting on this device")
        result = newsletter_agent.run_newsletter_pipeline(
            brief=payload["brief"],
            days=payload["days"],
            depth=payload["depth"],
            explanation_style=payload["explanation_style"],
            custom_style_instructions=payload["style_instructions"],
            settings=settings,
            output_dir=str(OUTPUT_ROOT),
        )

        run_record = fetch_run_by_id(result["run_id"])
        update_job(
            job_id,
            status="completed",
            message="Issue ready",
            result=run_record,
            error=None,
        )
    except SystemExit as exc:
        update_job(
            job_id,
            status="failed",
            message=str(exc),
            error={
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            message=str(exc),
            error={
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def update_job(job_id, **changes):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        job.update(changes)
        job["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        return dict(job)


def get_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def snapshot_job(job_id):
    return get_job(job_id)


def find_active_job():
    with JOBS_LOCK:
        job = find_active_job_locked()
        return dict(job) if job else None


def find_active_job_locked():
    for job in JOBS.values():
        if job["status"] in {"queued", "running"}:
            return job
    return None


def count_runs():
    initialize_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM newsletter_runs")
    total = cursor.fetchone()[0]
    conn.close()
    return total


def count_issues():
    if not OUTPUT_ROOT.exists():
        return 0
    return len(list(OUTPUT_ROOT.glob("*.html")))


def fetch_runs(limit=24):
    initialize_database()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, brief, depth, explanation_style, title, created_at, output_path
        FROM newsletter_runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [serialize_run_row(row) for row in rows]


def fetch_run_by_id(run_id):
    initialize_database()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, brief, depth, explanation_style, title, created_at, output_path
        FROM newsletter_runs
        WHERE id = ?
        """,
        (run_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return serialize_run_row(row) if row else None


def serialize_run_row(row):
    if row is None:
        return None

    output_paths = resolve_output_paths(row["output_path"])
    return {
        "id": row["id"],
        "title": row["title"] or "Untitled issue",
        "brief": row["brief"] or "",
        "depth": row["depth"] or "",
        "explanation_style": row["explanation_style"] or "",
        "created_at": row["created_at"],
        "output_path": output_paths["preferred_path"],
        "html_path": output_paths["html_path"],
        "markdown_path": output_paths["markdown_path"],
        "preview_url": build_newsletter_url(output_paths["html_path"]),
        "html_url": build_newsletter_url(output_paths["html_path"]),
        "markdown_url": build_newsletter_url(output_paths["markdown_path"]),
    }


def resolve_output_paths(output_path):
    preferred_path = resolve_project_path(output_path) if output_path else None
    html_path = None
    markdown_path = None

    if preferred_path is not None:
        suffix = preferred_path.suffix.lower()
        if suffix == ".html":
            html_path = preferred_path
            markdown_candidate = preferred_path.with_suffix(".md")
            if markdown_candidate.exists():
                markdown_path = markdown_candidate
        elif suffix == ".md":
            markdown_path = preferred_path
            html_candidate = preferred_path.with_suffix(".html")
            if html_candidate.exists():
                html_path = html_candidate
                preferred_path = html_candidate
        else:
            html_candidate = preferred_path.with_suffix(".html")
            markdown_candidate = preferred_path.with_suffix(".md")
            if html_candidate.exists():
                html_path = html_candidate
                preferred_path = html_candidate
            if markdown_candidate.exists():
                markdown_path = markdown_candidate

    return {
        "preferred_path": str(preferred_path) if preferred_path else "",
        "html_path": str(html_path) if html_path and html_path.exists() else "",
        "markdown_path": str(markdown_path) if markdown_path and markdown_path.exists() else "",
    }


def resolve_project_path(path_value):
    if not path_value:
        return None
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def build_newsletter_url(path_value):
    absolute_path = resolve_project_path(path_value)
    if absolute_path is None or not absolute_path.exists():
        return ""
    output_root = OUTPUT_ROOT.resolve()
    if output_root not in absolute_path.parents and absolute_path != output_root:
        return ""
    return "/newsletters/" + absolute_path.name


if __name__ == "__main__":
    main()

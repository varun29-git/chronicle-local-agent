import argparse
import html
import json
import os
import platform
import re
import sqlite3
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

from newsletter_schema import DB_PATH, initialize_database

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_ROOT = os.environ.get(
    "NEWSLETTER_AGENT_MODEL_ROOT",
    os.path.join(PROJECT_ROOT, "models"),
)
DEFAULT_MLX_MODEL_REPO = "mlx-community/gemma-3-4b-it-4bit"
DEFAULT_TRANSFORMERS_MODEL_REPO = "google/gemma-2-2b-it"
DEFAULT_LOCAL_MLX_MODEL_PATH = os.path.join(
    DEFAULT_MODEL_ROOT,
    "mlx-community",
    "gemma-3-4b-it-4bit",
)
DEFAULT_LOCAL_TRANSFORMERS_MODEL_PATH = os.path.join(
    DEFAULT_MODEL_ROOT,
    "google",
    "gemma-2-2b-it",
)
DEFAULT_MLX_MODEL_PATH = DEFAULT_LOCAL_MLX_MODEL_PATH
DEFAULT_TRANSFORMERS_MODEL_PATH = os.environ.get(
    "NEWSLETTER_AGENT_MODEL_TRANSFORMERS",
    os.environ.get("NEWSLETTER_AGENT_MODEL", DEFAULT_LOCAL_TRANSFORMERS_MODEL_PATH),
)
DEFAULT_MODEL_PATH = os.environ.get("NEWSLETTER_AGENT_MODEL", DEFAULT_MLX_MODEL_PATH)
DEFAULT_BROWSER_MODEL_ID = os.environ.get("NEWSLETTER_AGENT_BROWSER_MODEL_ID", "").strip()
DEFAULT_BROWSER_MODEL_CANDIDATES = tuple(
    candidate
    for candidate in (
        DEFAULT_BROWSER_MODEL_ID,
        "onnx-community/gemma-3n-E4B-it-ONNX",
        "gemma-3n-E4B-it-ONNX",
        "onnx-community/gemma-3n-E2B-it-ONNX",
        "gemma-3n-E2B-it-ONNX",
        "gemma-3-it",
    )
    if candidate
)
DEFAULT_DAYS = 7
DEFAULT_QUERIES = 4
DEFAULT_RESULTS_PER_QUERY = 3
MAX_ARTICLE_CHARS = 8000
REQUEST_TIMEOUT_SECONDS = 5
DEFAULT_RAM_RESERVE_GB = float(os.environ.get("NEWSLETTER_AGENT_RAM_RESERVE_GB", "4"))
MAX_NEWSLETTER_RUNTIME_SECONDS = int(
    float(os.environ.get("NEWSLETTER_AGENT_MAX_RUNTIME_SECONDS", "300"))
)
DEFAULT_WRITING_STYLE = "Sharp, analytical, and premium. Write like a high-end newsletter editor, not a corporate content bot."
SEARCH_NOISE_WORDS = {
    "a",
    "about",
    "after",
    "analysis",
    "and",
    "around",
    "before",
    "change",
    "changed",
    "commentary",
    "day",
    "days",
    "development",
    "developments",
    "expert",
    "for",
    "from",
    "how",
    "in",
    "into",
    "key",
    "last",
    "latest",
    "month",
    "months",
    "new",
    "news",
    "of",
    "on",
    "or",
    "outlook",
    "recent",
    "show",
    "tell",
    "that",
    "the",
    "this",
    "today",
    "update",
    "updates",
    "week",
    "what",
    "with",
    "why",
    "yesterday",
}
SLICE_MODEL_PATHS = {
    0.125: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_12_5", DEFAULT_MODEL_PATH),
    0.25: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_25", DEFAULT_MODEL_PATH),
    0.50: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_50", DEFAULT_MODEL_PATH),
    0.75: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_75", DEFAULT_MODEL_PATH),
    1.00: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_100", DEFAULT_MODEL_PATH),
}
EXPLANATION_STYLE_GUIDANCE = {
    "concise": "Explain ideas briefly and cleanly. Prioritize compression, signal, and fast readability.",
    "feynman": "Explain ideas simply and clearly, as if teaching an intelligent beginner. Break down jargon into plain language without sounding childish.",
    "soc": "Explain in a Socratic style. Lead the reader through the idea by posing and answering the right questions step by step, while staying concise and clear.",
}
DEPTH_PRESETS = {
    "low": {
        "query_limit": 2,
        "results_per_query": 2,
        "article_chars": 0,
        "summary_tokens": 100,
        "newsletter_tokens": 520,
        "research_budget_seconds": 12,
    },
    "medium": {
        "query_limit": 4,
        "results_per_query": 3,
        "article_chars": 0,
        "summary_tokens": 120,
        "newsletter_tokens": 620,
        "research_budget_seconds": 20,
    },
    "high": {
        "query_limit": 5,
        "results_per_query": 3,
        "article_chars": 0,
        "summary_tokens": 160,
        "newsletter_tokens": 760,
        "research_budget_seconds": 28,
    },
}
MODEL_PROFILES = {
    "constrained": {
        "model_path": os.environ.get("NEWSLETTER_AGENT_MODEL_LOW", DEFAULT_MODEL_PATH),
        "draft_model_path": os.environ.get("NEWSLETTER_AGENT_DRAFT_MODEL_LOW", ""),
        "lazy": True,
        "num_draft_tokens": 0,
    },
    "balanced": {
        "model_path": os.environ.get("NEWSLETTER_AGENT_MODEL_MEDIUM", DEFAULT_MODEL_PATH),
        "draft_model_path": os.environ.get("NEWSLETTER_AGENT_DRAFT_MODEL_MEDIUM", ""),
        "lazy": False,
        "num_draft_tokens": 0,
    },
    "expanded": {
        "model_path": os.environ.get("NEWSLETTER_AGENT_MODEL_HIGH", DEFAULT_MODEL_PATH),
        "draft_model_path": os.environ.get("NEWSLETTER_AGENT_DRAFT_MODEL_HIGH", ""),
        "lazy": False,
        "num_draft_tokens": 4,
    },
}
INSECURE_SSL_CONTEXT = ssl._create_unverified_context()
initialize_database()

MODEL = None
TOKENIZER = None
MODEL_RUNTIME = None
DEVICE_CLASS_OVERRIDE = os.environ.get("NEWSLETTER_AGENT_DEVICE_CLASS", "").strip().lower() or None


def main():
    parser = argparse.ArgumentParser(description="Local newsletter research agent")
    parser.add_argument("--brief", help="Newsletter brief or topic")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--depth", choices=("low", "medium", "high"))
    parser.add_argument("--device-class", choices=("macbook", "midrange_laptop", "gaming_laptop", "midrange_phone", "flagship_phone"))
    parser.add_argument("--explanation-style", choices=("concise", "feynman", "soc", "custom"))
    parser.add_argument("--style-instructions")
    parser.add_argument("--queries", type=int)
    parser.add_argument("--results-per-query", type=int)
    parser.add_argument("--output-dir", default="output/newsletters")
    args = parser.parse_args()

    brief = args.brief or input("Newsletter brief: ").strip()
    if not brief:
        raise ValueError("A newsletter brief is required")

    initialize_model_runtime(args.device_class)

    depth = args.depth or prompt_for_depth()
    explanation_style, custom_style_instructions = resolve_explanation_style(
        args.explanation_style,
        args.style_instructions,
    )
    settings = build_research_settings(depth, args.queries, args.results_per_query)

    run_newsletter_pipeline(
        brief=brief,
        days=args.days,
        depth=depth,
        explanation_style=explanation_style,
        custom_style_instructions=custom_style_instructions,
        settings=settings,
        output_dir=args.output_dir,
    )


def initialize_model_runtime(device_class_override=None):
    global MODEL
    global TOKENIZER
    global MODEL_RUNTIME
    global DEVICE_CLASS_OVERRIDE

    if MODEL_RUNTIME is not None:
        return MODEL_RUNTIME

    if device_class_override:
        DEVICE_CLASS_OVERRIDE = device_class_override

    system_info = detect_system_info()
    runtime_profile = choose_model_profile(system_info, DEVICE_CLASS_OVERRIDE)

    print(f"Using model slice: {runtime_profile['slice_label']}")
    print(f"Using runtime backend: {runtime_profile['runtime_backend']}")

    if runtime_profile["runtime_backend"] == "mlx":
        backend_runtime = initialize_mlx_runtime(system_info, runtime_profile)
    elif runtime_profile["runtime_backend"] == "transformers":
        backend_runtime = initialize_transformers_runtime(runtime_profile)
    else:
        raise SystemExit(f"Unsupported runtime backend: {runtime_profile['runtime_backend']}")

    MODEL = backend_runtime["model"]
    TOKENIZER = backend_runtime["tokenizer"]
    MODEL_RUNTIME = {
        "system_info": system_info,
        "device_class": runtime_profile["device_class"],
        "profile_name": runtime_profile["profile_name"],
        "runtime_backend": runtime_profile["runtime_backend"],
        "slice_ratio": runtime_profile["slice_ratio"],
        "model_path": runtime_profile["model_path"],
        "draft_model": backend_runtime.get("draft_model"),
        "num_draft_tokens": backend_runtime.get("num_draft_tokens", 0),
        "device": backend_runtime.get("device", "cpu"),
        "generate_fn": backend_runtime.get("generate_fn"),
    }

    print("Newsletter agent ready.")
    return MODEL_RUNTIME


def initialize_mlx_runtime(system_info, runtime_profile):
    if not system_info["metal_supported"]:
        raise SystemExit("No Metal-capable Apple device detected. The MLX runtime requires Metal support.")

    try:
        from mlx_lm import generate as mlx_generate, load as mlx_load
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: mlx_lm.\n"
            "Run this script with the project virtual environment:\n"
            "  gemma-env/bin/python newsletter_agent.py --brief \"Your newsletter brief\"\n"
            "Or activate it first:\n"
            "  source gemma-env/bin/activate"
        ) from exc

    print("Loading newsletter brain into RAM")
    model_path = ensure_model_path_ready(runtime_profile["model_path"], "mlx")
    model, tokenizer = mlx_load(
        model_path,
        lazy=runtime_profile["lazy"],
    )

    draft_model = None
    if runtime_profile["draft_model_path"]:
        try:
            draft_model_path = normalize_model_reference(runtime_profile["draft_model_path"])
            draft_model, _ = mlx_load(draft_model_path, lazy=True)
            print(f"Loaded draft model: {draft_model_path}")
        except Exception as exc:
            print(f"Draft model load failed, continuing without it: {exc}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "draft_model": draft_model,
        "num_draft_tokens": runtime_profile["num_draft_tokens"],
        "device": "metal",
        "generate_fn": mlx_generate,
    }


def initialize_transformers_runtime(runtime_profile):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency for portable runtime: transformers and torch.\n"
            "Install them for your platform, then rerun the script.\n"
            "You can also set NEWSLETTER_AGENT_MODEL_TRANSFORMERS to a local or hub model path."
        ) from exc

    device = choose_transformers_device(torch)
    dtype = choose_transformers_dtype(torch, device)
    print("Loading newsletter brain into RAM")

    model_path = ensure_model_path_ready(runtime_profile["model_path"], "transformers")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )

    model.eval()
    if device != "cpu":
        model = model.to(device)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return {
        "model": model,
        "tokenizer": tokenizer,
        "draft_model": None,
        "num_draft_tokens": 0,
        "device": device,
    }


def choose_transformers_device(torch):
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return "mps"

    return "cpu"


def choose_transformers_dtype(torch, device):
    if device == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def detect_system_info():
    system_name = platform.system()
    profiler_data = run_system_profiler(system_name)
    hardware = profiler_data.get("SPHardwareDataType", [{}])
    displays = profiler_data.get("SPDisplaysDataType", [{}])
    hardware_info = hardware[0] if hardware else {}
    display_info = displays[0] if displays else {}

    gpu_info = detect_gpu_info(system_name, display_info)
    memory_total_gb = detect_total_memory_gb(system_name, hardware_info)
    memory_available_gb = detect_available_memory_gb(system_name)
    is_android = bool(os.environ.get("ANDROID_ROOT") or os.environ.get("TERMUX_VERSION"))
    is_ios = sys.platform == "ios"
    chip = hardware_info.get("chip_type", "")
    machine = platform.machine()
    is_apple_silicon = system_name == "Darwin" and (
        machine == "arm64" or str(chip).startswith("Apple ")
    )
    metal_supported = (
        display_info.get("spdisplays_metal") == "spdisplays_supported" or is_apple_silicon
    )

    return {
        "system_name": system_name,
        "platform": platform.platform(),
        "machine": machine,
        "chip": chip,
        "hardware_model": hardware_info.get("machine_model", ""),
        "gpu_model": gpu_info["model"],
        "gpu_vendor": gpu_info["vendor"],
        "gpu_memory_gb": gpu_info["memory_gb"],
        "dedicated_gpu": gpu_info["dedicated"],
        "gpu_cores": display_info.get("sppci_cores", ""),
        "metal_supported": metal_supported,
        "is_apple_silicon": is_apple_silicon,
        "is_android": is_android,
        "is_ios": is_ios,
        "is_mobile": is_android or is_ios,
        "memory_total_gb": memory_total_gb,
        "memory_available_gb": memory_available_gb,
    }


def run_system_profiler(system_name):
    if system_name != "Darwin":
        return {}

    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            check=True,
        )
        return json.loads(result.stdout)
    except Exception:
        return {}


def detect_gpu_info(system_name, display_info):
    if system_name == "Darwin":
        return {
            "model": str(display_info.get("sppci_model", "")),
            "vendor": "apple",
            "memory_gb": 0.0,
            "dedicated": False,
        }

    nvidia_info = detect_nvidia_gpu()
    if nvidia_info:
        return nvidia_info

    pci_hint = detect_pci_gpu_hint()
    if pci_hint:
        return pci_hint

    return {
        "model": "",
        "vendor": "",
        "memory_gb": 0.0,
        "dedicated": False,
    }


def detect_nvidia_gpu():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            check=True,
        )
    except Exception:
        return None

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not line:
        return None

    parts = [part.strip() for part in line.split(",")]
    memory_gb = (float(parts[1]) / 1024.0) if len(parts) > 1 else 0.0
    return {
        "model": parts[0],
        "vendor": "nvidia",
        "memory_gb": memory_gb,
        "dedicated": True,
    }


def detect_pci_gpu_hint():
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            check=True,
        )
    except Exception:
        return None

    output = result.stdout.lower()
    keywords = ("nvidia", "geforce", "rtx", "gtx", "radeon", "amd")
    if not any(keyword in output for keyword in keywords):
        return None

    return {
        "model": "discrete-gpu",
        "vendor": "nvidia/amd",
        "memory_gb": 0.0,
        "dedicated": True,
    }


def detect_total_memory_gb(system_name, hardware_info):
    memory_text = str(hardware_info.get("physical_memory", "")).strip()
    if memory_text:
        match = re.match(r"([0-9.]+)\s*GB", memory_text, re.IGNORECASE)
        if match:
            return float(match.group(1))

    if system_name in {"Linux", "Android"} or os.environ.get("ANDROID_ROOT"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        kilobytes = int(line.split()[1])
                        return kilobytes / (1024**2)
        except Exception:
            pass

    if system_name == "Windows":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            memory_status = MemoryStatus()
            memory_status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
            return memory_status.ullTotalPhys / (1024**3)
        except Exception:
            pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        return (page_size * total_pages) / (1024**3)
    except Exception:
        return 0.0


def detect_available_memory_gb(system_name):
    if system_name in {"Linux", "Android"} or os.environ.get("ANDROID_ROOT"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        kilobytes = int(line.split()[1])
                        return kilobytes / (1024**2)
        except Exception:
            pass

    if system_name == "Windows":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            memory_status = MemoryStatus()
            memory_status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
            return memory_status.ullAvailPhys / (1024**3)
        except Exception:
            pass

    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            check=True,
        )
    except Exception:
        return 0.0

    page_size_match = re.search(r"page size of (\d+) bytes", result.stdout)
    page_size = int(page_size_match.group(1)) if page_size_match else 4096
    page_counts = {}

    for line in result.stdout.splitlines():
        match = re.match(r"([^:]+):\s+([0-9]+)\.", line.strip())
        if match:
            page_counts[match.group(1)] = int(match.group(2))

    available_pages = (
        page_counts.get("Pages free", 0)
        + page_counts.get("Pages inactive", 0)
        + page_counts.get("Pages speculative", 0)
        + page_counts.get("Pages purgeable", 0)
    )
    return (available_pages * page_size) / (1024**3)


def choose_model_profile(system_info, device_class_override=None):
    device_class = detect_device_class(system_info, device_class_override)
    slice_ratio = choose_slice_ratio(device_class, system_info)
    runtime_backend = choose_runtime_backend(system_info)

    if slice_ratio <= 0.25:
        profile_name = "constrained"
    elif slice_ratio >= 0.75:
        profile_name = "expanded"
    else:
        profile_name = "balanced"

    profile = dict(MODEL_PROFILES[profile_name])
    profile["profile_name"] = profile_name
    profile["device_class"] = device_class
    profile["runtime_backend"] = runtime_backend
    profile["slice_ratio"] = slice_ratio
    profile["slice_label"] = format_slice_label(slice_ratio)
    profile["model_path"] = resolve_model_path_for_backend(
        choose_model_path_for_slice(slice_ratio, profile["model_path"]),
        runtime_backend,
    )
    if runtime_backend != "mlx":
        profile["draft_model_path"] = ""
        profile["num_draft_tokens"] = 0
    return profile


def detect_device_class(system_info, device_class_override=None):
    override = device_class_override or DEVICE_CLASS_OVERRIDE
    if override:
        return override

    total_gb = system_info["memory_total_gb"]
    if system_info["is_mobile"]:
        return "flagship_phone" if total_gb >= 8 else "midrange_phone"

    if system_info["system_name"] == "Darwin":
        return "macbook"

    if system_info["dedicated_gpu"] or system_info["gpu_memory_gb"] >= 6 or total_gb >= 24:
        return "gaming_laptop"

    return "midrange_laptop"


def choose_slice_ratio(device_class, system_info):
    available_gb = system_info.get("memory_available_gb", 0.0) or 0.0
    total_gb = system_info.get("memory_total_gb", 0.0) or 0.0
    reserve_gb = max(DEFAULT_RAM_RESERVE_GB, total_gb * 0.35, 1.0)
    headroom_gb = max(available_gb - reserve_gb, 0.0) if available_gb else 0.0

    if not available_gb:
        return 0.25
    if headroom_gb <= 2:
        return 0.125
    if headroom_gb <= 5:
        return 0.25
    if headroom_gb <= 9:
        return 0.50
    if headroom_gb <= 14:
        return 0.75
    return 1.00


def choose_runtime_backend(system_info):
    override = os.environ.get("NEWSLETTER_AGENT_BACKEND", "").strip().lower()
    if override in {"mlx", "transformers"}:
        return override

    if (
        system_info["system_name"] == "Darwin"
        and not system_info.get("is_ios")
        and (system_info["metal_supported"] or system_info.get("is_apple_silicon"))
    ):
        return "mlx"
    return "transformers"


def choose_model_path_for_slice(slice_ratio, default_path):
    rounded_ratio = round(slice_ratio, 3)
    return SLICE_MODEL_PATHS.get(rounded_ratio, default_path)


def resolve_model_path_for_backend(model_path, runtime_backend):
    normalized_model_path = normalize_model_reference(model_path)

    if runtime_backend == "mlx":
        return normalized_model_path

    if str(model_path).startswith("mlx-community/"):
        return normalize_model_reference(DEFAULT_TRANSFORMERS_MODEL_PATH)

    if normalized_model_path == normalize_model_reference(DEFAULT_LOCAL_MLX_MODEL_PATH):
        return normalize_model_reference(DEFAULT_TRANSFORMERS_MODEL_PATH)

    return normalized_model_path


def normalize_model_reference(model_path):
    model_path = str(model_path).strip()
    if not model_path:
        return model_path

    if is_local_model_reference(model_path):
        expanded_path = os.path.expanduser(model_path)
        if os.path.isabs(expanded_path):
            return os.path.normpath(expanded_path)
        return os.path.normpath(os.path.join(PROJECT_ROOT, expanded_path))

    return model_path


def is_local_model_reference(model_path):
    model_path = str(model_path).strip()
    if not model_path:
        return False

    expanded_path = os.path.expanduser(model_path)
    normalized_path = model_path.replace("\\", "/")
    if os.path.isabs(expanded_path):
        return True
    if normalized_path.startswith(("./", "../", "~/", "models/")):
        return True
    return os.path.exists(os.path.join(PROJECT_ROOT, expanded_path))


def ensure_model_path_ready(model_path, runtime_backend):
    normalized_model_path = normalize_model_reference(model_path)
    if not is_local_model_reference(normalized_model_path):
        return normalized_model_path

    if os.path.exists(normalized_model_path):
        return normalized_model_path

    raise SystemExit(build_missing_local_model_message(runtime_backend, normalized_model_path))


def build_missing_local_model_message(runtime_backend, model_path):
    if runtime_backend == "mlx":
        expected_default_path = DEFAULT_LOCAL_MLX_MODEL_PATH
        default_repo = DEFAULT_MLX_MODEL_REPO
        override_hint = "NEWSLETTER_AGENT_MODEL or NEWSLETTER_AGENT_MODEL_SLICE_*"
    else:
        expected_default_path = DEFAULT_LOCAL_TRANSFORMERS_MODEL_PATH
        default_repo = DEFAULT_TRANSFORMERS_MODEL_REPO
        override_hint = "NEWSLETTER_AGENT_MODEL_TRANSFORMERS or NEWSLETTER_AGENT_MODEL"

    return (
        "Local model files not found.\n"
        f"Expected model directory: {model_path}\n"
        "This project now defaults to local model directories so end users do not need "
        "Hugging Face tokens.\n"
        "To fix this:\n"
        f"  1. Place the model files at: {expected_default_path}\n"
        f"  2. Or point {override_hint} to a local model directory\n"
        f"  3. If you intentionally want Hugging Face downloads, set the override to a hub "
        f"model id such as {default_repo} after authenticating for gated access"
    )


def describe_browser_model():
    missing_descriptor = None

    for candidate in DEFAULT_BROWSER_MODEL_CANDIDATES:
        descriptor = build_browser_model_descriptor(candidate)
        if descriptor["ready"]:
            return descriptor
        if missing_descriptor is None:
            missing_descriptor = descriptor

    return missing_descriptor or {
        "ready": False,
        "model_id": "",
        "model_path": "",
        "model_type": "",
        "architecture": "",
        "supports_slicing": False,
        "max_slices": 1,
        "display_name": "No local browser model found",
    }


def build_browser_model_descriptor(model_reference):
    normalized_reference = str(model_reference or "").strip().replace("\\", "/").strip("/")
    model_path, model_id = resolve_browser_model_location(normalized_reference)
    config_path = os.path.join(model_path, "config.json") if model_path else ""
    config_data = {}

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config_data = json.load(handle)
        except Exception:
            config_data = {}

    model_type = str(config_data.get("model_type") or "").strip()
    architectures = config_data.get("architectures") or []
    architecture = str(architectures[0]).strip() if architectures else ""
    lower_hints = " ".join(
        part
        for part in (
            normalized_reference.lower(),
            model_type.lower(),
            architecture.lower(),
        )
        if part
    )
    supports_slicing = "gemma3n" in lower_hints or "gemma-3n" in lower_hints
    preferred_precision, dtype_map = detect_browser_model_precision(model_path)

    display_name, model_variant = derive_browser_display_name(normalized_reference, model_type, architecture)

    return {
        "ready": bool(model_path and os.path.isdir(model_path) and os.path.exists(config_path)),
        "model_id": model_id,
        "model_path": model_path,
        "model_type": model_type,
        "architecture": architecture,
        "model_variant": model_variant,
        "supports_slicing": supports_slicing,
        "max_slices": 8 if supports_slicing else 1,
        "preferred_precision": preferred_precision,
        "dtype_map": dtype_map,
        "display_name": display_name,
    }


def derive_browser_display_name(model_reference, model_type, architecture):
    lower_hints = " ".join(
        part.lower()
        for part in (model_reference, model_type, architecture)
        if part
    )

    if "gemma-3n-e4b" in lower_hints:
        return "Gemma 3n E4B adaptive", "E4B"
    if "gemma-3n-e2b" in lower_hints:
        return "Gemma 3n adaptive", "E2B"
    if "gemma-3n" in lower_hints:
        return "Gemma 3n adaptive", "3n"
    if "gemma-3" in lower_hints:
        return "Gemma 3", "3"

    fallback_name = architecture or model_type or model_reference or "Local browser model"
    return fallback_name, ""


def resolve_browser_model_location(model_reference):
    if not model_reference:
        return "", ""

    model_root = os.path.abspath(DEFAULT_MODEL_ROOT)
    expanded_reference = os.path.expanduser(model_reference)
    if os.path.isabs(expanded_reference):
        normalized_path = os.path.normpath(expanded_reference)
        try:
            model_id = os.path.relpath(normalized_path, model_root).replace("\\", "/")
        except ValueError:
            return normalized_path, ""
        if model_id.startswith("../"):
            return normalized_path, ""
        return normalized_path, model_id

    normalized_reference = expanded_reference.replace("\\", "/").strip("/")
    if normalized_reference.startswith("models/"):
        normalized_reference = normalized_reference.removeprefix("models/")

    model_path = os.path.normpath(os.path.join(model_root, normalized_reference))
    model_id = normalized_reference
    return model_path, model_id


def detect_browser_model_precision(model_path):
    if not model_path:
        return "", {}

    onnx_root = os.path.join(model_path, "onnx")
    if not os.path.isdir(onnx_root):
        return "", {}

    decoder_precision_pairs = (
        ("q4", "decoder_model_merged_q4.onnx", "embed_tokens_q4.onnx"),
        ("q4f16", "decoder_model_merged_q4f16.onnx", "embed_tokens_q4f16.onnx"),
        ("quantized", "decoder_model_merged_quantized.onnx", "embed_tokens_quantized.onnx"),
        ("uint8", "decoder_model_merged_uint8.onnx", "embed_tokens_uint8.onnx"),
    )
    dtype_map = {}
    preferred_precision = ""

    for precision, decoder_name, embedding_name in decoder_precision_pairs:
        if os.path.exists(os.path.join(onnx_root, decoder_name)) and os.path.exists(
            os.path.join(onnx_root, embedding_name)
        ):
            preferred_precision = precision
            dtype_map["decoder_model_merged"] = precision
            dtype_map["embed_tokens"] = precision
            break

    auxiliary_component_precisions = {
        "audio_encoder": ("q4f16", "q4", "int8", "uint8", "quantized", "fp16", "fp32"),
        "vision_encoder": ("uint8", "int8", "quantized", "fp16", "fp32"),
    }
    for component_name, precision_order in auxiliary_component_precisions.items():
        selected_precision = detect_onnx_component_precision(
            onnx_root,
            component_name,
            precision_order,
        )
        if selected_precision:
            dtype_map[component_name] = selected_precision

    return preferred_precision, dtype_map


def detect_onnx_component_precision(onnx_root, component_name, precision_order):
    for precision in precision_order:
        if precision == "fp32":
            filename = f"{component_name}.onnx"
        else:
            filename = f"{component_name}_{precision}.onnx"
        if os.path.exists(os.path.join(onnx_root, filename)):
            return precision
    return ""


def format_slice_label(slice_ratio):
    percentage = slice_ratio * 100
    if percentage.is_integer():
        return f"{int(percentage)}%"
    return f"{percentage:.1f}%"


def run_newsletter_pipeline(
    brief,
    days,
    depth,
    explanation_style,
    custom_style_instructions,
    settings,
    output_dir,
):
    plan = build_research_plan(brief, days, settings["query_limit"], depth)
    run_id = save_run(plan, brief, depth, explanation_style, custom_style_instructions)
    market_snapshot = fetch_market_snapshot(brief)
    research_budget_seconds = min(
        int(settings.get("research_budget_seconds", 75)),
        max(30, MAX_NEWSLETTER_RUNTIME_SECONDS - 120),
    )
    research_deadline = time.monotonic() + research_budget_seconds

    print("\nPlanning complete.")
    print(f"Title: {plan['title']}")
    print(f"Research depth: {depth}")
    print(f"Explanation style: {explanation_style}")
    print(f"Research budget: {research_budget_seconds}s")
    if market_snapshot:
        print(f"Structured market data collected for {len(market_snapshot)} assets.")

    collected_sources = []
    research_budget_hit = False
    for query in plan["queries"]:
        if time.monotonic() >= research_deadline:
            research_budget_hit = True
            print("\nResearch budget reached. Drafting with the sources already collected.")
            break

        print(f"\nSearching: {query}")
        results = search_web(query, settings["results_per_query"], deadline=research_deadline)
        if not results:
            print("  No search results collected for this query.")
            continue
        for rank, result in enumerate(results, start=1):
            if time.monotonic() >= research_deadline:
                research_budget_hit = True
                print("  Research budget reached while processing results.")
                break

            article_text = fetch_article_text(result["url"], settings["article_chars"])
            source_text = build_source_text(result, article_text)
            if not source_text:
                print("  Skipped source: no article text or search snippet available")
                continue

            try:
                source_digest = summarize_source(
                    brief,
                    plan,
                    result,
                    source_text,
                    settings["summary_tokens"],
                )
            except Exception as exc:
                print(f"  Skipped source: summary failed: {exc}")
                continue

            source_record = {
                "query": query,
                "rank_index": rank,
                "title": result["title"],
                "url": result["url"],
                "snippet": result["snippet"],
                "article_text": article_text,
                "source_summary": source_digest["summary"],
                "relevance_score": float(source_digest["relevance"]),
            }
            save_source(run_id, source_record)
            collected_sources.append(
                {
                    "query": query,
                    "rank_index": rank,
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": result["snippet"],
                    "source_summary": source_digest["summary"],
                    "relevance_score": float(source_digest["relevance"]),
                }
            )
            del article_text
            del source_text
            del source_record
            print(f"  Saved source: {result['title']}")

        if research_budget_hit:
            break

    if not collected_sources:
        print(
            "\nNo usable sources were collected from the web search step. "
            "Falling back to a source-free draft."
        )
        newsletter_markdown = compose_fallback_newsletter(
            brief,
            plan,
            days,
            market_snapshot,
            depth,
            explanation_style,
            custom_style_instructions,
            settings["newsletter_tokens"],
        )
    else:
        newsletter_markdown = compose_newsletter(
            brief,
            plan,
            collected_sources,
            days,
            market_snapshot,
            depth,
            explanation_style,
            custom_style_instructions,
            settings["newsletter_tokens"],
        )
    output_files = write_newsletter_files(output_dir, plan["title"], newsletter_markdown)
    update_run_output_path(run_id, output_files["html_path"])

    print("\nNewsletter generated.")
    print(f"Saved editable HTML to: {output_files['html_path']}")
    print(f"Saved markdown source to: {output_files['markdown_path']}")
    return {
        "run_id": run_id,
        "title": plan["title"],
        "output_files": output_files,
    }


def build_research_settings(depth, query_limit_override, results_per_query_override):
    settings = dict(DEPTH_PRESETS[depth])
    if query_limit_override is not None:
        settings["query_limit"] = query_limit_override
    if results_per_query_override is not None:
        settings["results_per_query"] = results_per_query_override
    return settings


def prompt_for_depth():
    while True:
        value = input("Research depth (low/medium/high) [medium]: ").strip().lower()
        if not value:
            return "medium"
        if value in DEPTH_PRESETS:
            return value
        print("Please choose low, medium, or high.")


def resolve_explanation_style(explanation_style, style_instructions):
    style = explanation_style or prompt_for_explanation_style()
    custom_instructions = ""

    if style == "custom":
        custom_instructions = (style_instructions or input("Custom style instructions: ").strip())
        if not custom_instructions:
            raise ValueError("Custom style instructions are required when explanation style is custom")
    elif style_instructions:
        custom_instructions = style_instructions.strip()

    return style, custom_instructions


def prompt_for_explanation_style():
    while True:
        value = input("Explanation style (concise/feynman/soc/custom) [concise]: ").strip().lower()
        if not value:
            return "concise"
        if value in {"concise", "feynman", "soc", "custom"}:
            return value
        print("Please choose concise, feynman, soc, or custom.")


def build_explanation_guidance(explanation_style, custom_style_instructions):
    if explanation_style == "custom":
        return custom_style_instructions.strip()
    return EXPLANATION_STYLE_GUIDANCE[explanation_style]


def build_research_plan(brief, days, query_limit, depth):
    prompt = f"""<start_of_turn>user
You are the planning brain for a newsletter agent.

Task:
- understand the brief
- decide what web searches are needed
- decide what sections the newsletter should have
- produce a plan for a newsletter covering the last {days} days
- adjust the breadth of research for {depth} depth

Brief:
"{brief}"

Return ONLY JSON in this format:
{{"title":"str","audience":"str","tone":"str","queries":["q1","q2"],"sections":["s1","s2","s3"]}}

Rules:
- choose exactly {query_limit} focused search queries
- make the sections useful for a real newsletter
- keep title concise
- for low depth, keep the scope tight
- for medium depth, balance breadth and efficiency
- for high depth, cover the topic more comprehensively
- do not include any text outside JSON
<end_of_turn>
<start_of_turn>model
{{"title":"""

    try:
        plan = generate_json_from_prompt(
            prompt,
            '{"title":',
            220,
            schema_hint='{"title":"str","audience":"str","tone":"str","queries":["q1","q2"],"sections":["s1","s2","s3"]}',
        )
    except Exception as exc:
        print(f"Planning fallback activated: {exc}")
        return build_fallback_research_plan(brief, days, query_limit, depth)

    queries = [str(item).strip() for item in plan.get("queries", []) if str(item).strip()]
    sections = [str(item).strip() for item in plan.get("sections", []) if str(item).strip()]
    if len(queries) < query_limit:
        print("Planning fallback activated: model did not provide enough queries")
        return build_fallback_research_plan(brief, days, query_limit, depth)
    if not sections:
        print("Planning fallback activated: model did not provide newsletter sections")
        return build_fallback_research_plan(brief, days, query_limit, depth)

    return {
        "title": str(plan.get("title", "")).strip() or "Weekly Newsletter",
        "audience": str(plan.get("audience", "")).strip() or "General readers",
        "tone": str(plan.get("tone", "")).strip() or DEFAULT_WRITING_STYLE,
        "queries": queries[:query_limit],
        "sections": sections,
    }


def build_fallback_research_plan(brief, days, query_limit, depth):
    normalized_brief = clean_text(brief)
    focus_phrase = derive_search_focus_phrase(normalized_brief)
    keyword_variants = build_search_query_variants(focus_phrase or normalized_brief)
    query_candidates = [
        normalized_brief,
        focus_phrase,
        f"{focus_phrase} news" if focus_phrase else "",
        f"{focus_phrase} latest" if focus_phrase else "",
        f"{focus_phrase} this week" if focus_phrase and days <= 10 else "",
        f"{focus_phrase} last {days} days" if focus_phrase and days > 10 else "",
        f"{focus_phrase} analysis" if focus_phrase else "",
        *keyword_variants,
    ]

    queries = []
    seen = set()
    for query in query_candidates:
        cleaned_query = clean_text(query)
        if not cleaned_query:
            continue
        lowered = cleaned_query.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        queries.append(cleaned_query)
        if len(queries) >= query_limit:
            break

    sections = [
        "What happened",
        "Why it matters",
        "Key developments",
        "What to watch next",
    ]

    return {
        "title": generate_fallback_title(normalized_brief),
        "audience": "General readers who want a useful weekly briefing",
        "tone": DEFAULT_WRITING_STYLE,
        "queries": queries,
        "sections": sections[: max(3, min(len(sections), 4 if depth == "high" else 3))],
    }


def generate_fallback_title(brief):
    focus_phrase = derive_search_focus_phrase(brief)
    words = [word for word in re.split(r"\s+", focus_phrase or brief) if word]
    compact = " ".join(words[:4]).strip()
    if not compact:
        return "Weekly Newsletter"
    return compact.title()


def derive_search_focus_phrase(text):
    keywords = extract_search_keywords(text)
    if keywords:
        return " ".join(keywords[:4])
    simplified = clean_text(
        re.sub(
            r"\b(what changed in|what changed|tell me about|show me|latest developments|key news|analysis and outlook)\b",
            " ",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )
    return simplified


def extract_search_keywords(text):
    keywords = []
    seen = set()
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", str(text or "").lower()):
        normalized = word.strip(".'-")
        if len(normalized) < 3 or normalized.isdigit():
            continue
        if normalized in SEARCH_NOISE_WORDS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        keywords.append(normalized)
    return keywords


def simplify_search_query(query):
    simplified = re.sub(
        r"\b(latest developments?|key news|analysis and outlook|expert commentary|what changed this week|what changed in|what changed|latest|this week|last \d+ days?)\b",
        " ",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    return clean_text(simplified)


def build_search_query_variants(query):
    original_query = clean_text(query)
    simplified_query = simplify_search_query(original_query)
    keywords = extract_search_keywords(simplified_query or original_query)
    focus_phrase = " ".join(keywords[:4]) if keywords else simplified_query or original_query
    last_keyword = keywords[-1] if keywords else ""

    variants = []
    seen = set()

    def add(candidate):
        text = clean_text(candidate)
        if not text:
            return
        lowered = text.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        variants.append(text)

    add(original_query)
    add(simplified_query)
    if focus_phrase:
        add(f"{focus_phrase} news")
        add(f"{focus_phrase} latest")
        add(f"{focus_phrase} analysis")
        add(f"{focus_phrase} this week")
    if last_keyword:
        add(f"{last_keyword} latest news")

    return variants[:5]


def fetch_market_snapshot(brief):
    if not looks_like_crypto_brief(brief):
        return []

    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        "?vs_currency=usd&order=market_cap_desc&per_page=5&page=1"
        "&sparkline=false&price_change_percentage=7d"
    )

    try:
        payload = fetch_url(url)
        data = json.loads(payload)
    except Exception as exc:
        print(f"Structured market data fetch failed: {exc}")
        return []

    snapshot = []
    for item in data[:5]:
        snapshot.append(
            {
                "name": item.get("name", ""),
                "symbol": str(item.get("symbol", "")).upper(),
                "price_usd": item.get("current_price"),
                "market_cap_rank": item.get("market_cap_rank"),
                "change_24h_pct": item.get("price_change_percentage_24h"),
                "change_7d_pct": item.get("price_change_percentage_7d_in_currency"),
            }
        )

    return snapshot


def looks_like_crypto_brief(brief):
    lowered = brief.lower()
    keywords = (
        "crypto",
        "bitcoin",
        "ethereum",
        "blockchain",
        "defi",
        "token",
        "altcoin",
        "web3",
    )
    return any(keyword in lowered for keyword in keywords)


def search_web(query, max_results, deadline=None):
    if deadline and time.monotonic() >= deadline:
        print("  Search budget reached before Google search started.")
        return []

    query_variant = build_search_query_variants(query)[0]
    try:
        results = search_google_news_rss(query_variant, max_results)
    except Exception as exc:
        print(f"  Google search failed: {exc}")
        return []

    print(f"  Google search: {len(results)} result(s)")
    return prioritize_sources(results)[:max_results]


def search_google_news_rss(query, max_results):
    rss_query = urllib.parse.quote(f"{query} when:7d")
    url = (
        "https://news.google.com/rss/search?q="
        f"{rss_query}&hl=en-US&gl=US&ceid=US:en"
    )
    xml_text = fetch_url(url)

    results = []
    root = ET.fromstring(xml_text)
    for item in root.findall(".//item"):
        title = clean_text(item.findtext("title", ""))
        url = clean_text(item.findtext("link", ""))
        snippet = strip_tags(item.findtext("description", ""))
        if not title or not url:
            continue
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
            }
        )
        if len(results) >= max_results:
            break

    return results


def is_indirect_source_url(url):
    lowered = str(url or "").lower()
    return "news.google.com/" in lowered


def prioritize_sources(results):
    return sorted(
        results,
        key=lambda item: (
            1 if is_indirect_source_url(item.get("url", "")) else 0,
            -len(str(item.get("snippet", "") or "")),
        ),
    )


def search_duckduckgo_html(query, max_results):
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    html_text = fetch_url(url)

    results = []
    seen_urls = set()
    link_matches = re.finditer(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html_text,
        re.IGNORECASE | re.DOTALL,
    )
    snippet_matches = iter(
        re.findall(
            r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>|'
            r'<div[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>',
            html_text,
            re.IGNORECASE | re.DOTALL,
        )
    )

    for match in link_matches:
        href, raw_title = match.groups()
        resolved_url = normalize_result_url(html.unescape(href))
        if not resolved_url or resolved_url in seen_urls:
            continue

        title = clean_text(strip_tags(raw_title))
        if not title:
            continue

        raw_snippet = ""
        try:
            snippet_groups = next(snippet_matches)
            raw_snippet = next((group for group in snippet_groups if group), "")
        except StopIteration:
            raw_snippet = ""

        seen_urls.add(resolved_url)
        results.append(
            {
                "title": title,
                "url": resolved_url,
                "snippet": clean_text(strip_tags(html.unescape(raw_snippet))),
            }
        )
        if len(results) >= max_results:
            break

    return results


def search_duckduckgo_lite(query, max_results):
    url = "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(query)
    html_text = fetch_url(url)

    results = []
    seen_urls = set()
    for block in re.findall(r"<tr>.*?</tr>", html_text, re.IGNORECASE | re.DOTALL):
        link_match = re.search(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.IGNORECASE | re.DOTALL)
        if not link_match:
            continue

        href, raw_title = link_match.groups()
        resolved_url = normalize_result_url(html.unescape(href))
        if not resolved_url or resolved_url in seen_urls:
            continue

        snippet_match = re.search(
            r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
            block,
            re.IGNORECASE | re.DOTALL,
        )
        seen_urls.add(resolved_url)
        results.append(
            {
                "title": clean_text(strip_tags(raw_title)),
                "url": resolved_url,
                "snippet": clean_text(strip_tags(snippet_match.group(1))) if snippet_match else "",
            }
        )
        if len(results) >= max_results:
            break

    return results


def search_bing_html(query, max_results):
    url = "https://www.bing.com/search?q=" + urllib.parse.quote(query)
    html_text = fetch_url(url)

    results = []
    seen_urls = set()
    for block in re.findall(r'<li[^>]*class="b_algo"[^>]*>.*?</li>', html_text, re.IGNORECASE | re.DOTALL):
        link_match = re.search(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.IGNORECASE | re.DOTALL)
        if not link_match:
            continue

        href, raw_title = link_match.groups()
        resolved_url = normalize_result_url(html.unescape(href))
        if not resolved_url or resolved_url in seen_urls:
            continue

        snippet_match = re.search(r"<p>(.*?)</p>", block, re.IGNORECASE | re.DOTALL)
        seen_urls.add(resolved_url)
        results.append(
            {
                "title": clean_text(strip_tags(raw_title)),
                "url": resolved_url,
                "snippet": clean_text(strip_tags(snippet_match.group(1))) if snippet_match else "",
            }
        )
        if len(results) >= max_results:
            break

    return results


def normalize_result_url(href):
    if href.startswith("//"):
        href = "https:" + href

    if "duckduckgo.com/l/" in href:
        parsed = urllib.parse.urlparse(href)
        params = urllib.parse.parse_qs(parsed.query)
        target = params.get("uddg", [""])[0]
        return urllib.parse.unquote(target)

    if href.startswith("http://") or href.startswith("https://"):
        return href

    return ""


def strip_tags(value):
    without_tags = re.sub(r"(?s)<[^>]+>", " ", str(value or ""))
    return html.unescape(clean_text(without_tags))


def fetch_url(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        },
    )
    return read_response(request)


def read_response(request):
    with urllib.request.urlopen(
        request,
        timeout=REQUEST_TIMEOUT_SECONDS,
        context=INSECURE_SSL_CONTEXT,
    ) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_article_text(url, max_article_chars):
    try:
        raw_html = fetch_url(url)
    except Exception as exc:
        print(f"  Article fetch failed for {url}: {exc}")
        return ""

    text = raw_html
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = clean_text(text)
    if looks_like_placeholder_article_text(text):
        return ""
    return text[:max_article_chars]


def build_source_text(result, article_text):
    if article_text:
        return article_text

    snippet = clean_text(result.get("snippet", ""))
    if not snippet:
        return ""

    return clean_text(
        f'Title: {result.get("title", "")}\n'
        f'URL: {result.get("url", "")}\n'
        f'Snippet: {snippet}'
    )


def looks_like_placeholder_article_text(text):
    normalized = clean_text(text).lower()
    if not normalized:
        return True
    if normalized in {"google news", "bing", "duckduckgo"}:
        return True
    return len(normalized) < 120


def summarize_source(brief, plan, result, article_text, summary_tokens):
    prompt = f"""<start_of_turn>user
You are summarizing one web source for a newsletter writer.

Newsletter brief:
"{brief}"

Planned audience:
"{plan['audience']}"

Planned sections:
{json.dumps(plan['sections'])}

Source title:
"{result['title']}"

Source URL:
"{result['url']}"

Source text:
\"\"\"
{article_text}
\"\"\"

Return ONLY JSON in this format:
{{"summary":"str","relevance":0.0,"key_points":["p1","p2","p3"]}}

Rules:
- keep summary factual and concise
- relevance must be between 0 and 1
- focus on information useful for the newsletter brief
- do not include any text outside JSON
<end_of_turn>
<start_of_turn>model
{{"summary":"""

    data = generate_json_from_prompt(
        prompt,
        '{"summary":',
        summary_tokens,
        schema_hint='{"summary":"str","relevance":0.0,"key_points":["p1","p2","p3"]}',
    )
    summary = str(data.get("summary", "")).strip()
    if not summary:
        raise ValueError("Empty source summary")

    return {
        "summary": summary,
        "relevance": float(data.get("relevance", 0) or 0),
        "key_points": data.get("key_points", []),
    }


def compose_newsletter(
    brief,
    plan,
    collected_sources,
    days,
    market_snapshot,
    depth,
    explanation_style,
    custom_style_instructions,
    newsletter_tokens,
):
    compact_sources = []
    for index, source in enumerate(collected_sources, start=1):
        compact_sources.append(
            {
                "id": index,
                "title": source["title"],
                "url": source["url"],
                "summary": source["source_summary"],
                "relevance": source["relevance_score"],
            }
        )

    market_data_block = json.dumps(market_snapshot, ensure_ascii=True)
    explanation_guidance = build_explanation_guidance(
        explanation_style,
        custom_style_instructions,
    )
    editorial_brief = build_editorial_brief(
        brief,
        plan,
        compact_sources,
        market_snapshot,
        depth,
    )

    prompt = f"""<start_of_turn>user
You are the writing brain for a newsletter agent.

Write a complete markdown newsletter using the structured market data, editorial brief, and source summaries provided.

Newsletter brief:
"{brief}"

Title:
"{plan['title']}"

Audience:
"{plan['audience']}"

Tone:
"{plan['tone']}"

House style:
"{DEFAULT_WRITING_STYLE}"

Explanation style:
{explanation_style}

Explanation guidance:
"{explanation_guidance}"

Coverage window:
Last {days} days

Research depth:
{depth}

Planned sections:
{json.dumps(plan['sections'])}

Editorial brief:
{json.dumps(editorial_brief, ensure_ascii=True)}

Structured market data:
{market_data_block}

Source summaries:
{json.dumps(compact_sources, ensure_ascii=True)}

Requirements:
- write a strong headline
- aim for roughly 550 to 800 words
- include a short opening note with a point of view
- organize the body around the planned sections
- make it readable, analytical, and confident
- do not sound generic, padded, or corporate
- do not merely aggregate events; synthesize them into a sharp argument
- prove the thesis aggressively instead of hinting at it
- surface the hidden pattern the average reader would miss
- make at least one concrete interpretation about what mattered most this week
- include one killer insight that makes the reader rethink the topic
- prefer sharp-smart over safe-smart
- avoid consultant language, hedging, and empty acceleration talk
- follow the explanation guidance exactly
- if depth is low, keep it crisp and selective
- if depth is medium, balance signal and concision
- if depth is high, go deeper on implications and cross-source synthesis
- use inline citations like [1], [2]
- if structured market data is present, cite it as [M1]
- if structured market data is provided, use the exact percentage moves and prices from it instead of vague descriptions
- when you mention top market movers, include the actual numbers such as 24h or 7d percentage moves
- end with a short closing note
- include a final Sources section listing [id]: title - url
- if structured market data is present, add: [M1]: CoinGecko Markets API - https://www.coingecko.com/
- return markdown only
<end_of_turn>
<start_of_turn>model
"""

    return generate_text_from_prompt(prompt, newsletter_tokens).strip()


def compose_fallback_newsletter(
    brief,
    plan,
    days,
    market_snapshot,
    depth,
    explanation_style,
    custom_style_instructions,
    newsletter_tokens,
):
    explanation_guidance = build_explanation_guidance(
        explanation_style,
        custom_style_instructions,
    )
    prompt = f"""<start_of_turn>user
You are the writing brain for a newsletter agent.

Write a markdown newsletter draft even though no web sources were successfully collected.

Newsletter brief:
"{brief}"

Title:
"{plan['title']}"

Audience:
"{plan['audience']}"

Tone:
"{plan['tone']}"

House style:
"{DEFAULT_WRITING_STYLE}"

Explanation style:
{explanation_style}

Explanation guidance:
"{explanation_guidance}"

Coverage window:
Last {days} days

Research depth:
{depth}

Planned sections:
{json.dumps(plan['sections'])}

Structured market data:
{json.dumps(market_snapshot, ensure_ascii=True)}

Requirements:
- begin with a bold note that live web source collection failed
- keep the draft within roughly 450 to 700 words
- state clearly that the draft is based on the brief, plan, and any structured market data only
- do not invent recent events, dates, quotes, numbers, or citations from external reporting
- if structured market data is present, you may reference it and cite it as [M1]
- if there is not enough verified information, explicitly say what is unknown
- organize the body around the planned sections
- keep the writing useful, analytical, and honest about uncertainty
- end with a short next-steps note
- include a final Sources section
- if structured market data is present, list only: [M1]: CoinGecko Markets API - https://www.coingecko.com/
- if no structured market data is present, say in Sources that no external sources were successfully collected
- return markdown only
<end_of_turn>
<start_of_turn>model
"""

    return generate_text_from_prompt(prompt, newsletter_tokens).strip()


def build_editorial_brief(brief, plan, compact_sources, market_snapshot, depth):
    prompt = f"""<start_of_turn>user
You are the editorial strategist for a premium newsletter.

Your job is to transform raw source summaries into a sharp point of view before drafting begins.

Newsletter brief:
"{brief}"

Audience:
"{plan['audience']}"

Tone:
"{plan['tone']}"

Depth:
{depth}

Planned sections:
{json.dumps(plan['sections'])}

Structured market data:
{json.dumps(market_snapshot, ensure_ascii=True)}

Source summaries:
{json.dumps(compact_sources, ensure_ascii=True)}

Return ONLY JSON in this format:
{{"core_thesis":"str","hidden_pattern":"str","killer_insight":"str","contrarian_take":"str","proof_points":["p1","p2","p3"]}}

Rules:
- the thesis must be explicit and defensible
- the hidden pattern must go beyond restating events
- the killer insight should feel memorable and surprising
- the contrarian take should sharpen the voice without becoming fake or sensational
- proof_points should be concrete claims the final newsletter should prove
- do not include any text outside JSON
<end_of_turn>
<start_of_turn>model
{{"core_thesis":"""

    try:
        return generate_json_from_prompt(
            prompt,
            '{"core_thesis":',
            220,
            schema_hint='{"core_thesis":"str","hidden_pattern":"str","killer_insight":"str","contrarian_take":"str","proof_points":["p1","p2","p3"]}',
        )
    except Exception:
        return {
            "core_thesis": "This week was not just busy; it showed a clearer strategic direction in the field.",
            "hidden_pattern": "Parallel progress across multiple fronts usually matters more than any single headline.",
            "killer_insight": "When separate parts of an industry start solving adjacent bottlenecks at once, the story shifts from isolated progress to ecosystem readiness.",
            "contrarian_take": "The real signal is not hype volume but whether different constraints are easing at the same time.",
            "proof_points": [
                "Multiple sources point to simultaneous progress rather than one-off noise.",
                "The most important development is the change in system-level readiness.",
                "Readers should come away with a sharper frame, not just a recap.",
            ],
        }


def generate_json_from_prompt(prompt, prefill, max_tokens, schema_hint=None, retries=2):
    response = generate_with_runtime(prompt, max_tokens)
    parsed = try_parse_model_json(response, prefill)
    if parsed is not None:
        return parsed

    last_response = response
    for _ in range(retries):
        repair_prompt = build_json_repair_prompt(last_response, prefill, schema_hint)
        repaired_response = generate_text_from_prompt(repair_prompt, max_tokens)
        parsed = try_parse_model_json(repaired_response, "")
        if parsed is not None:
            return parsed
        last_response = repaired_response

    raise ValueError("Model did not return valid JSON")


def generate_text_from_prompt(prompt, max_tokens):
    return generate_with_runtime(prompt, max_tokens).strip()


def generate_with_runtime(prompt, max_tokens):
    runtime = initialize_model_runtime()

    if runtime["runtime_backend"] == "mlx":
        return runtime["generate_fn"](
            MODEL,
            TOKENIZER,
            prompt=prompt,
            max_tokens=max_tokens,
            draft_model=runtime["draft_model"],
            num_draft_tokens=runtime["num_draft_tokens"],
        ).strip()

    if runtime["runtime_backend"] == "transformers":
        return generate_with_transformers(prompt, max_tokens, runtime)

    raise RuntimeError(f"Unsupported runtime backend: {runtime['runtime_backend']}")


def generate_with_transformers(prompt, max_tokens, runtime):
    import torch

    encoded = TOKENIZER(prompt, return_tensors="pt")
    device = runtime.get("device", "cpu")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]
    pad_token_id = TOKENIZER.pad_token_id
    if pad_token_id is None:
        pad_token_id = TOKENIZER.eos_token_id

    with torch.no_grad():
        output = MODEL.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    generated = output[0][input_length:]
    return TOKENIZER.decode(generated, skip_special_tokens=True).strip()


def try_parse_model_json(response_text, prefill):
    candidates = []
    stripped = strip_code_fences(response_text.strip())
    if prefill:
        candidates.append(prefill + stripped)
    candidates.append(stripped)

    for candidate in candidates:
        parsed = try_parse_json_candidate(candidate)
        if parsed is not None:
            return parsed

    return None


def try_parse_json_candidate(text):
    clean_text_value = "".join(char for char in text if ord(char) >= 32)

    try:
        return json.loads(clean_text_value)
    except json.JSONDecodeError:
        pass

    extracted = extract_json_object(clean_text_value)
    if not extracted:
        return None

    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        return None


def extract_json_object(text):
    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return ""


def strip_code_fences(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def build_json_repair_prompt(response_text, prefill, schema_hint):
    expected_shape = schema_hint or "Return one valid JSON object."
    prompt = f"""<start_of_turn>user
Convert the following model output into one valid JSON object.

Expected JSON shape:
{expected_shape}

Original output:
\"\"\"
{prefill}{response_text.strip()}
\"\"\"

Rules:
- return valid JSON only
- do not include markdown fences
- do not include explanations
<end_of_turn>
<start_of_turn>model
"""
    return prompt


def save_run(plan, brief, depth, explanation_style, custom_style_instructions):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO newsletter_runs (
            brief,
            depth,
            explanation_style,
            custom_style_instructions,
            audience,
            tone,
            title,
            queries_json,
            sections_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            brief,
            depth,
            explanation_style,
            custom_style_instructions,
            plan["audience"],
            plan["tone"],
            plan["title"],
            json.dumps(plan["queries"]),
            json.dumps(plan["sections"]),
        ),
    )
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id


def save_source(run_id, source_record):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO source_items (
            run_id,
            query_text,
            rank_index,
            title,
            url,
            snippet,
            article_text,
            source_summary,
            relevance_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            source_record["query"],
            source_record["rank_index"],
            source_record["title"],
            source_record["url"],
            source_record["snippet"],
            source_record["article_text"],
            source_record["source_summary"],
            source_record["relevance_score"],
        ),
    )
    conn.commit()
    conn.close()


def update_run_output_path(run_id, output_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE newsletter_runs SET output_path = ? WHERE id = ?",
        (output_path, run_id),
    )
    conn.commit()
    conn.close()


def write_newsletter_files(output_dir, title, markdown):
    os.makedirs(output_dir, exist_ok=True)
    created_at = datetime.now()
    timestamp = created_at.strftime("%Y%m%d_%H%M%S")
    slug = slugify(title)
    normalized_markdown = normalize_newsletter_markdown(markdown)
    markdown_path = os.path.join(output_dir, f"{timestamp}_{slug}.md")
    html_path = os.path.join(output_dir, f"{timestamp}_{slug}.html")

    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(normalized_markdown + "\n")

    editable_html = render_editable_newsletter_html(title, normalized_markdown, created_at)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(editable_html)

    return {
        "markdown_path": markdown_path,
        "html_path": html_path,
    }


def normalize_newsletter_markdown(markdown):
    normalized = strip_code_fences(str(markdown or "")).replace("\r\n", "\n").strip()
    return normalized


def render_editable_newsletter_html(fallback_title, markdown, created_at):
    document = parse_newsletter_markdown(markdown, fallback_title)
    display_title = document["title"] or fallback_title or "Newsletter"
    standfirst = document["standfirst"] or "Add an opening note that frames the issue with conviction."
    body_html = render_newsletter_body_html(document["blocks"])
    sources_html = render_sources_html(document["sources"])
    edition_label = created_at.strftime("%B %d, %Y")
    storage_key = f"newsletter-studio::{slugify(display_title)}::{created_at.strftime('%Y%m%d%H%M%S')}"
    description = html.escape(extract_plain_text(standfirst)[:160], quote=True)
    safe_title = html.escape(display_title)
    safe_standfirst = format_inline_markdown(standfirst)
    download_name = f"{slugify(display_title)}-editable.html"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <meta name="description" content="{description}">
  <style>
    :root {{
      --bg: #f3ede4;
      --paper: rgba(255, 250, 244, 0.88);
      --paper-strong: #fffaf4;
      --ink: #17120d;
      --muted: #6e6153;
      --accent: #b85c38;
      --accent-deep: #7f3621;
      --accent-soft: rgba(184, 92, 56, 0.14);
      --line: rgba(40, 28, 18, 0.12);
      --shadow: 0 28px 90px rgba(58, 32, 15, 0.14);
      --display-font: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      --ui-font: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      --mono-font: "SFMono-Regular", "Menlo", "Consolas", monospace;
    }}

    * {{
      box-sizing: border-box;
    }}

    html {{
      scroll-behavior: smooth;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 220, 194, 0.9), transparent 34%),
        radial-gradient(circle at 85% 12%, rgba(201, 112, 67, 0.16), transparent 22%),
        linear-gradient(180deg, #f8f1e7 0%, #efe5d8 48%, #ecdfd1 100%);
      font-family: var(--ui-font);
    }}

    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.28), transparent 40%),
        repeating-linear-gradient(
          0deg,
          rgba(23, 18, 13, 0.02),
          rgba(23, 18, 13, 0.02) 1px,
          transparent 1px,
          transparent 8px
        );
      mix-blend-mode: multiply;
      opacity: 0.42;
    }}

    .page-shell {{
      width: min(1280px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 72px;
      position: relative;
      z-index: 1;
    }}

    .studio-toolbar {{
      position: sticky;
      top: 18px;
      z-index: 20;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 14px 18px;
      border: 1px solid rgba(255, 255, 255, 0.45);
      border-radius: 22px;
      background: rgba(27, 18, 10, 0.78);
      box-shadow: 0 18px 40px rgba(15, 10, 7, 0.18);
      backdrop-filter: blur(22px);
      color: #fff8ef;
    }}

    .toolbar-copy {{
      display: flex;
      flex-direction: column;
      gap: 3px;
      min-width: 220px;
    }}

    .toolbar-kicker {{
      font-size: 0.72rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: rgba(255, 248, 239, 0.64);
    }}

    .toolbar-title {{
      font-size: 0.98rem;
      font-weight: 600;
      letter-spacing: 0.01em;
    }}

    .toolbar-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}

    .toolbar-button,
    .toolbar-pill {{
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      font: inherit;
      font-size: 0.88rem;
      cursor: pointer;
      transition: transform 160ms ease, background 160ms ease, color 160ms ease;
    }}

    .toolbar-button {{
      background: rgba(255, 255, 255, 0.12);
      color: #fff8ef;
    }}

    .toolbar-button:hover,
    .toolbar-button:focus-visible {{
      transform: translateY(-1px);
      background: rgba(255, 255, 255, 0.2);
      outline: none;
    }}

    .toolbar-button.primary {{
      background: linear-gradient(135deg, #d8794e, #b65333);
      color: white;
    }}

    .toolbar-pill {{
      background: rgba(255, 255, 255, 0.08);
      color: rgba(255, 248, 239, 0.7);
      cursor: default;
    }}

    .studio-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 28px;
      margin-top: 28px;
      align-items: start;
    }}

    .newsletter-card {{
      position: relative;
      overflow: hidden;
      border-radius: 34px;
      border: 1px solid rgba(255, 255, 255, 0.5);
      background:
        radial-gradient(circle at top right, rgba(255, 227, 205, 0.84), transparent 28%),
        linear-gradient(180deg, rgba(255, 252, 248, 0.96), rgba(255, 248, 240, 0.94));
      box-shadow: var(--shadow);
      animation: rise 480ms ease both;
    }}

    .newsletter-card::after {{
      content: "";
      position: absolute;
      right: -90px;
      top: -120px;
      width: 320px;
      height: 320px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(216, 121, 78, 0.22), transparent 68%);
      pointer-events: none;
    }}

    .hero {{
      position: relative;
      padding: 68px 74px 34px;
    }}

    .hero-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 18px;
      align-items: center;
    }}

    .eyebrow,
    .meta-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 14px;
      font-size: 0.76rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }}

    .eyebrow {{
      background: rgba(255, 255, 255, 0.72);
      color: var(--accent-deep);
      border: 1px solid rgba(184, 92, 56, 0.18);
    }}

    .meta-chip {{
      background: rgba(24, 18, 13, 0.06);
      color: rgba(23, 18, 13, 0.65);
      border: 1px solid rgba(23, 18, 13, 0.08);
    }}

    .headline {{
      margin: 0;
      max-width: 11ch;
      font-family: var(--display-font);
      font-size: clamp(3rem, 8vw, 5.9rem);
      line-height: 0.94;
      letter-spacing: -0.05em;
    }}

    .standfirst {{
      margin: 22px 0 0;
      max-width: 760px;
      color: rgba(23, 18, 13, 0.76);
      font-size: clamp(1.02rem, 1.2vw + 0.8rem, 1.34rem);
      line-height: 1.75;
    }}

    .content-shell {{
      display: grid;
      gap: 30px;
      padding: 0 34px 34px;
    }}

    .story-panel,
    .sources-panel,
    .rail-card {{
      border-radius: 28px;
      border: 1px solid var(--line);
      background: var(--paper);
      backdrop-filter: blur(12px);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.46);
    }}

    .story-panel {{
      padding: 42px 40px 44px;
    }}

    .story-panel p,
    .story-panel li {{
      margin: 0 0 1.2em;
      font-family: var(--display-font);
      font-size: 1.14rem;
      line-height: 1.92;
      color: rgba(23, 18, 13, 0.92);
    }}

    .story-panel p:last-child,
    .story-panel li:last-child {{
      margin-bottom: 0;
    }}

    .story-panel p:first-child::first-letter {{
      float: left;
      margin: 0.12em 0.12em 0 0;
      font-family: var(--display-font);
      font-size: 4.7rem;
      line-height: 0.75;
      color: var(--accent-deep);
    }}

    .story-panel h2,
    .story-panel h3 {{
      margin: 2.3rem 0 1rem;
      color: var(--ink);
      line-height: 1.08;
    }}

    .story-panel h2 {{
      font-family: var(--ui-font);
      font-size: 0.92rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }}

    .story-panel h3 {{
      font-family: var(--display-font);
      font-size: 1.8rem;
      letter-spacing: -0.03em;
    }}

    .story-panel ul,
    .story-panel ol {{
      margin: 0 0 1.5rem;
      padding-left: 1.3rem;
    }}

    .story-panel blockquote {{
      margin: 1.8rem 0;
      padding: 1rem 1.2rem 1rem 1.4rem;
      border-left: 4px solid var(--accent);
      background: rgba(184, 92, 56, 0.06);
      font-family: var(--display-font);
      font-size: 1.14rem;
      color: rgba(23, 18, 13, 0.88);
    }}

    .story-panel hr {{
      border: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(23, 18, 13, 0.18), transparent);
      margin: 2rem 0;
    }}

    .citation {{
      display: inline-flex;
      transform: translateY(-0.1em);
      margin-left: 0.08em;
      padding: 0.1em 0.48em;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-deep);
      font-family: var(--ui-font);
      font-size: 0.76rem;
      font-weight: 600;
      line-height: 1.4;
      vertical-align: middle;
    }}

    .insight-card {{
      margin: 2rem 0;
      padding: 1.4rem 1.5rem;
      border-radius: 22px;
      background:
        linear-gradient(135deg, rgba(184, 92, 56, 0.12), rgba(255, 255, 255, 0.82)),
        var(--paper-strong);
      border: 1px solid rgba(184, 92, 56, 0.18);
      box-shadow: 0 14px 28px rgba(184, 92, 56, 0.08);
    }}

    .insight-label {{
      margin-bottom: 0.55rem;
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent-deep);
    }}

    .insight-card p {{
      margin: 0;
    }}

    .sources-panel {{
      padding: 34px 34px 36px;
    }}

    .section-label {{
      margin: 0 0 8px;
      color: var(--accent-deep);
      font-size: 0.74rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }}

    .section-title {{
      margin: 0 0 20px;
      font-family: var(--display-font);
      font-size: clamp(1.9rem, 3vw, 2.5rem);
      line-height: 1;
      letter-spacing: -0.04em;
    }}

    .sources-grid {{
      display: grid;
      gap: 14px;
    }}

    .source-card {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 16px;
      align-items: start;
      padding: 16px 18px;
      border-radius: 20px;
      border: 1px solid rgba(23, 18, 13, 0.08);
      background: rgba(255, 255, 255, 0.72);
      color: inherit;
      text-decoration: none;
    }}

    .source-ref {{
      min-width: 52px;
      padding-top: 2px;
      color: var(--accent-deep);
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }}

    .source-title {{
      display: block;
      font-size: 1rem;
      font-weight: 600;
      line-height: 1.45;
      color: var(--ink);
    }}

    .source-url {{
      display: inline-block;
      margin-top: 6px;
      color: rgba(23, 18, 13, 0.58);
      font-size: 0.84rem;
      word-break: break-word;
    }}

    .sources-empty {{
      margin: 0;
      color: rgba(23, 18, 13, 0.56);
      font-size: 0.98rem;
      line-height: 1.7;
    }}

    .rail {{
      display: grid;
      gap: 18px;
      position: sticky;
      top: 106px;
    }}

    .rail-card {{
      padding: 18px 18px 20px;
    }}

    .rail-card h3 {{
      margin: 0 0 8px;
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent-deep);
    }}

    .rail-card p,
    .rail-card li {{
      margin: 0;
      color: rgba(23, 18, 13, 0.72);
      font-size: 0.95rem;
      line-height: 1.7;
    }}

    .rail-card ul {{
      margin: 12px 0 0;
      padding-left: 1.1rem;
    }}

    .metric {{
      display: flex;
      align-items: baseline;
      gap: 8px;
      margin-top: 12px;
    }}

    .metric strong {{
      font-family: var(--display-font);
      font-size: 2.5rem;
      line-height: 1;
      letter-spacing: -0.04em;
    }}

    .editable {{
      outline: none;
      transition: box-shadow 160ms ease, background 160ms ease;
    }}

    .editable:focus {{
      background: rgba(255, 255, 255, 0.58);
      box-shadow: 0 0 0 3px rgba(184, 92, 56, 0.18);
      border-radius: 12px;
    }}

    .status-ok {{
      color: #ffd9c9;
    }}

    .toolbar-hint {{
      color: rgba(255, 248, 239, 0.58);
      font-size: 0.78rem;
    }}

    @keyframes rise {{
      from {{
        opacity: 0;
        transform: translateY(14px);
      }}
      to {{
        opacity: 1;
        transform: translateY(0);
      }}
    }}

    @media (max-width: 1040px) {{
      .studio-grid {{
        grid-template-columns: 1fr;
      }}

      .rail {{
        position: static;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      }}
    }}

    @media (max-width: 760px) {{
      .page-shell {{
        width: min(100vw - 16px, 100%);
        padding-top: 16px;
      }}

      .studio-toolbar {{
        top: 10px;
        padding: 12px;
        border-radius: 18px;
      }}

      .hero {{
        padding: 42px 24px 22px;
      }}

      .headline {{
        max-width: none;
      }}

      .content-shell {{
        padding: 0 16px 16px;
      }}

      .story-panel,
      .sources-panel {{
        padding: 26px 22px 28px;
      }}
    }}

    @media print {{
      body {{
        background: white;
      }}

      body::before,
      .studio-toolbar,
      .rail {{
        display: none !important;
      }}

      .page-shell {{
        width: 100%;
        padding: 0;
      }}

      .newsletter-card,
      .story-panel,
      .sources-panel {{
        border: none;
        box-shadow: none;
        background: white;
      }}
    }}
  </style>
</head>
<body>
  <div class="page-shell">
    <div class="studio-toolbar">
      <div class="toolbar-copy">
        <span class="toolbar-kicker">Newsletter Studio</span>
        <strong class="toolbar-title">{safe_title}</strong>
      </div>
      <div class="toolbar-actions">
        <button class="toolbar-button" type="button" data-command="bold">Bold</button>
        <button class="toolbar-button" type="button" data-command="italic">Italic</button>
        <button class="toolbar-button" type="button" data-command="formatBlock" data-value="H2">H2</button>
        <button class="toolbar-button" type="button" data-command="formatBlock" data-value="BLOCKQUOTE">Quote</button>
        <button class="toolbar-button" type="button" data-command="insertHorizontalRule">Rule</button>
        <button class="toolbar-button primary" type="button" data-action="download">Save HTML</button>
        <button class="toolbar-button" type="button" data-action="print">Print</button>
        <button class="toolbar-button" type="button" data-action="reset">Reset</button>
        <span class="toolbar-pill" id="save-status">Draft loaded</span>
        <span class="toolbar-pill" id="word-count">0 words</span>
      </div>
    </div>

    <div class="studio-grid">
      <article class="newsletter-card" data-editable-root>
        <header class="hero">
          <div class="hero-meta">
            <span class="eyebrow">Ready to publish</span>
            <span class="meta-chip">{html.escape(edition_label)}</span>
            <span class="meta-chip">Editable local draft</span>
          </div>
          <h1 class="headline editable" contenteditable="true" spellcheck="true">{safe_title}</h1>
          <p class="standfirst editable" contenteditable="true" spellcheck="true">{safe_standfirst}</p>
        </header>

        <div class="content-shell">
          <section class="story-panel editable" contenteditable="true" spellcheck="true" id="story-body">
{body_html}
          </section>

          <section class="sources-panel">
            <p class="section-label">Verification trail</p>
            <h2 class="section-title editable" contenteditable="true" spellcheck="true">Sources</h2>
            <div class="sources-grid editable" contenteditable="true" spellcheck="true">
{sources_html}
            </div>
          </section>
        </div>
      </article>

      <aside class="rail">
        <section class="rail-card">
          <h3>Editing flow</h3>
          <p>Click anywhere in the headline, dek, body, or sources and edit directly. The page saves your draft in local browser storage.</p>
        </section>
        <section class="rail-card">
          <h3>Publish finish</h3>
          <p>Use <strong>Save HTML</strong> when the newsletter is ready. Use <strong>Print</strong> for a clean PDF or a quick browser-based export.</p>
        </section>
        <section class="rail-card">
          <h3>Word count</h3>
          <div class="metric">
            <strong id="word-count-rail">0</strong>
            <span>words</span>
          </div>
          <p style="margin-top: 12px;">This count updates as you edit the draft.</p>
        </section>
      </aside>
    </div>
  </div>

  <script>
    (() => {{
      const editorRoot = document.querySelector("[data-editable-root]");
      const storyBody = document.getElementById("story-body");
      const saveStatus = document.getElementById("save-status");
      const wordCountBadge = document.getElementById("word-count");
      const wordCountRail = document.getElementById("word-count-rail");
      const storageKey = {json.dumps(storage_key)};
      const downloadName = {json.dumps(download_name)};
      const initialMarkup = editorRoot.innerHTML;
      let saveTimer = null;

      function countWords(text) {{
        const trimmed = text.trim();
        return trimmed ? trimmed.split(/\\s+/).length : 0;
      }}

      function updateWordCount() {{
        const total = countWords(storyBody.innerText);
        wordCountBadge.textContent = `${{total}} words`;
        wordCountRail.textContent = String(total);
      }}

      function setStatus(message) {{
        saveStatus.textContent = message;
      }}

      function persistDraft() {{
        try {{
          localStorage.setItem(storageKey, editorRoot.innerHTML);
          setStatus("Saved locally");
        }} catch (error) {{
          setStatus("Local save unavailable");
        }}
        updateWordCount();
      }}

      function queueSave() {{
        setStatus("Editing...");
        window.clearTimeout(saveTimer);
        saveTimer = window.setTimeout(persistDraft, 250);
      }}

      function restoreDraft() {{
        try {{
          const savedDraft = localStorage.getItem(storageKey);
          if (savedDraft) {{
            editorRoot.innerHTML = savedDraft;
            setStatus("Recovered local draft");
          }}
        }} catch (error) {{
          setStatus("Draft loaded");
        }}
      }}

      function downloadHtml() {{
        const content = "<!DOCTYPE html>\\n" + document.documentElement.outerHTML;
        const blob = new Blob([content], {{ type: "text/html;charset=utf-8" }});
        const href = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = href;
        link.download = downloadName;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(href);
        setStatus("HTML exported");
      }}

      function resetDraft() {{
        editorRoot.innerHTML = initialMarkup;
        try {{
          localStorage.removeItem(storageKey);
        }} catch (error) {{
          // ignore storage failures during reset
        }}
        setStatus("Reset to generated draft");
        updateWordCount();
      }}

      document.querySelectorAll("[data-command]").forEach((button) => {{
        button.addEventListener("click", () => {{
          const command = button.dataset.command;
          const value = button.dataset.value || null;
          document.execCommand(command, false, value);
          queueSave();
        }});
      }});

      document.querySelectorAll("[data-action]").forEach((button) => {{
        button.addEventListener("click", () => {{
          const action = button.dataset.action;
          if (action === "download") {{
            downloadHtml();
          }} else if (action === "print") {{
            window.print();
          }} else if (action === "reset") {{
            resetDraft();
          }}
        }});
      }});

      editorRoot.addEventListener("input", queueSave);
      restoreDraft();
      updateWordCount();
    }})();
  </script>
</body>
</html>
"""


def parse_newsletter_markdown(markdown, fallback_title):
    lines = normalize_newsletter_markdown(markdown).splitlines()
    title = fallback_title.strip() or "Newsletter"
    blocks = []
    sources = []
    paragraph_lines = []
    list_items = []
    list_type = None
    in_sources_section = False

    def flush_paragraph():
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        blocks.append(
            {
                "type": "paragraph",
                "text": " ".join(line.strip() for line in paragraph_lines).strip(),
            }
        )
        paragraph_lines = []

    def flush_list():
        nonlocal list_items
        nonlocal list_type
        if not list_items:
            return
        blocks.append(
            {
                "type": "list",
                "ordered": list_type == "ordered",
                "items": list_items[:],
            }
        )
        list_items = []
        list_type = None

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        title_match = re.match(r"^#\s+(.+)$", stripped)
        if title_match:
            flush_paragraph()
            flush_list()
            title = extract_plain_text(title_match.group(1)) or title
            continue

        if is_sources_heading(stripped):
            flush_paragraph()
            flush_list()
            in_sources_section = True
            continue

        source_match = re.match(r"^\[([^\]]+)\]:\s*(.+)$", stripped)
        if source_match:
            flush_paragraph()
            flush_list()
            in_sources_section = True
            sources.append(build_source_entry(source_match.group(1), source_match.group(2)))
            continue

        if in_sources_section:
            if sources:
                sources[-1]["title"] = clean_text(f"{sources[-1]['title']} {stripped}")
            continue

        heading_match = re.match(r"^(#{2,6})\s+(.+)$", stripped)
        if heading_match:
            flush_paragraph()
            flush_list()
            blocks.append(
                {
                    "type": "heading",
                    "level": min(max(len(heading_match.group(1)), 2), 3),
                    "text": heading_match.group(2).strip(),
                }
            )
            continue

        standalone_bold_match = re.match(r"^\*\*(.+?)\*\*$", stripped)
        if standalone_bold_match:
            flush_paragraph()
            flush_list()
            blocks.append(
                {
                    "type": "heading",
                    "level": 2,
                    "text": standalone_bold_match.group(1).strip(),
                }
            )
            continue

        unordered_match = re.match(r"^[-*]\s+(.+)$", stripped)
        if unordered_match:
            flush_paragraph()
            item_text = unordered_match.group(1).strip()
            if list_type not in {None, "unordered"}:
                flush_list()
            list_type = "unordered"
            list_items.append(item_text)
            continue

        ordered_match = re.match(r"^\d+\.\s+(.+)$", stripped)
        if ordered_match:
            flush_paragraph()
            item_text = ordered_match.group(1).strip()
            if list_type not in {None, "ordered"}:
                flush_list()
            list_type = "ordered"
            list_items.append(item_text)
            continue

        if stripped == "---":
            flush_paragraph()
            flush_list()
            blocks.append({"type": "divider"})
            continue

        if list_items:
            flush_list()
        paragraph_lines.append(stripped)

    flush_paragraph()
    flush_list()

    standfirst = ""
    if blocks and blocks[0]["type"] == "paragraph":
        standfirst = blocks.pop(0)["text"]

    return {
        "title": title,
        "standfirst": standfirst,
        "blocks": blocks,
        "sources": sources,
    }


def is_sources_heading(text):
    normalized = extract_plain_text(text).lower().rstrip(":")
    return normalized in {"sources", "final sources"}


def build_source_entry(label, content):
    source_text = clean_text(content)
    url = ""
    title = source_text
    url_match = re.search(r"(https?://\S+)$", source_text)
    if url_match:
        url = url_match.group(1).rstrip(").,")
        title = source_text[: url_match.start()].rstrip(" -:")
    return {
        "label": f"[{label}]",
        "title": title or source_text,
        "url": url,
    }


def render_newsletter_body_html(blocks):
    if not blocks:
        return "            <p>Start writing. This body is directly editable.</p>\n"

    fragments = []
    for block in blocks:
        if block["type"] == "heading":
            tag_name = "h2" if block["level"] <= 2 else "h3"
            fragments.append(f"            <{tag_name}>{format_inline_markdown(block['text'])}</{tag_name}>")
            continue

        if block["type"] == "paragraph":
            insight_match = re.match(r"^\*\*Killer Insight:\*\*\s*(.+)$", block["text"], re.IGNORECASE)
            if insight_match:
                fragments.append(
                    "            <aside class=\"insight-card\">"
                    "<div class=\"insight-label\">Killer Insight</div>"
                    f"<p>{format_inline_markdown(insight_match.group(1).strip())}</p>"
                    "</aside>"
                )
            else:
                fragments.append(f"            <p>{format_inline_markdown(block['text'])}</p>")
            continue

        if block["type"] == "list":
            tag_name = "ol" if block["ordered"] else "ul"
            list_items = "".join(
                f"<li>{format_inline_markdown(item)}</li>"
                for item in block["items"]
            )
            fragments.append(f"            <{tag_name}>{list_items}</{tag_name}>")
            continue

        if block["type"] == "divider":
            fragments.append("            <hr>")

    return "\n".join(fragments) + "\n"


def render_sources_html(sources):
    if not sources:
        return (
            "              <p class=\"sources-empty\">"
            "Add links, citations, or reporting notes here before publishing."
            "</p>\n"
        )

    cards = []
    for source in sources:
        label = html.escape(source["label"])
        title = html.escape(source["title"])
        url = html.escape(source["url"], quote=True)
        display_url = html.escape(source["url"] or "Add source URL")
        if source["url"]:
            card_html = (
                "              <a class=\"source-card\" "
                f"href=\"{url}\" target=\"_blank\" rel=\"noreferrer\">"
                f"<span class=\"source-ref\">{label}</span>"
                "<span>"
                f"<span class=\"source-title\">{title}</span>"
                f"<span class=\"source-url\">{display_url}</span>"
                "</span>"
                "</a>"
            )
        else:
            card_html = (
                "              <div class=\"source-card\">"
                f"<span class=\"source-ref\">{label}</span>"
                "<span>"
                f"<span class=\"source-title\">{title}</span>"
                f"<span class=\"source-url\">{display_url}</span>"
                "</span>"
                "</div>"
            )
        cards.append(card_html)

    return "\n".join(cards) + "\n"


def format_inline_markdown(text):
    safe_text = html.escape(text)
    safe_text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        lambda match: (
            f'<a href="{html.escape(match.group(2), quote=True)}" target="_blank" '
            f'rel="noreferrer">{html.escape(match.group(1))}</a>'
        ),
        safe_text,
    )
    safe_text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe_text)
    safe_text = re.sub(r"(?<!\*)\*(.+?)\*(?!\*)", r"<em>\1</em>", safe_text)
    safe_text = re.sub(
        r"(?<!\w)\[(M?\d+)\]",
        r'<span class="citation">[\1]</span>',
        safe_text,
    )
    return safe_text


def extract_plain_text(text):
    plain_text = str(text or "")
    plain_text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", plain_text)
    plain_text = re.sub(r"[*_`#>\[\]]", "", plain_text)
    return clean_text(plain_text)


def slugify(value):
    lowered = value.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return cleaned or "newsletter"


def clean_text(value):
    stripped = re.sub(r"\s+", " ", value)
    return stripped.strip()


if __name__ == "__main__":
    main()

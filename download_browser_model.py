import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DESTINATION = PROJECT_ROOT / "models" / "onnx-community" / "gemma-3n-E2B-it-ONNX"
DEFAULT_REPO_ID = "onnx-community/gemma-3n-E2B-it-ONNX"
CORE_ROOT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
PRECISION_CANDIDATES = (
    (
        "q4",
        (
            "onnx/decoder_model_merged_q4.onnx",
            "onnx/decoder_model_merged_q4.onnx_data",
            "onnx/embed_tokens_q4.onnx",
            "onnx/embed_tokens_q4.onnx_data",
        ),
    ),
    (
        "q4f16",
        (
            "onnx/decoder_model_merged_q4f16.onnx",
            "onnx/decoder_model_merged_q4f16.onnx_data",
            "onnx/embed_tokens_q4f16.onnx",
            "onnx/embed_tokens_q4f16.onnx_data",
        ),
    ),
    (
        "quantized",
        (
            "onnx/decoder_model_merged_quantized.onnx",
            "onnx/decoder_model_merged_quantized.onnx_data",
            "onnx/embed_tokens_quantized.onnx",
            "onnx/embed_tokens_quantized.onnx_data",
        ),
    ),
    (
        "default",
        (
            "onnx/decoder_model_merged.onnx",
            "onnx/decoder_model_merged.onnx_data",
            "onnx/embed_tokens.onnx",
            "onnx/embed_tokens.onnx_data",
        ),
    ),
)
AUXILIARY_COMPONENTS = {
    "audio_encoder": ("q4f16", "q4", "int8", "uint8", "quantized", "fp16", "fp32"),
    "vision_encoder": ("uint8", "int8", "quantized", "fp16", "fp32"),
}
GENERIC_ONNX_BASES = {
    "q4": ("onnx/model_q4.onnx",),
    "q4f16": ("onnx/model_q4f16.onnx",),
    "quantized": ("onnx/model_quantized.onnx", "onnx/model_int8.onnx", "onnx/model_uint8.onnx"),
    "default": ("onnx/model.onnx", "onnx/model_fp16.onnx"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Download a public Gemma 3n ONNX browser bundle with no Hugging Face token.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--destination", default=str(DEFAULT_DESTINATION))
    parser.add_argument(
        "--precision",
        choices=[item[0] for item in PRECISION_CANDIDATES],
        help="Preferred ONNX precision. Defaults to the lightest supported pair available in the repo.",
    )
    parser.add_argument(
        "--include-multimodal",
        action="store_true",
        help="Include audio/vision encoder files. Default is text-only bundle.",
    )
    args = parser.parse_args()

    destination = Path(args.destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(args.repo_id, repo_type="model")
    selected_precision, allow_patterns = build_allow_patterns(
        repo_files,
        args.precision,
        include_multimodal=args.include_multimodal,
    )

    print(f"Downloading {args.repo_id}")
    print(f"Destination: {destination}")
    print(f"Selected precision: {selected_precision}")
    print("Files:")
    for pattern in allow_patterns:
        print(f"  - {pattern}")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    print("Browser model download complete.")


def build_allow_patterns(repo_files, requested_precision, include_multimodal=False):
    repo_file_set = set(repo_files)
    allow_patterns = [file_name for file_name in CORE_ROOT_FILES if file_name in repo_file_set]

    selected_precision = None
    candidate_order = []
    if requested_precision:
        candidate_order.append(requested_precision)
    candidate_order.extend(
        precision
        for precision, _ in PRECISION_CANDIDATES
        if precision != requested_precision
    )

    for precision_name in candidate_order:
        matched_files = collect_precision_files(repo_files, precision_name)
        if matched_files:
            selected_precision = precision_name
            allow_patterns.extend(matched_files)
            break

    if not selected_precision:
        for precision_name in candidate_order:
            matched_files = collect_generic_precision_files(repo_files, precision_name)
            if matched_files:
                selected_precision = precision_name
                allow_patterns.extend(matched_files)
                break

    if not selected_precision:
        raise SystemExit("No supported ONNX precision set was found in the repo.")

    if include_multimodal:
        for component_name, precision_order in AUXILIARY_COMPONENTS.items():
            auxiliary_files = collect_component_files(repo_files, component_name, precision_order)
            allow_patterns.extend(auxiliary_files)

    return selected_precision, sorted(dict.fromkeys(allow_patterns))


def collect_precision_files(repo_files, precision_name):
    for candidate_name, base_files in PRECISION_CANDIDATES:
        if candidate_name != precision_name:
            continue

        matched_files = []
        for base_file in base_files:
            if any(file_name == base_file or file_name.startswith(f"{base_file}_") for file_name in repo_files):
                matched_files.extend(
                    file_name
                    for file_name in repo_files
                    if file_name == base_file or file_name.startswith(f"{base_file}_")
                )
            else:
                matched_files = []
                break
        return sorted(dict.fromkeys(matched_files))
    return []


def collect_component_files(repo_files, component_name, precision_order):
    for precision in precision_order:
        if precision == "fp32":
            base_file = f"onnx/{component_name}.onnx"
        else:
            base_file = f"onnx/{component_name}_{precision}.onnx"

        matched_files = [
            file_name
            for file_name in repo_files
            if file_name == base_file or file_name.startswith(f"{base_file}_")
        ]
        if matched_files:
            return sorted(dict.fromkeys(matched_files))

    return []


def collect_generic_precision_files(repo_files, precision_name):
    bases = GENERIC_ONNX_BASES.get(precision_name, ())
    if not bases:
        return []

    for base_file in bases:
        matched_files = [
            file_name
            for file_name in repo_files
            if file_name == base_file or file_name.startswith(f"{base_file}_")
        ]
        if matched_files:
            return sorted(dict.fromkeys(matched_files))

    return []


if __name__ == "__main__":
    main()

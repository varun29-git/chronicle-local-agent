import os
from huggingface_hub import snapshot_download

# This script pulls the ONNX weights for the local Gemma 3 brain.
# You might need to run `huggingface-cli login` first or set HF_TOKEN env var.

MODEL_REPO = "onnx-community/gemma-3-4b-it-ONNX"
LOCAL_DIR = os.path.join(os.getcwd(), "models", "gemma-3-it")

print(f"[*] Downloading {MODEL_REPO} into {LOCAL_DIR}...")

try:
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=LOCAL_DIR,
        # We only need the ONNX files and configs
        allow_patterns=["*.json", "*.onnx", "*.onnx_data", "*.txt"]
    )
    print("\n[SUCCESS] Model weights are now ready in /models/gemma-3-it/")
    print("[*] Refresh your browser (Cmd+Shift+R) to start the local brain.")
except Exception as e:
    print(f"\n[ERROR] Failed to download weights: {e}")
    print("[TIP] Make sure you have accepted the Google Gemma 3 license on Hugging Face.")
    print("[TIP] You may need to run: export HF_TOKEN=your_token_here")

# Local Model Setup

This project now prefers local model directories by default so end users do not need to authenticate with Hugging Face on first run.

Default layout:

```text
models/
  mlx-community/
    gemma-3-4b-it-4bit/
  google/
    gemma-2-2b-it/
```

Backend behavior:

- Apple Silicon macOS defaults to the MLX model at `models/mlx-community/gemma-3-4b-it-4bit`
- Other platforms default to the Transformers model at `models/google/gemma-2-2b-it`

Optional overrides:

- `NEWSLETTER_AGENT_MODEL_ROOT`
- `NEWSLETTER_AGENT_MODEL`
- `NEWSLETTER_AGENT_MODEL_TRANSFORMERS`
- `NEWSLETTER_AGENT_MODEL_LOW`
- `NEWSLETTER_AGENT_MODEL_MEDIUM`
- `NEWSLETTER_AGENT_MODEL_HIGH`
- `NEWSLETTER_AGENT_MODEL_SLICE_12_5`
- `NEWSLETTER_AGENT_MODEL_SLICE_25`
- `NEWSLETTER_AGENT_MODEL_SLICE_50`
- `NEWSLETTER_AGENT_MODEL_SLICE_75`
- `NEWSLETTER_AGENT_MODEL_SLICE_100`

If you intentionally want Hugging Face downloads instead of local files, point the relevant override at a hub model id and authenticate separately for gated models.

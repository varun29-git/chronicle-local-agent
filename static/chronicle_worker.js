const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@next";

let runtimePromise = null;
let tokenizer = null;
let model = null;
let currentModelId = "";

self.onmessage = async (event) => {
  const payload = event.data || {};
  const type = String(payload.type || "");
  const id = payload.id;

  try {
    if (type === "init") {
      await initializeWorkerModel(payload);
      self.postMessage({ type: "ready", id });
      return;
    }

    if (type === "generate") {
      const text = await generateText(payload);
      self.postMessage({ type: "result", id, text });
      return;
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      id,
      message: String(error?.message || error || "Worker failure"),
    });
  }
};

async function loadRuntime(localModelPath, numThreads) {
  if (!runtimePromise) {
    runtimePromise = import(TRANSFORMERS_CDN).then((runtime) => {
      runtime.env.allowRemoteModels = false;
      runtime.env.allowLocalModels = true;
      runtime.env.localModelPath = localModelPath || "/models";
      runtime.env.useBrowserCache = true;
      if (runtime.env.backends?.onnx?.wasm) {
        runtime.env.backends.onnx.wasm.numThreads = Math.min(4, Number(numThreads || 2));
      }
      return runtime;
    });
  }
  return runtimePromise;
}

async function initializeWorkerModel(options) {
  const modelId = String(options.model || "").trim();
  if (!modelId) {
    throw new Error("Worker init missing model id.");
  }
  if (tokenizer && model && currentModelId === modelId) {
    return;
  }

  tokenizer = null;
  model = null;
  currentModelId = "";

  const runtime = await loadRuntime(options.localModelPath, options.numThreads);
  const progress_callback = (progress) => {
    self.postMessage({ type: "progress", progress });
  };

  tokenizer = await runtime.AutoTokenizer.from_pretrained(modelId, { progress_callback });
  model = await runtime.AutoModelForCausalLM.from_pretrained(modelId, {
    device: String(options.device || "wasm"),
    dtype: String(options.dtype || "q4f16"),
    progress_callback,
  });
  currentModelId = modelId;
}

async function generateText(options) {
  if (!tokenizer || !model) {
    throw new Error("Worker model is not initialized.");
  }

  const promptText = String(options.prompt || "").trim();
  if (!promptText) {
    throw new Error("Empty prompt.");
  }

  const messages = [{ role: "user", content: promptText }];
  const prompt = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
  });
  const inputs = await tokenizer(prompt, { add_special_tokens: false, return_tensor: true });
  const inputLength = resolveInputLength(inputs.input_ids);

  const output = await withTimeout(
    model.generate({
      ...inputs,
      max_new_tokens: Number(options.maxNewTokens || 120),
      do_sample: Boolean(options.doSample),
      temperature: Boolean(options.doSample) ? Number(options.temperature || 0.7) : undefined,
      top_p: Boolean(options.doSample) ? Number(options.topP || 0.92) : undefined,
      repetition_penalty: Number(options.repetitionPenalty || 1.08),
    }),
    Number(options.timeoutMs || 70000),
    "Browser generation timed out before Chronicle could finish the step.",
  );

  const generatedTokens = output.slice(null, [inputLength, null]);
  const decoded = tokenizer.batch_decode(generatedTokens, {
    skip_special_tokens: true,
  });
  return Array.isArray(decoded) ? String(decoded[0] || "") : String(decoded || "");
}

function resolveInputLength(inputIds) {
  if (!inputIds) {
    return 0;
  }
  if (typeof inputIds.dims?.[1] === "number") {
    return inputIds.dims[1];
  }
  if (Array.isArray(inputIds) && Array.isArray(inputIds[0])) {
    return inputIds[0].length;
  }
  if (Array.isArray(inputIds)) {
    return inputIds.length;
  }
  return 0;
}

function withTimeout(promise, timeoutMs, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      self.setTimeout(() => reject(new Error(message)), timeoutMs);
    }),
  ]);
}

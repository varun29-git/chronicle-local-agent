/**
 * Headless Browser Agent Core
 * Powered by Gemma 3n MatFormer via Transformers.js
 */

const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@next";
const MODEL_ID = "onnx-community/gemma-3n-E2B-it-ONNX";

let generatorPipeline = null;
let currentProfile = null;

/**
 * Detects device hardware configuration directly in the browser.
 * Returns the available RAM (GB), CPU Concurrency, and whether WebGPU is supported.
 */
function detectDeviceConfig() {
  const hasWebGPU = typeof navigator !== "undefined" && Boolean(navigator.gpu);
  const deviceMemory = Number(navigator.deviceMemory || 0); // Note: Typically capped at 8 on browsers for fingerprinting
  const hardwareConcurrency = Number(navigator.hardwareConcurrency || 0);

  return {
    hasWebGPU,
    deviceMemory, // GB
    hardwareConcurrency, // Cores
  };
}

/**
 * Mathematically maps the device configuration to the MatFormer percentage slices.
 * Calculates what percentage of the model should be used, returning both the percentage
 * and the calculated `num_slices` mapping (Assuming max 8 slices: 1 = 12.5%, 8 = 100%).
 */
function calculateModelSlice(config) {
  let slicePercentage = 0;

  // Since browsers cap navigator.deviceMemory at 8 for fingerprinting,
  // we combine hardwareConcurrency and GPU capability to approximate flagship limits.
  if (config.hasWebGPU && (config.deviceMemory >= 8 || config.hardwareConcurrency >= 16)) {
    // Flagship laptop with a GPU
    slicePercentage = 100;
  } else if (config.hasWebGPU && config.deviceMemory >= 8) {
    // High-mid
    slicePercentage = 75;
  } else if (config.deviceMemory >= 8 || config.hardwareConcurrency >= 8) {
    // Mid-range mapping
    slicePercentage = 50;
  } else if (config.deviceMemory >= 4) {
    // Medium end mobile/laptop
    slicePercentage = 25;
  } else {
    // Low end mobile phone
    slicePercentage = 12.5;
  }

  // Calculate the direct slice mapping equivalent (assuming 8 maximum slices)
  const totalMaxSlices = 8;
  let num_slices = Math.round((slicePercentage / 100) * totalMaxSlices);
  num_slices = Math.max(1, Math.min(num_slices, totalMaxSlices)); // Clamp between 1 and 8

  return {
    percentage: slicePercentage,
    num_slices: num_slices,
    backend: config.hasWebGPU ? "webgpu" : "wasm",
  };
}

/**
 * Initializes the Transformers.js pipeline with the calculated MatFormer slice configuration.
 */
export async function initializeAgent() {
  if (generatorPipeline) return generatorPipeline;

  const config = detectDeviceConfig();
  const sliceProfile = calculateModelSlice(config);
  currentProfile = sliceProfile;

  console.log(`[Agent] Detected Config: RAM ~${config.deviceMemory}GB, Cores ~${config.hardwareConcurrency}, WebGPU: ${config.hasWebGPU}`);
  console.log(`[Agent] Allocating MatFormer Submodel: ${sliceProfile.percentage}% (Slice ${sliceProfile.num_slices}/8) via ${sliceProfile.backend}`);

  // Dynamically import Transformers.js
  const { AutoModelForImageTextToText, AutoProcessor, env } = await import(TRANSFORMERS_CDN);

  // Configure environment for local browser execution (bypassing HF Token gates)
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = "/models";
  env.useBrowserCache = true;
  
  if (env.backends?.onnx?.wasm && sliceProfile.backend === "wasm") {
    env.backends.onnx.wasm.numThreads = Math.min(4, config.hardwareConcurrency || 2);
  }

  // Initialize the text-generation pipeline with the calculated slices
  const processor = await AutoProcessor.from_pretrained(MODEL_ID);
  const model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
    device: sliceProfile.backend,
    dtype: {
      audio_encoder: "q4",
      vision_encoder: "uint8",
      decoder_model_merged: "q4",
      embed_tokens: "q4",
    },
    model_kwargs: {
      num_slices: sliceProfile.num_slices,
    },
  });
  generatorPipeline = { processor, model };

  console.log(`[Agent] Gemma 3n ready (${sliceProfile.percentage}% slice active).`);
  return generatorPipeline;
}

/**
 * Headless execution entry point for processing a newsletter request.
 */
export async function generateNewsletter(prompt) {
  if (!generatorPipeline) {
    throw new Error("Agent has not been initialized. Call initializeAgent() first.");
  }

  console.log("[Agent] Processing user request locally...");

  // Execute inference on the device seamlessly
  const messages = [
    {
      role: "user",
      content: [
        {
          type: "text",
          text: prompt,
        },
      ],
    },
  ];
  const formattedPrompt = generatorPipeline.processor.apply_chat_template(messages, {
    add_generation_prompt: true,
  });
  const inputs = await generatorPipeline.processor(formattedPrompt, null, null, {
    add_special_tokens: false,
  });
  const inputLength = inputs.input_ids?.dims?.at(-1);
  const output = await generatorPipeline.model.generate({
    ...inputs,
    max_new_tokens: 1500, // Reasonable cap for newsletter bodies
    do_sample: false, // Greedy decoding for logical newsletters
  });

  const generatedText = generatorPipeline.processor.batch_decode(
    output.slice(null, [inputLength, null]),
    { skip_special_tokens: true },
  )[0];

  return {
    status: "success",
    metrics: {
      processedWithSlice: currentProfile.percentage + "%",
      backend: currentProfile.backend,
    },
    markdown: generatedText.trim(),
  };
}

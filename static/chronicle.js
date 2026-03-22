const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0";

const state = {
  runs: [],
  selectedRunId: null,
  activeJobId: null,
  pollTimer: null,
  serverRuntime: null,
  browserConfig: null,
  browserCapabilities: null,
  browserSession: null,
};

const elements = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindSegments();
  bindComposer();
  hydrateApp().catch((error) => {
    console.error(error);
    renderJobError(error.message || "Chronicle failed to initialize.");
  });
});

function cacheElements() {
  elements.runtimeBadge = document.getElementById("runtime-badge");
  elements.runtimeHeadline = document.getElementById("runtime-headline");
  elements.runtimeSubline = document.getElementById("runtime-subline");
  elements.browserAiBadge = document.getElementById("browser-ai-badge");
  elements.browserAiCopy = document.getElementById("browser-ai-copy");
  elements.statBackend = document.getElementById("stat-backend");
  elements.statSlice = document.getElementById("stat-slice");
  elements.statMemory = document.getElementById("stat-memory");
  elements.statModelReady = document.getElementById("stat-model-ready");
  elements.detailHostname = document.getElementById("detail-hostname");
  elements.detailPlatform = document.getElementById("detail-platform");
  elements.detailDeviceClass = document.getElementById("detail-device-class");
  elements.detailModelPath = document.getElementById("detail-model-path");
  elements.detailModelRoot = document.getElementById("detail-model-root");

  elements.composerForm = document.getElementById("composer-form");
  elements.generateButton = document.getElementById("generate-button");
  elements.customStyleField = document.getElementById("custom-style-field");

  elements.jobBadge = document.getElementById("job-badge");
  elements.jobMessage = document.getElementById("job-message");
  elements.jobSubcopy = document.getElementById("job-subcopy");
  elements.jobRunTitle = document.getElementById("job-run-title");
  elements.jobRunTime = document.getElementById("job-run-time");
  elements.jobOpenHtml = document.getElementById("job-open-html");
  elements.jobOpenMd = document.getElementById("job-open-md");

  elements.libraryCount = document.getElementById("library-count");
  elements.runList = document.getElementById("run-list");

  elements.previewTitle = document.getElementById("preview-title");
  elements.previewMode = document.getElementById("preview-mode");
  elements.previewFrame = document.getElementById("preview-frame");
  elements.previewOpenHtml = document.getElementById("preview-open-html");
  elements.previewOpenMd = document.getElementById("preview-open-md");
}

function bindSegments() {
  document.querySelectorAll("[data-segment]").forEach((segmentGroup) => {
    const hiddenInput = segmentGroup.parentElement.querySelector('input[type="hidden"]');
    segmentGroup.querySelectorAll(".segment").forEach((button) => {
      button.addEventListener("click", () => {
        segmentGroup.querySelectorAll(".segment").forEach((segment) => {
          segment.classList.remove("is-active");
        });
        button.classList.add("is-active");
        hiddenInput.value = button.dataset.value;
        if (hiddenInput.name === "explanation_style") {
          toggleCustomStyleField(button.dataset.value === "custom");
        }
      });
    });
  });
}

function bindComposer() {
  elements.composerForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const payload = buildPayload();
    setGenerateBusy(true);
    stopPolling();

    try {
      await runBrowserGeneration(payload);
    } catch (error) {
      console.error(error);
      if (canUseServerFallback()) {
        renderJobState({
          status: "running",
          message: "Browser AI unavailable. Falling back to the Chronicle runtime on this device.",
        });
        await startServerGeneration(payload);
      } else {
        renderJobError(error.message || "Chronicle could not generate the issue on this device.");
        setGenerateBusy(false);
      }
    }
  });
}

function buildPayload() {
  const formData = new FormData(elements.composerForm);
  return {
    brief: formData.get("brief"),
    depth: formData.get("depth"),
    explanation_style: formData.get("explanation_style"),
    style_instructions: formData.get("style_instructions") || "",
    days: Number(formData.get("days")),
    queries: Number(formData.get("queries")),
    results_per_query: Number(formData.get("results_per_query")),
    device_class: formData.get("device_class") || "",
  };
}

async function hydrateApp() {
  toggleCustomStyleField(false);
  const [statusResponse, runsResponse] = await Promise.all([
    fetchJSON("/api/status"),
    fetchJSON("/api/runs?limit=30"),
  ]);

  state.serverRuntime = statusResponse.runtime;
  state.browserConfig = statusResponse.browser_ai;
  state.browserCapabilities = detectBrowserCapabilities(statusResponse.browser_ai);

  renderRuntime(statusResponse.runtime, statusResponse.browser_ai, state.browserCapabilities);
  renderRuns(runsResponse.runs || []);

  if (statusResponse.active_job) {
    state.activeJobId = statusResponse.active_job.id;
    renderJobState(statusResponse.active_job);
    startPolling(statusResponse.active_job.id);
  } else {
    renderJobIdle();
  }
}

function detectBrowserCapabilities(browserConfig) {
  const hasWebGPU = typeof navigator !== "undefined" && Boolean(navigator.gpu);
  const deviceMemory = Number(navigator.deviceMemory || 0);
  const hardwareConcurrency = Number(navigator.hardwareConcurrency || 0);
  const highTier = hasWebGPU && deviceMemory >= 8 && hardwareConcurrency >= 8;
  const balancedTier = hasWebGPU;

  const candidates = [];
  if (highTier) {
    candidates.push({
      device: "webgpu",
      model: browserConfig.high_tier_model,
      label: "WebGPU high tier",
      maxNewTokens: 900,
      temperature: 0.2,
    });
  }
  if (balancedTier) {
    candidates.push({
      device: "webgpu",
      model: browserConfig.low_tier_model,
      label: highTier ? "WebGPU balanced fallback" : "WebGPU balanced",
      maxNewTokens: 640,
      temperature: 0.2,
    });
  }
  candidates.push({
    device: "wasm",
    model: browserConfig.cpu_fallback_model,
    label: hasWebGPU ? "WASM CPU fallback" : "WASM universal fallback",
    maxNewTokens: 420,
    temperature: 0.25,
  });

  return {
    hasWebGPU,
    deviceMemory,
    hardwareConcurrency,
    candidates,
  };
}

function renderRuntime(runtime, browserConfig, browserCapabilities) {
  const dependenciesReady = Boolean(runtime.dependencies_ready);
  const modelReady = Boolean(runtime.model_ready);
  if (!dependenciesReady) {
    setBadge(elements.runtimeBadge, "Deps missing", "is-danger");
  } else if (!modelReady) {
    setBadge(elements.runtimeBadge, "Model missing", "is-warm");
  } else {
    setBadge(elements.runtimeBadge, "Server ready", "");
  }

  elements.runtimeHeadline.textContent =
    `Chronicle is using ${runtime.hostname} as the local device anchor.`;
  if (!dependenciesReady) {
    elements.runtimeSubline.textContent =
      `${runtime.dependency_message}. The browser AI path can still run if the user’s browser supports it, but native runtime fallback on this machine is not ready yet.`;
  } else if (!modelReady) {
    elements.runtimeSubline.textContent =
      "The server-side local runtime is alive, but the configured local model path is missing. Browser AI can still run independently when WebGPU or WASM is available.";
  } else {
    elements.runtimeSubline.textContent =
      "Both Chronicle modes are visible here: browser AI for cross-device inference and the native local runtime as a fallback path.";
  }

  renderBrowserAiStrip(browserConfig, browserCapabilities);

  elements.statBackend.textContent = browserCapabilities.hasWebGPU ? "WEBGPU" : "WASM";
  elements.statSlice.textContent = runtime.slice_label || "—";
  elements.statMemory.textContent =
    browserCapabilities.deviceMemory > 0
      ? `${browserCapabilities.deviceMemory.toFixed(1)} GB`
      : runtime.memory_total_gb
        ? `${runtime.memory_total_gb.toFixed(1)} GB`
        : "Unknown";
  elements.statModelReady.textContent = modelReady ? "Fallback ready" : "Fallback limited";

  elements.detailHostname.textContent = runtime.hostname || "This device";
  elements.detailPlatform.textContent = [
    runtime.system_name,
    runtime.machine,
    runtime.chip || runtime.hardware_model || runtime.gpu_model,
  ]
    .filter(Boolean)
    .join(" • ");
  elements.detailDeviceClass.textContent = runtime.device_class || "Auto";
  elements.detailModelPath.textContent = runtime.model_path || "—";
  elements.detailModelRoot.textContent = runtime.local_model_root || "—";
}

function renderBrowserAiStrip(browserConfig, browserCapabilities) {
  const firstCandidate = browserCapabilities.candidates[0];
  if (browserCapabilities.hasWebGPU) {
    setBadge(elements.browserAiBadge, "WebGPU ready", "");
    elements.browserAiCopy.textContent =
      `Chronicle will try ${firstCandidate.model} in the browser with WebGPU first, then fall back through lighter local paths if the device cannot hold that tier.`;
  } else {
    setBadge(elements.browserAiBadge, "WASM fallback", "is-warm");
    elements.browserAiCopy.textContent =
      `WebGPU is not available in this browser, so Chronicle will fall back to ${browserConfig.cpu_fallback_model} with browser-side WASM execution. It should still run, but slower.`;
  }
}

async function runBrowserGeneration(payload) {
  renderJobState({
    status: "running",
    message: "Collecting research through Chronicle",
  });
  const researchResponse = await fetchJSON("/api/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const aiSession = await ensureBrowserSession();
  renderJobState({
    status: "running",
    message: `Drafting with ${aiSession.profile.label} on this device`,
  });

  const markdown = await generateNewsletterMarkdown(researchResponse.research, aiSession);
  const normalizedMarkdown = stripMarkdownFences(markdown);
  const title = extractTitleFromMarkdown(normalizedMarkdown, researchResponse.research.plan.title);

  renderJobState({
    status: "running",
    message: "Saving the Chronicle issue",
  });
  const saveResponse = await fetchJSON("/api/runs/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      brief: payload.brief,
      title,
      markdown: normalizedMarkdown,
      depth: payload.depth,
      explanation_style: payload.explanation_style,
      style_instructions: payload.style_instructions,
      audience: researchResponse.research.plan.audience,
      tone: researchResponse.research.plan.tone,
      queries: researchResponse.research.plan.queries,
      sections: researchResponse.research.plan.sections,
      sources: researchResponse.research.sources.map((source) => ({
        ...source,
        source_summary: source.source_text,
        relevance_score: 0.5,
      })),
    }),
  });

  await syncAfterBrowserSave(saveResponse.run);
  setGenerateBusy(false);
  renderJobState({
    status: "completed",
    result: saveResponse.run,
  });
}

async function ensureBrowserSession() {
  if (state.browserSession?.generator) {
    return state.browserSession;
  }

  const { pipeline, env } = await import(TRANSFORMERS_CDN);
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  if (env.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
  }

  let lastError = null;
  for (const candidate of state.browserCapabilities.candidates) {
    try {
      renderJobState({
        status: "running",
        message: `Loading ${candidate.model} with ${candidate.device.toUpperCase()} on this device`,
      });
      const generator = await pipeline("text-generation", candidate.model, {
        device: candidate.device,
      });
      state.browserSession = {
        generator,
        profile: candidate,
      };
      return state.browserSession;
    } catch (error) {
      console.warn(`Chronicle browser candidate failed: ${candidate.model}`, error);
      lastError = error;
    }
  }

  throw new Error(lastError?.message || "No browser inference backend could be initialized.");
}

async function generateNewsletterMarkdown(research, aiSession) {
  const prompt = buildNewsletterPrompt(research);
  const output = await aiSession.generator(prompt, {
    max_new_tokens: aiSession.profile.maxNewTokens,
    do_sample: false,
    temperature: aiSession.profile.temperature,
    return_full_text: false,
  });

  if (Array.isArray(output) && output[0]?.generated_text) {
    return output[0].generated_text;
  }
  if (typeof output === "string") {
    return output;
  }
  if (output?.generated_text) {
    return output.generated_text;
  }
  throw new Error("Browser model returned an unexpected output shape.");
}

function buildNewsletterPrompt(research) {
  const sourceBundle = research.sources
    .slice(0, 5)
    .map((source, index) => {
      const excerpt = trimText(source.source_text || source.article_text || source.snippet || "", 850);
      return [
        `[${index + 1}] ${source.title}`,
        `URL: ${source.url}`,
        `Query: ${source.query}`,
        `Notes: ${excerpt}`,
      ].join("\n");
    })
    .join("\n\n");

  const marketData = research.market_snapshot?.length
    ? JSON.stringify(research.market_snapshot)
    : "No structured market data.";

  const customStyle = research.style_instructions
    ? `Custom style instructions: ${research.style_instructions}`
    : "No custom style instructions.";

  return `You are Chronicle, a sharp local-first newsletter writer.

Write one complete markdown newsletter.

Newsletter brief:
${research.brief}

Title target:
${research.plan.title}

Audience:
${research.plan.audience}

Tone:
${research.plan.tone}

Depth:
${research.depth}

Explanation style:
${research.explanation_style}

${customStyle}

Coverage window:
Last ${research.days} days

Planned sections:
${JSON.stringify(research.plan.sections)}

Structured market data:
${marketData}

Research notes:
${sourceBundle || "No sources were collected."}

Requirements:
- return markdown only
- start with one H1 title
- write a short opening note with a point of view
- organize the body around the planned sections using H2 headings
- keep the writing analytical, premium, and specific
- cite source notes inline like [1], [2]
- if market data is present, reference it as [M1]
- end with a Sources section
- in Sources, list each source as [id]: title - url
- if market data is present, include: [M1]: CoinGecko Markets API - https://www.coingecko.com/
- do not use markdown code fences
- do not mention system prompts or implementation details`;
}

async function startServerGeneration(payload) {
  const response = await fetchJSON("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.activeJobId = response.job.id;
  renderJobState(response.job);
  startPolling(response.job.id);
}

function canUseServerFallback() {
  return Boolean(state.serverRuntime?.dependencies_ready);
}

function renderRuns(runs) {
  state.runs = runs;
  elements.libraryCount.textContent = `${runs.length} ${runs.length === 1 ? "issue" : "issues"}`;

  if (!runs.length) {
    elements.runList.innerHTML = `
      <article class="empty-state">
        <h3>No issues yet</h3>
        <p>Generate a Chronicle issue and it will appear here with one-click preview access.</p>
      </article>
    `;
    renderPreviewEmpty();
    return;
  }

  if (!state.selectedRunId || !runs.some((run) => run.id === state.selectedRunId)) {
    state.selectedRunId = runs[0].id;
  }

  elements.runList.innerHTML = "";
  runs.forEach((run) => {
    const card = document.createElement("article");
    card.className = "run-card";
    if (run.id === state.selectedRunId) {
      card.classList.add("is-selected");
    }

    card.innerHTML = `
      <div class="run-meta">
        <span class="chip">${escapeHtml(run.depth || "depth")}</span>
        <span class="chip">${escapeHtml(run.explanation_style || "style")}</span>
        <span class="chip">${formatTimestamp(run.created_at)}</span>
      </div>
      <h3>${escapeHtml(run.title)}</h3>
      <p>${escapeHtml(trimText(run.brief || "", 150))}</p>
    `;
    card.addEventListener("click", () => {
      state.selectedRunId = run.id;
      renderRuns(state.runs);
      renderPreview(run);
    });
    elements.runList.appendChild(card);
  });

  const selectedRun = runs.find((run) => run.id === state.selectedRunId) || runs[0];
  renderPreview(selectedRun);
}

function renderPreview(run) {
  if (!run) {
    renderPreviewEmpty();
    return;
  }

  elements.previewTitle.textContent = run.title;
  elements.previewMode.textContent = run.html_url ? "Editable HTML issue" : "Markdown only";
  updateActionLink(elements.previewOpenHtml, run.html_url);
  updateActionLink(elements.previewOpenMd, run.markdown_url);

  if (run.preview_url) {
    elements.previewFrame.removeAttribute("srcdoc");
    elements.previewFrame.src = run.preview_url;
  } else {
    elements.previewFrame.removeAttribute("src");
    elements.previewFrame.srcdoc = `
      <p style="font-family: sans-serif; color: #ddd; padding: 2rem;">
        This issue does not have an editable HTML preview yet.
      </p>
    `;
  }
}

function renderPreviewEmpty() {
  elements.previewTitle.textContent = "Choose an issue";
  elements.previewMode.textContent = "HTML editor";
  updateActionLink(elements.previewOpenHtml, "");
  updateActionLink(elements.previewOpenMd, "");
  elements.previewFrame.removeAttribute("src");
  elements.previewFrame.srcdoc = `
    <p style="font-family: sans-serif; color: #ddd; padding: 2rem;">
      Select an issue to preview it here.
    </p>
  `;
}

function renderJobIdle() {
  setGenerateBusy(false);
  setBadge(elements.jobBadge, "Idle", "");
  elements.jobMessage.textContent = "No Chronicle issue is running right now.";
  elements.jobSubcopy.textContent =
    "When a run starts, Chronicle will research through the backend, draft on the user’s device, and then save the finished issue back to the library.";
  updateActionLink(elements.jobOpenHtml, "");
  updateActionLink(elements.jobOpenMd, "");

  const latestRun = state.runs[0];
  elements.jobRunTitle.textContent = latestRun ? latestRun.title : "—";
  elements.jobRunTime.textContent = latestRun ? formatTimestamp(latestRun.created_at) : "—";
}

function renderJobState(job) {
  const status = job.status || "running";
  if (status === "failed") {
    renderJobError(job.error?.message || job.message || "Chronicle failed to finish the run.");
    return;
  }

  if (status === "completed") {
    const result = job.result || null;
    setGenerateBusy(false);
    setBadge(elements.jobBadge, "Ready", "");
    elements.jobMessage.textContent = result
      ? `Chronicle finished “${result.title}” on the user’s device.`
      : "Chronicle finished the current run.";
    elements.jobSubcopy.textContent =
      "Open the editable HTML issue or switch to the raw markdown source if you want the plain document.";
    if (result) {
      elements.jobRunTitle.textContent = result.title;
      elements.jobRunTime.textContent = formatTimestamp(result.created_at);
      updateActionLink(elements.jobOpenHtml, result.html_url);
      updateActionLink(elements.jobOpenMd, result.markdown_url);
      state.selectedRunId = result.id;
    }
    return;
  }

  setGenerateBusy(true);
  setBadge(elements.jobBadge, upper(status), "is-warm");
  elements.jobMessage.textContent = job.message || "Chronicle is running locally.";
  elements.jobSubcopy.textContent =
    "The browser model is doing the generation work on the user’s device. Chronicle only uses the backend for research collection and saving outputs.";
}

function renderJobError(message) {
  setGenerateBusy(false);
  stopPolling();
  setBadge(elements.jobBadge, "Failed", "is-danger");
  elements.jobMessage.textContent = message;
  elements.jobSubcopy.textContent =
    "If WebGPU is unavailable, Chronicle will try WASM. If the browser path still fails, enable the native runtime fallback on the machine hosting the site.";
}

function setGenerateBusy(isBusy) {
  elements.generateButton.disabled = isBusy;
  elements.generateButton.textContent = isBusy
    ? "Generating on this device…"
    : "Generate with browser AI";
}

function toggleCustomStyleField(visible) {
  elements.customStyleField.classList.toggle("is-hidden", !visible);
}

function updateActionLink(link, url) {
  if (url) {
    link.href = url;
    link.classList.remove("is-disabled");
  } else {
    link.href = "#";
    link.classList.add("is-disabled");
  }
}

function setBadge(element, text, modifierClass) {
  element.textContent = text;
  element.classList.remove("is-warm", "is-danger");
  if (modifierClass) {
    element.classList.add(modifierClass);
  }
}

function startPolling(jobId) {
  state.activeJobId = jobId;
  stopPolling();
  state.pollTimer = window.setInterval(async () => {
    try {
      const response = await fetchJSON(`/api/jobs/${jobId}`);
      const job = response.job;
      renderJobState(job);
      if (job.status === "completed" || job.status === "failed") {
        stopPolling();
        await syncAfterJob(job);
      }
    } catch (error) {
      stopPolling();
      renderJobError(error.message || "Chronicle lost contact with the local server.");
    }
  }, 1800);
}

function stopPolling() {
  if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function syncAfterBrowserSave(run) {
  const [statusResponse, runsResponse] = await Promise.all([
    fetchJSON("/api/status"),
    fetchJSON("/api/runs?limit=30"),
  ]);
  state.serverRuntime = statusResponse.runtime;
  state.browserConfig = statusResponse.browser_ai;
  state.browserCapabilities = detectBrowserCapabilities(statusResponse.browser_ai);
  renderRuntime(statusResponse.runtime, statusResponse.browser_ai, state.browserCapabilities);
  renderRuns(runsResponse.runs || []);
  if (run) {
    state.selectedRunId = run.id;
    const selectedRun = (runsResponse.runs || []).find((item) => item.id === run.id);
    if (selectedRun) {
      renderPreview(selectedRun);
    }
  }
}

async function syncAfterJob(job) {
  const [statusResponse, runsResponse] = await Promise.all([
    fetchJSON("/api/status"),
    fetchJSON("/api/runs?limit=30"),
  ]);
  state.serverRuntime = statusResponse.runtime;
  state.browserConfig = statusResponse.browser_ai;
  state.browserCapabilities = detectBrowserCapabilities(statusResponse.browser_ai);
  renderRuntime(statusResponse.runtime, statusResponse.browser_ai, state.browserCapabilities);
  renderRuns(runsResponse.runs || []);
  if (job.result) {
    state.selectedRunId = job.result.id;
    const selectedRun = (runsResponse.runs || []).find((run) => run.id === job.result.id);
    if (selectedRun) {
      renderPreview(selectedRun);
    }
  }
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  let data = {};
  try {
    data = await response.json();
  } catch (error) {
    data = {};
  }
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function stripMarkdownFences(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed.startsWith("```")) {
    return trimmed;
  }
  return trimmed.replace(/^```[a-zA-Z0-9_-]*\s*/, "").replace(/\s*```$/, "").trim();
}

function extractTitleFromMarkdown(markdown, fallbackTitle) {
  const match = markdown.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : fallbackTitle;
}

function trimText(text, maximumLength) {
  if (text.length <= maximumLength) {
    return text;
  }
  return `${text.slice(0, maximumLength - 1).trim()}…`;
}

function formatTimestamp(value) {
  if (!value) {
    return "—";
  }
  const date = new Date(value.replace(" ", "T"));
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function upper(value) {
  return value ? String(value).toUpperCase() : "—";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

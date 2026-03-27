const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@next";

const state = {
  serverRuntime: null,
  browserConfig: null,
  browserCapabilities: null,
  browserSession: null,
  activeJobId: null,
  pollTimer: null,
  currentTurn: null,
  hasRenderedIntro: false,
};

const elements = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindComposer();
  hydrateApp().catch((error) => {
    console.error(error);
    appendSystemMessage(error.message || "Chronicle failed to initialize.");
  });
});

function cacheElements() {
  elements.statusPill = document.getElementById("status-pill");
  elements.statusCopy = document.getElementById("status-copy");
  elements.chatThread = document.getElementById("chat-thread");
  elements.composerForm = document.getElementById("composer-form");
  elements.brief = document.getElementById("brief");
  elements.explanationStyle = document.getElementById("explanation-style");
  elements.customStyleField = document.getElementById("custom-style-field");
  elements.generateButton = document.getElementById("generate-button");
}

function bindComposer() {
  elements.explanationStyle.addEventListener("change", () => {
    toggleCustomStyleField(elements.explanationStyle.value === "custom");
  });

  elements.composerForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = buildPayload();
    if (!payload.brief) {
      return;
    }

    stopPolling();
    startNewTurn(payload.brief);
    setGenerateBusy(true);

    try {
      await runBrowserGeneration(payload);
    } catch (error) {
      console.error(error);
      appendLogLines([
        `Browser runtime failed: ${error.message || "Unknown error"}`,
      ]);

      if (canUseServerFallback()) {
        updateTurnStatus("Browser runtime failed. Falling back to the host runtime.");
        await startServerGeneration(payload);
      } else {
        finishTurnWithError(error.message || "Chronicle could not generate the issue on this device.");
      }
    }
  });
}

function buildPayload() {
  const formData = new FormData(elements.composerForm);
  return {
    brief: String(formData.get("brief") || "").trim(),
    depth: String(formData.get("depth") || "medium").trim(),
    explanation_style: String(formData.get("explanation_style") || "concise").trim(),
    style_instructions: String(formData.get("style_instructions") || "").trim(),
    days: Number(formData.get("days") || 7),
  };
}

async function hydrateApp() {
  toggleCustomStyleField(false);
  const statusResponse = await fetchJSON("/api/status");
  applyStatus(statusResponse);

  if (!state.hasRenderedIntro) {
    appendAssistantMessage(buildIntroMessage());
    state.hasRenderedIntro = true;
  }

  if (statusResponse.active_job) {
    startRecoveredTurn();
    setGenerateBusy(true);
    renderJobState(statusResponse.active_job);
    startPolling(statusResponse.active_job.id);
  }
}

function applyStatus(statusResponse) {
  state.serverRuntime = statusResponse.runtime;
  state.browserConfig = statusResponse.browser_ai;
  state.browserCapabilities = detectBrowserCapabilities(statusResponse.browser_ai);
  renderHeaderStatus();
}

function renderHeaderStatus() {
  const runtime = state.serverRuntime;
  const browserConfig = state.browserConfig;
  const browserCapabilities = state.browserCapabilities;
  const browserReady = Boolean(browserConfig?.local_model_ready);
  const supportsSlicing = Boolean(browserConfig?.supports_slicing);
  const primaryProfile = browserCapabilities?.candidates?.[0];
  const modelName = browserConfig?.display_name || "Gemma 3n adaptive";

  if (browserReady) {
    setStatusPill(browserCapabilities?.hasWebGPU ? "Browser ready" : "WASM ready", "");
  } else if (runtime?.dependencies_ready && runtime?.model_ready) {
    setStatusPill("Server fallback only", "is-warm");
  } else {
    setStatusPill("Runtime incomplete", "is-danger");
  }

  if (browserReady && primaryProfile) {
    const runtimeMode = browserCapabilities.hasWebGPU ? "WebGPU" : "WASM";
    const sliceText = supportsSlicing ? primaryProfile.sliceLabel : "single local bundle";
    elements.statusCopy.textContent =
      `${modelName} is available locally. Chronicle will use ${runtimeMode} and target ${sliceText} on this device.`;
  } else if (runtime?.dependencies_ready && runtime?.model_ready) {
    elements.statusCopy.textContent =
      "The local browser model is unavailable, but the host runtime can still generate newsletters here.";
  } else {
    elements.statusCopy.textContent =
      "Chronicle could not find a complete local runtime path yet. Add the local model bundle or host fallback model to continue.";
  }
}

function buildIntroMessage() {
  const browserConfig = state.browserConfig;
  const browserCapabilities = state.browserCapabilities;
  const modelName = browserConfig?.display_name || "Gemma 3n adaptive";

  if (browserConfig?.local_model_ready) {
    const firstCandidate = browserCapabilities?.candidates?.[0];
    return [
      "Chronicle is ready.",
      `Local model: ${modelName}`,
      firstCandidate ? `Current device target: ${firstCandidate.label}` : "",
      "Send a brief and I’ll show planning, evidence selection, drafting, and the full search log while I work.",
    ]
      .filter(Boolean)
      .join("\n");
  }

  if (state.serverRuntime?.dependencies_ready && state.serverRuntime?.model_ready) {
    return "Chronicle is ready through the host runtime. Send a brief and I’ll show the search and drafting log in the chat.";
  }

  return "Chronicle is online, but the local runtime still needs a complete model path before generation can succeed.";
}

function startNewTurn(userPrompt) {
  appendUserMessage(userPrompt);
  elements.brief.value = "";
  state.currentTurn = {
    statusNode: appendAssistantMessage("Starting research…"),
    logContainer: appendLogCard(),
    displayedJobLogCount: 0,
    resultNode: null,
    completed: false,
  };
}

function startRecoveredTurn() {
  state.currentTurn = {
    statusNode: appendAssistantMessage("Resuming the current local run…"),
    logContainer: appendLogCard(),
    displayedJobLogCount: 0,
    resultNode: null,
    completed: false,
  };
}

function updateTurnStatus(text) {
  const turn = ensureCurrentTurn();
  const body = turn.statusNode.querySelector(".message-body");
  body.textContent = text;
  scrollThreadToBottom();
}

function appendLogLines(lines) {
  const turn = ensureCurrentTurn();
  const fragment = document.createDocumentFragment();
  (lines || []).forEach((line) => {
    const text = String(line || "").trim();
    if (!text) {
      return;
    }
    const row = document.createElement("div");
    row.className = "log-line";
    row.textContent = text;
    fragment.appendChild(row);
  });
  turn.logContainer.appendChild(fragment);
  scrollThreadToBottom();
}

function appendResultCard(run, markdown) {
  const turn = ensureCurrentTurn();
  if (turn.completed) {
    return;
  }

  const visibleText = stripMarkdownFences(markdown || "").trim() || run?.title || "Newsletter ready.";
  const card = ensureResultNode();
  card.querySelector(".message-body").textContent = visibleText;
  card.querySelector(".result-actions").innerHTML = `
    ${run?.html_url ? `<a class="result-link" href="${escapeHtml(run.html_url)}" target="_blank" rel="noreferrer">Open HTML</a>` : ""}
    ${run?.markdown_url ? `<a class="result-link" href="${escapeHtml(run.markdown_url)}" target="_blank" rel="noreferrer">Open markdown</a>` : ""}
  `;
  turn.completed = true;
  scrollThreadToBottom();
}

function ensureResultNode() {
  const turn = ensureCurrentTurn();
  if (turn.resultNode) {
    return turn.resultNode;
  }

  const card = document.createElement("article");
  card.className = "message message--assistant";
  card.innerHTML = `
    <div class="message-card result-card">
      <p class="message-label">Chronicle</p>
      <div class="message-body"></div>
      <div class="result-actions"></div>
    </div>
  `;
  elements.chatThread.appendChild(card);
  turn.resultNode = card;
  scrollThreadToBottom();
  return card;
}

function updateLiveDraft(text) {
  const card = ensureResultNode();
  card.querySelector(".message-body").textContent = text || "Drafting…";
  scrollThreadToBottom();
}

function finishTurnWithError(message) {
  setGenerateBusy(false);
  stopPolling();
  updateTurnStatus("Run failed.");
  appendSystemMessage(message);
}

async function runBrowserGeneration(payload) {
  updateTurnStatus("Planning and collecting research…");
  const researchResponse = await fetchJSON("/api/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const research = researchResponse.research;
  appendLogLines(research.logs || buildResearchLogs(research));
  const packet = buildEditorialPacket(research);
  appendAssistantMessage(buildReasoningSummary(packet), "Reasoning summary");
  appendLogLines([
    `Evidence selected: ${packet.selectedSources.length} unique source(s)`,
    `Working thesis: ${packet.workingThesis}`,
    `Coverage gaps: ${packet.coverageGap}`,
  ]);

  let markdown = "";
  let usedFallbackDraft = false;

  try {
    const aiSession = await ensureBrowserSession();
    updateTurnStatus(`Drafting locally with ${aiSession.profile.label}…`);
    appendLogLines([
      `Loading local model: ${aiSession.profile.label}`,
      `Generation backend: ${aiSession.profile.device.toUpperCase()}`,
      `Generation budget: ${Math.round(aiSession.profile.generationTimeoutMs / 1000)}s`,
    ]);

    updateLiveDraft("Drafting locally from the selected evidence…");
    markdown = await generateNewsletterMarkdown(research, packet, aiSession, (partialText) => {
      if (partialText.trim()) {
        updateLiveDraft(partialText.trim());
      }
    });
  } catch (error) {
    appendLogLines([
      `Model drafting failed: ${error.message || "Unknown error"}`,
      "Switching to deterministic drafting from the collected evidence.",
    ]);
    markdown = renderFallbackNewsletter(packet);
    usedFallbackDraft = true;
  }

  const normalizedMarkdown = finalizeNewsletterMarkdown(
    stripMarkdownFences(markdown),
    packet,
  );
  const title = extractTitleFromMarkdown(normalizedMarkdown, packet.title);

  updateTurnStatus("Saving the newsletter…");
  appendLogLines([
    usedFallbackDraft
      ? "Deterministic draft complete. Saving issue files…"
      : "Model draft complete. Saving issue files…",
  ]);

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
      audience: packet.audience,
      tone: packet.tone,
      queries: packet.queries,
      sections: packet.sections,
      sources: packet.selectedSources.map((source) => ({
        ...source,
        source_summary: source.sourceText,
        relevance_score: source.relevanceScore,
      })),
    }),
  });

  setGenerateBusy(false);
  updateTurnStatus(
    usedFallbackDraft
      ? `Issue ready: ${saveResponse.run.title} (deterministic draft)`
      : `Issue ready: ${saveResponse.run.title}`,
  );
  appendLogLines([
    `Saved HTML: ${saveResponse.run.html_path || "Unavailable"}`,
    `Saved markdown: ${saveResponse.run.markdown_path || "Unavailable"}`,
  ]);
  appendResultCard(saveResponse.run, normalizedMarkdown);
  await refreshStatus();
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

function renderJobState(job) {
  if (!job) {
    return;
  }

  if (job.status === "queued" || job.status === "running") {
    setGenerateBusy(true);
  }
  updateTurnStatus(job.message || "Chronicle is working…");
  appendJobLogs(job);

  if (job.status === "failed") {
    finishTurnWithError(job.error?.message || job.message || "Chronicle failed to finish the run.");
    return;
  }

  if (job.status === "completed") {
    setGenerateBusy(false);
    stopPolling();
    if (job.result) {
      appendResultCard(job.result, "");
    }
    refreshStatus().catch(console.error);
  }
}

function appendJobLogs(job) {
  const turn = ensureCurrentTurn();
  const logs = job.logs || [];
  const newLines = logs.slice(turn.displayedJobLogCount);
  if (newLines.length) {
    appendLogLines(newLines);
    turn.displayedJobLogCount = logs.length;
  }
}

function startPolling(jobId) {
  state.activeJobId = jobId;
  stopPolling();
  state.pollTimer = window.setInterval(async () => {
    try {
      const response = await fetchJSON(`/api/jobs/${jobId}`);
      renderJobState(response.job);
    } catch (error) {
      finishTurnWithError(error.message || "Chronicle lost contact with the local server.");
    }
  }, 1500);
}

function stopPolling() {
  if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function refreshStatus() {
  const statusResponse = await fetchJSON("/api/status");
  applyStatus(statusResponse);
}

function canUseServerFallback() {
  return Boolean(state.serverRuntime?.dependencies_ready && state.serverRuntime?.model_ready);
}

function ensureCurrentTurn() {
  if (!state.currentTurn) {
    startRecoveredTurn();
  }
  return state.currentTurn;
}

function appendUserMessage(text) {
  return appendMessage("user", text);
}

function appendAssistantMessage(text, label = "") {
  return appendMessage("assistant", text, label);
}

function appendSystemMessage(text, label = "") {
  return appendMessage("system", text, label);
}

function appendMessage(role, text, label = "") {
  const article = document.createElement("article");
  article.className = `message message--${role}`;
  article.innerHTML = `
    <div class="message-card">
      ${label ? `<p class="message-label">${escapeHtml(label)}</p>` : ""}
      <div class="message-body"></div>
    </div>
  `;
  article.querySelector(".message-body").textContent = text;
  elements.chatThread.appendChild(article);
  scrollThreadToBottom();
  return article;
}

function appendLogCard() {
  const article = document.createElement("article");
  article.className = "message message--assistant";
  article.innerHTML = `
    <div class="message-card log-card">
      <p class="message-label">Search log</p>
      <div class="log-list"></div>
    </div>
  `;
  elements.chatThread.appendChild(article);
  scrollThreadToBottom();
  return article.querySelector(".log-list");
}

function setGenerateBusy(isBusy) {
  elements.generateButton.disabled = isBusy;
  elements.generateButton.textContent = isBusy ? "Working…" : "Send";
}

function toggleCustomStyleField(visible) {
  elements.customStyleField.classList.toggle("is-hidden", !visible);
}

function setStatusPill(text, modifierClass) {
  elements.statusPill.textContent = text;
  elements.statusPill.classList.remove("is-warm", "is-danger");
  if (modifierClass) {
    elements.statusPill.classList.add(modifierClass);
  }
}

function scrollThreadToBottom() {
  elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
}

function buildResearchLogs(research) {
  const logs = [
    "Planning complete.",
    `Title: ${research.plan?.title || "Untitled"}`,
  ];
  (research.plan?.queries || []).forEach((query) => {
    logs.push(`Searching: ${query}`);
  });
  return logs;
}

function buildEditorialPacket(research) {
  const selectedSources = selectTopSources(research.sources || [], 5);
  const themeTerms = extractThemeTerms(selectedSources, 6);
  const keyEvidence = selectedSources
    .slice(0, 3)
    .map((source) => cleanupSourceTitle(source.title));
  const workingThesis = buildWorkingThesis(research, selectedSources, themeTerms);
  const coverageGap = buildCoverageGap(selectedSources);

  return {
    brief: research.brief,
    title: research.plan?.title || "Newsletter",
    audience: research.plan?.audience || "General readers",
    tone: research.plan?.tone || "Sharp and analytical",
    days: research.days,
    depth: research.depth,
    explanationStyle: research.explanation_style,
    styleInstructions: research.style_instructions || "",
    queries: research.plan?.queries || [],
    sections: research.plan?.sections || ["What happened", "Why it matters", "What to watch next"],
    marketSnapshot: research.market_snapshot || [],
    selectedSources,
    themeTerms,
    keyEvidence,
    workingThesis,
    coverageGap,
  };
}

function selectTopSources(sources, limit) {
  const deduped = [];
  const seen = new Set();

  sources
    .map((source, index) => ({
      ...source,
      sourceText: trimText(
        source.source_text || source.article_text || source.snippet || "",
        420,
      ),
      relevanceScore: rankSourceForDraft(source, index),
    }))
    .sort((left, right) => right.relevanceScore - left.relevanceScore)
    .forEach((source) => {
      const key = normalizeSourceIdentity(source);
      if (seen.has(key)) {
        return;
      }
      seen.add(key);
      deduped.push(source);
    });

  return deduped.slice(0, limit);
}

function rankSourceForDraft(source, index) {
  let score = 10 - index * 0.15;
  if (!isIndirectSource(source)) {
    score += 4;
  }
  if (source.article_text) {
    score += 2;
  }
  if (source.snippet) {
    score += 1.5;
  }
  if (source.source_text) {
    score += Math.min(2, source.source_text.length / 280);
  }
  return score;
}

function normalizeSourceIdentity(source) {
  const titleKey = cleanupSourceTitle(source.title)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();

  try {
    const parsed = new URL(source.url || "", window.location.origin);
    if (parsed.hostname.includes("news.google.com")) {
      return `title:${titleKey}`;
    }
    return `url:${parsed.hostname}${parsed.pathname}`;
  } catch (error) {
    return `title:${titleKey}`;
  }
}

function isIndirectSource(source) {
  return /news\.google\.com/i.test(source?.url || "");
}

function cleanupSourceTitle(title) {
  return String(title || "")
    .replace(/\s+-\s+(Reuters|AP News|Associated Press|MSN|AOL\.com|Yahoo!?\s*News|DW\.com)$/i, "")
    .trim();
}

function extractThemeTerms(sources, limit) {
  const termScores = new Map();
  const stopwords = new Set([
    "about", "after", "amid", "analysis", "and", "are", "but", "from", "have", "into",
    "latest", "news", "over", "says", "saying", "that", "the", "their", "this", "what",
    "when", "where", "which", "will", "with", "would",
  ]);

  sources.forEach((source) => {
    const text = `${cleanupSourceTitle(source.title)} ${source.snippet || ""}`.toLowerCase();
    const words = text.match(/[a-z][a-z0-9-]{2,}/g) || [];
    const seenInSource = new Set();
    words.forEach((word) => {
      if (stopwords.has(word) || seenInSource.has(word)) {
        return;
      }
      seenInSource.add(word);
      termScores.set(word, (termScores.get(word) || 0) + 1);
    });
  });

  return [...termScores.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, limit)
    .map(([term]) => term);
}

function buildWorkingThesis(research, selectedSources, themeTerms) {
  if (!selectedSources.length) {
    return `Chronicle has not verified enough reporting yet on "${research.brief}" and should draft carefully from the confirmed brief only.`;
  }

  if (themeTerms.length >= 2) {
    return `The strongest verified signal in this brief is the overlap between ${themeTerms[0]} and ${themeTerms[1]}, not any single isolated headline.`;
  }

  return `The clearest verified pattern is that ${cleanupSourceTitle(selectedSources[0].title)} carries the lead signal for this briefing.`;
}

function buildCoverageGap(selectedSources) {
  if (!selectedSources.length) {
    return "No external reporting was verified, so the draft must stay explicit about uncertainty.";
  }
  if (selectedSources.every((source) => isIndirectSource(source))) {
    return "Most evidence is coming from headline-level feeds, so the draft should stay conservative about details.";
  }
  return "Use the selected sources directly and avoid claims that are not grounded in the retrieved evidence.";
}

function buildReasoningSummary(packet) {
  return [
    `Focus: ${packet.brief}`,
    `Working thesis: ${packet.workingThesis}`,
    `Evidence selected: ${packet.selectedSources.length} unique source(s)`,
    `Key evidence: ${packet.keyEvidence.join(" | ") || "No source headlines yet"}`,
    `Coverage gap: ${packet.coverageGap}`,
  ].join("\n");
}

function finalizeNewsletterMarkdown(markdown, packet) {
  let text = String(markdown || "").trim();
  if (!text) {
    text = renderFallbackNewsletter(packet);
  }
  if (!/^#\s+/m.test(text)) {
    text = `# ${packet.title}\n\n${text}`;
  }
  if (!/\n##\s+Sources\b/i.test(text)) {
    text = `${text.trim()}\n\n${buildSourcesSection(packet)}`;
  }
  return text.trim();
}

function renderFallbackNewsletter(packet) {
  const lines = [
    `# ${packet.title}`,
    "",
    `${packet.workingThesis}`,
    "",
  ];

  packet.sections.forEach((sectionName, index) => {
    lines.push(`## ${sectionName}`);
    const source = packet.selectedSources[index] || packet.selectedSources[0];
    if (source) {
      lines.push(
        `${cleanupSourceTitle(source.title)} provides the clearest verified anchor here. ${trimText(source.sourceText || source.snippet || "", 280)} [${Math.min(index + 1, packet.selectedSources.length)}]`,
      );
    } else {
      lines.push("Chronicle could not verify enough fresh reporting for this section, so this draft stays explicit about that gap.");
    }
    lines.push("");
  });

  lines.push("## Closing note");
  lines.push(packet.coverageGap);
  lines.push("");
  lines.push(buildSourcesSection(packet));
  return lines.join("\n");
}

function buildSourcesSection(packet) {
  const sources = packet.selectedSources.length
    ? packet.selectedSources
      .map((source, index) => `[${index + 1}]: ${cleanupSourceTitle(source.title)} - ${source.url}`)
      .join("\n")
    : "No external sources were successfully collected.";

  const marketLine = packet.marketSnapshot?.length
    ? "\n[M1]: CoinGecko Markets API - https://www.coingecko.com/"
    : "";

  return `## Sources\n${sources}${marketLine}`;
}

function detectBrowserCapabilities(browserConfig) {
  const hasWebGPU = typeof navigator !== "undefined" && Boolean(navigator.gpu);
  const deviceMemory = Number(navigator.deviceMemory || 0);
  const hardwareConcurrency = Number(navigator.hardwareConcurrency || 0);
  const isMobile = typeof navigator !== "undefined"
    && /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
  const recommendedProfile = calculateBrowserProfile({
    hasWebGPU,
    deviceMemory,
    hardwareConcurrency,
    isMobile,
    supportsSlicing: Boolean(browserConfig?.supports_slicing),
    maxSlices: Number(browserConfig?.max_slices || 1),
  });
  const candidates = buildBrowserCandidates(browserConfig, recommendedProfile, {
    hasWebGPU,
    deviceMemory,
    hardwareConcurrency,
    isMobile,
  });

  return {
    hasWebGPU,
    deviceMemory,
    hardwareConcurrency,
    isMobile,
    recommendedProfile,
    candidates,
  };
}

async function ensureBrowserSession() {
  if (state.browserSession?.model) {
    return state.browserSession;
  }

  if (!state.browserConfig?.local_model_ready || !state.browserConfig?.local_model_id) {
    throw new Error("No local browser model bundle is available on this server.");
  }

  const { AutoModelForImageTextToText, AutoProcessor, TextStreamer, env } = await import(TRANSFORMERS_CDN);
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = "/models";
  env.useBrowserCache = true;

  if (env.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
  }

  let lastError = null;
  for (const candidate of state.browserCapabilities.candidates) {
    try {
      const processor = await AutoProcessor.from_pretrained(candidate.model);
      const model = await AutoModelForImageTextToText.from_pretrained(
        candidate.model,
        candidate.modelOptions,
      );
      state.browserSession = {
        processor,
        model,
        TextStreamer,
        profile: candidate,
      };
      return state.browserSession;
    } catch (error) {
      lastError = error;
      appendLogLines([`Model candidate failed: ${candidate.label}`]);
      console.warn(`Chronicle browser candidate failed: ${candidate.label}`, error);
    }
  }

  throw new Error(lastError?.message || "No browser inference backend could be initialized.");
}

async function generateNewsletterMarkdown(research, packet, aiSession, onProgress) {
  const messages = [
    {
      role: "user",
      content: [
        {
          type: "text",
          text: buildNewsletterPrompt(research, packet),
        },
      ],
    },
  ];
  const prompt = aiSession.processor.apply_chat_template(messages, {
    add_generation_prompt: true,
  });
  const inputs = await aiSession.processor(prompt, null, null, {
    add_special_tokens: false,
  });
  const inputLength = inputs.input_ids?.dims?.at(-1);

  if (!inputLength) {
    throw new Error("Gemma 3n processor did not return input ids for generation.");
  }

  let streamedText = "";
  const streamer = aiSession.TextStreamer
    ? new aiSession.TextStreamer(aiSession.processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text) => {
        streamedText += text;
        onProgress?.(streamedText);
      },
    })
    : null;

  const output = await withTimeout(
    aiSession.model.generate({
      ...inputs,
      max_new_tokens: aiSession.profile.maxNewTokens,
      do_sample: false,
      streamer,
    }),
    aiSession.profile.generationTimeoutMs,
    "Browser generation timed out before the newsletter finished.",
  );
  const generatedTokens = output.slice(null, [inputLength, null]);
  const decoded = aiSession.processor.batch_decode(generatedTokens, {
    skip_special_tokens: true,
  });
  const generatedText = Array.isArray(decoded) ? decoded[0] : decoded;

  if (!generatedText && !streamedText.trim()) {
    throw new Error("Browser model returned an empty response.");
  }
  return generatedText || streamedText;
}

function buildNewsletterPrompt(research, packet) {
  const sourceBundle = packet.selectedSources
    .map((source, index) => {
      const excerpt = trimText(source.sourceText || "", 260);
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
Last ${packet.days} days

Planned sections:
${JSON.stringify(packet.sections)}

Reasoning summary:
${JSON.stringify({
  working_thesis: packet.workingThesis,
  key_evidence: packet.keyEvidence,
  coverage_gap: packet.coverageGap,
  dominant_themes: packet.themeTerms,
})}

Structured market data:
${marketData}

Research notes:
${sourceBundle || "No sources were collected."}

Requirements:
- return markdown only
- start with one H1 title
- aim for roughly 550 to 800 words
- write a short opening note with a point of view
- organize the body around the planned sections using H2 headings
- use the reasoning summary to keep a strong throughline instead of just listing events
- keep the writing analytical, premium, and specific
- cite source notes inline like [1], [2]
- if market data is present, reference it as [M1]
- end with a Sources section
- in Sources, list each source as [id]: title - url
- if market data is present, include: [M1]: CoinGecko Markets API - https://www.coingecko.com/
- do not use markdown code fences
- do not mention system prompts or implementation details`;
}

function calculateBrowserProfile(config) {
  const maxSlices = Math.max(1, Number(config.maxSlices || 1));
  const supportsSlicing = Boolean(config.supportsSlicing && maxSlices > 1);

  if (!supportsSlicing) {
    return {
      sliceCount: 1,
      percentage: 100,
      label: "Single bundle",
      maxNewTokens: config.hasWebGPU ? 420 : 280,
      generationTimeoutMs: config.hasWebGPU ? 180000 : 120000,
      temperature: 0.2,
    };
  }

  let sliceCount = 1;
  if (!config.hasWebGPU) {
    sliceCount = 1;
  } else if (config.isMobile) {
    sliceCount = config.deviceMemory >= 12 ? 2 : 1;
  } else if (config.deviceMemory >= 24 && config.hardwareConcurrency >= 16) {
    sliceCount = Math.min(maxSlices, 4);
  } else if (config.deviceMemory >= 16 || config.hardwareConcurrency >= 12) {
    sliceCount = Math.min(maxSlices, 3);
  } else if (config.deviceMemory >= 8 || config.hardwareConcurrency >= 8) {
    sliceCount = Math.min(maxSlices, 2);
  }

  const percentage = Math.round((sliceCount / maxSlices) * 100);
  let maxNewTokens = 360;
  if (percentage >= 50) {
    maxNewTokens = 520;
  } else if (percentage >= 25) {
    maxNewTokens = 440;
  }

  return {
    sliceCount,
    percentage,
    label: `${percentage}% slice (${sliceCount}/${maxSlices})`,
    maxNewTokens,
    generationTimeoutMs: percentage >= 50 ? 210000 : 150000,
    temperature: 0.2,
  };
}

function buildBrowserCandidates(browserConfig, recommendedProfile, deviceConfig) {
  if (!browserConfig?.local_model_ready || !browserConfig?.local_model_id) {
    return [];
  }

  const supportsSlicing = Boolean(browserConfig.supports_slicing && browserConfig.max_slices > 1);
  const candidates = [];
  const preferredSlices = supportsSlicing
    ? buildSliceFallbackChain(recommendedProfile.sliceCount, Number(browserConfig.max_slices || 1))
    : [1];

  if (deviceConfig.hasWebGPU) {
    preferredSlices.forEach((sliceCount) => {
      candidates.push(buildBrowserCandidate(browserConfig, "webgpu", sliceCount, recommendedProfile));
    });
  }

  const wasmSlices = supportsSlicing
    ? buildSliceFallbackChain(Math.min(recommendedProfile.sliceCount, 2), Number(browserConfig.max_slices || 1))
    : [1];
  wasmSlices.forEach((sliceCount) => {
    candidates.push(buildBrowserCandidate(browserConfig, "wasm", sliceCount, recommendedProfile));
  });

  return dedupeCandidates(candidates);
}

function buildBrowserCandidate(browserConfig, device, sliceCount, recommendedProfile) {
  const supportsSlicing = Boolean(browserConfig.supports_slicing && browserConfig.max_slices > 1);
  const normalizedSliceCount = Math.max(1, sliceCount || 1);
  const maxSlices = Math.max(1, Number(browserConfig.max_slices || 1));
  const percentage = Math.round((normalizedSliceCount / maxSlices) * 100);
  const sliceLabel = supportsSlicing
    ? `${percentage}% slice (${normalizedSliceCount}/${maxSlices})`
    : "Single bundle";
  const maxNewTokens = device === "webgpu"
    ? Math.max(320, recommendedProfile.maxNewTokens - Math.max(recommendedProfile.sliceCount - normalizedSliceCount, 0) * 60)
    : Math.min(280, recommendedProfile.maxNewTokens);
  const modelOptions = { device };

  if (browserConfig.dtype_map && Object.keys(browserConfig.dtype_map).length) {
    modelOptions.dtype = browserConfig.dtype_map;
  }

  if (supportsSlicing) {
    modelOptions.model_kwargs = { num_slices: normalizedSliceCount };
  }

  return {
    device,
    model: browserConfig.local_model_id,
    label: `${device === "webgpu" ? "WebGPU" : "WASM"} ${sliceLabel}`,
    sliceCount: normalizedSliceCount,
    sliceLabel,
    percentage,
    maxNewTokens,
    generationTimeoutMs: device === "webgpu"
      ? Math.max(120000, recommendedProfile.generationTimeoutMs - Math.max(recommendedProfile.sliceCount - normalizedSliceCount, 0) * 15000)
      : Math.min(120000, recommendedProfile.generationTimeoutMs),
    temperature: recommendedProfile.temperature,
    modelOptions,
  };
}

function buildSliceFallbackChain(targetSlices, maxSlices) {
  const chain = [];
  const normalizedTarget = Math.max(1, Math.min(targetSlices || 1, maxSlices || 1));
  [normalizedTarget, Math.ceil(normalizedTarget * 0.75), Math.ceil(normalizedTarget * 0.5), 2, 1]
    .forEach((value) => {
      const sliceCount = Math.max(1, Math.min(value, maxSlices));
      if (!chain.includes(sliceCount)) {
        chain.push(sliceCount);
      }
    });
  return chain;
}

function dedupeCandidates(candidates) {
  const seen = new Set();
  return candidates.filter((candidate) => {
    const key = `${candidate.device}:${candidate.model}:${candidate.sliceCount}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
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

function withTimeout(promise, timeoutMs, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      window.setTimeout(() => reject(new Error(message)), timeoutMs);
    }),
  ]);
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

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

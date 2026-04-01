const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@next";

const MODEL_CANDIDATES = [
  "onnx-community/tiny-random-gpt2-ONNX",
  "Xenova/distilgpt2",
  "Xenova/gpt2",
];

const GEMMA3N_DTYPE_MAP = {
  decoder_model_merged: "q4",
  embed_tokens: "q4",
  audio_encoder: "q4f16",
  vision_encoder: "uint8",
};

const GEMMA3N_TEXT_DTYPE_MAP = {
  decoder_model_merged: "q4",
  embed_tokens: "q4",
};

const GEMMA3N_TEXT_WASM_DTYPE_MAP = {
  decoder_model_merged: "fp32",
  embed_tokens: "fp32",
};

function isGemma3nModel(modelId) {
  return String(modelId || "").toLowerCase().includes("gemma-3n");
}

const DEPTH_CONFIG = {
  low: { queryCount: 1, resultsPerQuery: 1, summarizeCount: 1 },
  medium: { queryCount: 1, resultsPerQuery: 1, summarizeCount: 1 },
  high: { queryCount: 2, resultsPerQuery: 2, summarizeCount: 2 },
};

const TURN_STAGES = [
  { key: "query", index: "01", title: "Generate search queries" },
  { key: "search", index: "02", title: "Fetch Google results" },
  { key: "summarize", index: "03", title: "Summarize results" },
  { key: "newsletter", index: "04", title: "Generate newsletter" },
  { key: "render", index: "05", title: "Render and export" },
];

const MODE_COPY = {
  concise: "Write a sharp, selective editorial brief with minimal filler.",
  feynman: "Explain clearly in plain language, teaching the reader without sounding simplistic.",
  soc: "Use a Socratic structure, with section headings phrased as sharp questions answered clearly.",
};

const WRITER_POLICY = {
  name: "balanced_fast",
  openingEvidenceCount: 1,
  sectionEvidenceCount: 1,
  openingMaxTokens: 64,
  sectionMaxTokens: 90,
  fallbackVariantChars: [72],
  fallbackTimeoutMs: 50000,
  fallbackMaxTokens: { medium: 120, compact: 96 },
  polishOnPassOnly: false,
  polishMaxTokens: 120,
  polishTimeoutMs: 50000,
  openingAttemptTimeoutMs: 40000,
  sectionAttemptTimeoutMs: 40000,
};

const state = {
  browserCapabilities: null,
  browserProfile: null,
  browserSession: null,
  browserSessionPromise: null,
  browserWarmStarted: false,
  browserRuntimeStatus: "idle",
  browserRuntimeMessage: "Model is idle. Chronicle will load it when you click Generate.",
  browserRuntimeProgress: 0,
  browserRuntimeProgressText: "Ready to initialize",
  browserLoadProgressEntries: {},
  browserLoadCandidate: "",
  browserLastProgressAt: 0,
  browserLoadStartedAt: 0,
  transformersRuntimePromise: null,
  currentTurn: null,
  activeStageKey: null,
  autoScrollToBottom: true,
  lastWarmupUiUpdateAt: 0,
  modelWorker: null,
  workerPending: new Map(),
  workerRequestSeq: 0,
};

const elements = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindComposer();
  hydrateApp().catch((error) => {
    console.error(error);
    finishTurnWithError(error.message || "Chronicle failed to initialize.");
  });
});

function cacheElements() {
  elements.statusPill = document.getElementById("status-pill");
  elements.statusCopy = document.getElementById("status-copy");
  elements.runtimeProgressFill = document.getElementById("runtime-progress-fill");
  elements.runtimeProgressText = document.getElementById("runtime-progress-text");
  elements.chatThread = document.getElementById("chat-thread");
  elements.emptyState = document.getElementById("empty-state");
  elements.composerForm = document.getElementById("composer-form");
  elements.brief = document.getElementById("brief");
  elements.explanationStyle = document.getElementById("explanation-style");
  elements.customStyleField = document.getElementById("custom-style-field");
  elements.generateButton = document.getElementById("generate-button");
  elements.chatThread?.addEventListener("scroll", onChatThreadScroll, { passive: true });
}

function bindComposer() {
  elements.explanationStyle.addEventListener("change", () => {
    toggleCustomStyleField(elements.explanationStyle.value === "custom");
  });

  elements.composerForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = buildPayload();
    try {
      validatePayload(payload);
    } catch (error) {
      appendSystemMessage(error.message || "Chronicle needs a valid topic.");
      return;
    }

    startNewTurn(payload.brief);
    setGenerateBusy(true);

    try {
      await runBrowserPipeline(payload);
    } catch (error) {
      console.error(error);
      finishTurnWithError(error.message || "Chronicle could not complete the browser-only run.");
    }
  });
}

function buildPayload() {
  const formData = new FormData(elements.composerForm);
  return {
    brief: cleanText(formData.get("brief") || ""),
    depth: cleanText(formData.get("depth") || "medium").toLowerCase(),
    explanation_style: cleanText(formData.get("explanation_style") || "concise").toLowerCase(),
    style_instructions: cleanText(formData.get("style_instructions") || ""),
  };
}

function validatePayload(payload) {
  if (!payload.brief) {
    throw new Error("Enter a topic before starting Chronicle.");
  }

  if (payload.explanation_style === "custom" && !payload.style_instructions) {
    throw new Error("Add custom style instructions or switch to another reasoning mode.");
  }
}

async function hydrateApp() {
  toggleCustomStyleField(false);
  state.browserCapabilities = detectBrowserCapabilities();
  state.browserProfile = calculateBrowserProfile(state.browserCapabilities);
  renderHeaderStatus();
  warmBrowserSessionInBackground();
}

function renderHeaderStatus() {
  const runtimeLabel = state.browserCapabilities?.hasWebGPU ? "WebGPU" : "WASM";
  const progress = clampNumber(
    state.browserRuntimeStatus === "ready" ? 1 : state.browserRuntimeProgress || 0,
    0,
    1,
  );

  if (elements.runtimeProgressFill) {
    elements.runtimeProgressFill.style.width = `${Math.round(progress * 100)}%`;
  }
  if (elements.runtimeProgressText) {
    elements.runtimeProgressText.textContent = state.browserRuntimeProgressText || "Preparing model files…";
  }

  if (state.browserRuntimeStatus === "ready") {
    setStatusPill("Model ready");
    elements.statusCopy.textContent = state.browserRuntimeMessage;
    return;
  }

  if (state.browserRuntimeStatus === "idle") {
    setStatusPill("Model idle");
    elements.statusCopy.textContent = state.browserRuntimeMessage;
    return;
  }

  if (state.browserRuntimeStatus === "error") {
    setStatusPill("Model blocked", "is-danger");
    elements.statusCopy.textContent = state.browserRuntimeMessage;
    return;
  }

  setStatusPill("Preparing model", "is-warm");
  elements.statusCopy.textContent = state.browserRuntimeMessage || `Preparing model in ${runtimeLabel}.`;
}

function startNewTurn(userPrompt) {
  appendUserMessage(userPrompt);
  elements.brief.value = "";
  state.autoScrollToBottom = true;
  state.currentTurn = {
    statusNode: appendAssistantMessage("Generating search queries", "Stage"),
    stageNode: appendStageCard(),
    resultNode: null,
    completed: false,
  };
  state.activeStageKey = "query";
  TURN_STAGES.forEach((stage, index) => {
    setStageState(stage.key, index === 0 ? "active" : "pending");
  });
}

function updateTurnStatus(text) {
  const turn = state.currentTurn;
  if (!turn?.statusNode) {
    return;
  }
  const body = turn.statusNode.querySelector(".message-body");
  body.textContent = text;
  scrollThreadToBottom();
}

async function runBrowserPipeline(payload) {
  const config = DEPTH_CONFIG[payload.depth] || DEPTH_CONFIG.medium;
  const run = {
    config: payload,
    queryPlan: null,
    rawResults: [],
    resultSummaries: [],
    selectedSummaries: [],
    finalNewsletter: null,
  };

  state.activeStageKey = "query";
  updateTurnStatus("Loading Chronicle brain (first run can take 1-3 minutes)");
  setStageState("query", "active", getWarmupProgressLabel());
  const researchPromise = prepareDeterministicResearch(run, config);
  const aiSession = await waitForBrowserSessionReady();
  const research = await researchPromise;

  run.queryPlan = research.queryPlan;
  run.rawResults = research.rawResults;
  run.resultSummaries = research.resultSummaries;
  run.selectedSummaries = research.selectedSummaries;
  setStageState("search", "complete", `${run.rawResults.length} results collected`);
  setStageState("summarize", "complete", `${run.resultSummaries.length} summaries stored`);
  appendSystemMessage("Chronicle used a fast deterministic evidence pass to keep the first issue responsive.");

  state.activeStageKey = "newsletter";
  setStageState("newsletter", "active", "Writing issue");
  updateTurnStatus("Generating newsletter");
  run.newsletterStartedAt = performance.now();
  run.finalNewsletter = await generateNewsletter(aiSession, run);
  const stage4Seconds = Math.max(1, Math.round((run.writeTimings?.totalMs || 0) / 1000));
  setStageState("newsletter", "complete", `Draft complete (${stage4Seconds}s)`);

  state.activeStageKey = "render";
  setStageState("render", "active", "Preparing HTML issue");
  updateTurnStatus("Rendering newsletter");
  appendResultCard(run.finalNewsletter);
  setStageState("render", "complete", "Ready");
  updateTurnStatus(`Issue ready: ${run.finalNewsletter.title}`);
  setGenerateBusy(false);
}

async function prepareDeterministicResearch(run, config) {
  updateTurnStatus("Preparing search queries");
  run.queryPlan = buildFallbackQueryPlan(run.config.brief, config.queryCount);
  setStageState("query", "complete", `${run.queryPlan.queries.length} queries ready`);

  state.activeStageKey = "search";
  run.searchStartedAt = performance.now();
  setStageState("search", "active", "Connecting to Google");
  updateTurnStatus("Fetching Google results");
  run.rawResults = await fetchAllGoogleResults(run.queryPlan.queries, config.resultsPerQuery);
  if (!run.rawResults.length) {
    appendSystemMessage("Google returned no readable results. Chronicle will continue using synthetic placeholder evidence.");
    run.rawResults = buildSyntheticRawResults(run.queryPlan, run.config, config.summarizeCount);
  }
  setStageState("search", "complete", `${run.rawResults.length} results collected`);

  state.activeStageKey = "summarize";
  const prioritizedRawResults = prioritizeResultsForSummarization(run.rawResults, run.config.brief, config.summarizeCount);
  run.summarizeStartedAt = performance.now();
  setStageState("summarize", "active", `0/${prioritizedRawResults.length} summarized`);
  updateTurnStatus("Condensing results");
  for (let index = 0; index < prioritizedRawResults.length; index += 1) {
    const result = prioritizedRawResults[index];
    const summary = buildFallbackResultSummary(result);
    run.resultSummaries.push(summary);
    const progressText = `${index + 1}/${prioritizedRawResults.length} summarized`;
    const etaText = estimateLoopEta(run.summarizeStartedAt, index + 1, prioritizedRawResults.length);
    setStageState("summarize", "active", `${progressText}${etaText ? ` · ETA ${etaText}` : ""}`);
    updateTurnStatus(`Condensing results (${progressText}${etaText ? ` · ETA ${etaText}` : ""})`);
  }

  run.selectedSummaries = selectNarrativeSources(run.resultSummaries, run.config.brief);
  return {
    queryPlan: run.queryPlan,
    rawResults: run.rawResults,
    resultSummaries: run.resultSummaries,
    selectedSummaries: run.selectedSummaries,
  };
}

async function generateSearchPlan(aiSession, payload, queryCount) {
  const promptVariants = [
    [
      "Make Google queries for a news research task.",
      `Topic: ${payload.brief}`,
      `Need exactly ${queryCount} queries.`,
      "Return plain text only in this exact format:",
      "TITLE: short issue title",
      "Q1: search query",
      "Q2: search query",
      "Q3: search query",
      "Q4: search query",
      "Rules:",
      "- short literal search queries",
      "- focus on fresh reporting",
      "- no markdown",
      "- no explanation",
    ].join("\n"),
    [
      `Topic: ${payload.brief}`,
      `Return TITLE plus exactly ${queryCount} search lines.`,
      "Format:",
      "TITLE: ...",
      "Q1: ...",
      "Q2: ...",
    ].join("\n"),
  ];

  let lastText = "";
  for (const promptText of promptVariants) {
    try {
      const prepared = await preparePromptInputs(aiSession, promptText);
      if (isPromptTooLongForSession(aiSession, prepared)) {
        continue;
      }
      const generatedText = await generateFromPreparedInputs(aiSession, prepared, {
        maxNewTokens: 72,
        timeoutMs: 120000,
      });
      lastText = generatedText;
      const plan = parseSearchPlanText(generatedText, queryCount, payload.brief);
      if (plan) {
        return plan;
      }
    } catch (error) {
      console.warn("[Chronicle] Query-plan generation attempt failed.", error);
    }
  }

  const repairedPlan = parseSearchPlanText(lastText, queryCount, payload.brief);
  if (repairedPlan) {
    return repairedPlan;
  }

  return buildFallbackQueryPlan(payload.brief, queryCount);
}

async function fetchAllGoogleResults(queries, perQuery) {
  const allResults = [];
  const startedAt = performance.now();

  for (let index = 0; index < queries.length; index += 1) {
    const query = queries[index];
    const progressLabel = `Query ${index + 1}/${queries.length}`;
    const etaText = estimateLoopEta(startedAt, index, queries.length);
    setStageState("search", "active", `${progressLabel}${etaText ? ` · ETA ${etaText}` : ""}`);
    updateTurnStatus(`Fetching Google results (${progressLabel}${etaText ? ` · ETA ${etaText}` : ""})`);
    const results = await fetchGoogleResults(query, perQuery);
    allResults.push(...results);
  }

  return allResults;
}

async function fetchGoogleResults(query, limit) {
  const url = new URL("/search/google", window.location.origin);
  url.searchParams.set("q", query);
  url.searchParams.set("hl", "en-US");
  url.searchParams.set("gl", "US");
  url.searchParams.set("ceid", "US:en");

  let response;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 9000);
  try {
    response = await fetch(url.toString(), {
      method: "GET",
      cache: "no-store",
      signal: controller.signal,
    });
  } catch (error) {
    throw new Error("Chronicle could not reach the local Google relay.");
  } finally {
    window.clearTimeout(timeoutId);
  }

  if (!response.ok) {
    let relayError = "";
    try {
      const payload = await response.json();
      relayError = cleanText(payload.error || "");
    } catch (error) {
      relayError = "";
    }
    throw new Error(relayError || `Google search relay failed with status ${response.status}.`);
  }

  const xmlText = await response.text();
  const parser = new DOMParser();
  const xml = parser.parseFromString(xmlText, "application/xml");
  if (xml.querySelector("parsererror")) {
    throw new Error("Google returned unreadable search data.");
  }

  return Array.from(xml.querySelectorAll("item"))
    .slice(0, limit)
    .map((item, index) => {
      const title = cleanText(item.querySelector("title")?.textContent || "");
      const resultUrl = cleanText(item.querySelector("link")?.textContent || "");
      const description = item.querySelector("description")?.textContent || "";
      const snippet = extractSnippetFromDescription(description, title);

      return {
        result_id: `${slugify(query)}-${index + 1}`,
        query,
        rank: index + 1,
        title,
        url: resultUrl,
        snippet,
      };
    })
    .filter((item) => item.title && item.url);
}

async function summarizeSearchResult(aiSession, rawResult, options = {}) {
  const promptVariants = [
    [
      "Summarize one raw Google result without inventing facts.",
      `title: ${rawResult.title}`,
      `url: ${rawResult.url}`,
      `snippet: ${rawResult.snippet || "(empty)"}`,
      "Return plain text only in this exact format:",
      "SUMMARY: ...",
      "SIGNAL: ...",
      "CONFIDENCE: ...",
      "Rules:",
      "- SUMMARY max 2 sentences",
      "- rewrite the headline in fresh prose",
      "- do not copy the full title verbatim",
      "- SIGNAL max 12 words",
      "- CONFIDENCE must say this is headline/snippet level evidence",
    ].join("\n"),
    `SUMMARY: ...\nSIGNAL: ...\nCONFIDENCE: ...\ntitle: ${rawResult.title}\nsnippet: ${rawResult.snippet || "(empty)"}`,
  ];

  let lastText = "";
  let lastErrorMessage = "";
  const timeoutMs = options.longTimeout ? 180000 : 120000;
  for (const promptText of promptVariants) {
    try {
      const prepared = await preparePromptInputs(aiSession, promptText);
      if (isPromptTooLongForSession(aiSession, prepared)) {
        lastErrorMessage = "prompt_too_long";
        continue;
      }
      const generatedText = await generateFromPreparedInputs(aiSession, prepared, {
        maxNewTokens: 90,
        timeoutMs,
      });
      lastText = generatedText;
      const summary = parseResultSummaryText(generatedText, rawResult);
      if (summary) {
        summary.model_generated = true;
        return summary;
      }
    } catch (error) {
      lastErrorMessage = cleanText(error?.message || "") || "generation_failed";
      console.warn(`[Chronicle] Result summarization failed for ${rawResult.title}.`, error);
    }
  }

  const repaired = parseResultSummaryText(lastText, rawResult);
  if (repaired) {
    repaired.model_generated = true;
    return repaired;
  }
  const fallback = buildFallbackResultSummary(rawResult);
  fallback.model_error = lastErrorMessage || "fallback_without_model_output";
  return fallback;
}

async function generateNewsletter(aiSession, run) {
  run.writeTimings = {};
  const newsletterStartedAt = performance.now();
  const frame = buildFallbackNewsletterFrame(run);
  updateTurnStatus("Generating newsletter");
  setStageState("newsletter", "active", "Fast editorial render");
  const bodyStartedAt = performance.now();
  const generatedMarkdown = buildEmergencyNewsletterMarkdown(run, frame);
  run.writeTimings.bodyMs = Math.round(performance.now() - bodyStartedAt);
  run.writeTimings.totalMs = Math.round(performance.now() - newsletterStartedAt);

  const markdown = finalizeNewsletterMarkdown(generatedMarkdown, run);
  const title = extractTitleFromMarkdown(markdown, run.queryPlan.title);
  const storageKey = `chronicle::issue::${slugify(title)}::${Date.now()}`;
  const htmlDocument = renderEditableIssueHtml(title, markdown, storageKey);

  return {
    title,
    markdown,
    htmlDocument,
    storageKey,
    downloadName: `${slugify(title)}-editable.html`,
  };
}

async function generateRescueNewsletter(aiSession, run, frame) {
  const evidence = (run.selectedSummaries?.length ? run.selectedSummaries : run.resultSummaries).slice(0, 4);
  const summaryLines = evidence
    .map((summary, index) => (
      `[${index + 1}] ${trimText(summary.source_title, 84)}\n` +
      `summary: ${trimText(summary.summary, 96)}\n` +
      `signal: ${trimText(summary.signal, 52)}`
    ))
    .join("\n\n");

  const promptText = [
    "Write one complete markdown newsletter from the evidence below.",
    `Topic: ${run.config.brief}`,
    `Mode: ${run.config.explanation_style}`,
    `Style instructions: ${buildPresentationInstructions(run.config)}`,
    `Use this title: ${frame.title}`,
    `Use this subtitle: ${frame.subtitle}`,
    "",
    "Evidence:",
    summaryLines,
    "",
    "Format:",
    "- H1 title",
    "- italic subtitle",
    "- opening paragraph",
    "- three H2 sections with original headings",
    "- inline citations like [1]",
    "- markdown only",
  ].join("\n");

  try {
    const prepared = await preparePromptInputs(aiSession, promptText);
    if (isPromptTooLongForSession(aiSession, prepared)) {
      return "";
    }
    const generatedText = await generateFromPreparedInputs(aiSession, prepared, {
      maxNewTokens: aiTokenBudget(run, 740),
      timeoutMs: aiTimeoutBudget(run, 150000),
      doSample: true,
      temperature: 0.74,
      topP: 0.92,
    });
    const cleaned = stripMarkdownFences(generatedText).trim();
    return isUsableNewsletterMarkdown(cleaned, frame) ? cleaned : "";
  } catch (error) {
    console.warn("[Chronicle] Rescue newsletter generation failed.", error);
    return "";
  }
}

function buildNewsletterPromptVariants(run, frame) {
  const styleInstructions = buildPresentationInstructions(run.config);
  const variants = [];
  const evidence = (run.selectedSummaries?.length ? run.selectedSummaries : run.resultSummaries).slice(0, 6);

  for (const summaryChars of WRITER_POLICY.fallbackVariantChars) {
    const summaryLines = evidence
      .map((summary, index) => (
        `[${index + 1}] ${trimText(summary.source_title, summaryChars >= 140 ? 96 : 72)}\n` +
        `summary: ${trimText(summary.summary, summaryChars)}\n` +
        `signal: ${trimText(summary.signal, summaryChars >= 140 ? 64 : 48)}\n` +
        `confidence: ${trimText(summary.confidence_note, 92)}`
      ))
      .join("\n\n");

    variants.push({
      promptText: [
        "You are Chronicle.",
        "Use only the summarized evidence below to write one original markdown newsletter.",
        "Do not dump headlines. Synthesize them into one coherent argument.",
        `Topic: ${run.config.brief}`,
        `Title: ${frame.title}`,
        `Subtitle: ${frame.subtitle}`,
        `Mode: ${run.config.explanation_style}`,
        `Style instructions: ${styleInstructions}`,
        `Editorial angle: ${frame.angle}`,
        `Section headings: ${frame.headings.join(" | ")}`,
        "",
        "Summarized evidence:",
        summaryLines,
        "",
        "Write one markdown newsletter with these rules:",
        "- start with one H1 title using the provided title",
        "- immediately add one italic subtitle line using the provided subtitle",
        "- write one opening paragraph that states the main thesis",
        "- use the provided section headings as H2s",
        "- each section must advance the argument, not repeat source titles",
        "- do not paste or mirror headline text line-by-line",
        "- cite sources inline as [1], [2], [3] when making claims",
        "- end with ## Sources",
        "- do not use raw URLs outside the Sources section",
        "- write only markdown",
      ].join("\n"),
      maxNewTokens: summaryChars >= 110 ? aiTokenBudget(run, WRITER_POLICY.fallbackMaxTokens.medium) : aiTokenBudget(run, WRITER_POLICY.fallbackMaxTokens.compact),
      timeoutMs: aiTimeoutBudget(run, WRITER_POLICY.fallbackTimeoutMs),
    });
  }

  return variants;
}

function buildPresentationInstructions(config) {
  if (config.explanation_style === "custom" && config.style_instructions) {
    return trimText(config.style_instructions, 220);
  }
  return MODE_COPY[config.explanation_style] || MODE_COPY.concise;
}

async function generateNewsletterFrame(aiSession, run) {
  const evidence = (run.selectedSummaries?.length ? run.selectedSummaries : run.resultSummaries).slice(0, 6);
  const summaryLines = evidence
    .map((summary, index) => (
      `[${index + 1}] ${trimText(summary.source_title, 88)}\n` +
      `summary: ${trimText(summary.summary, 132)}\n` +
      `signal: ${trimText(summary.signal, 56)}`
    ))
    .join("\n\n");

  const promptVariants = [
    [
      "Create the editorial frame for a Chronicle newsletter.",
      `Topic: ${run.config.brief}`,
      `Mode: ${run.config.explanation_style}`,
      `Style instructions: ${buildPresentationInstructions(run.config)}`,
      "",
      "Evidence:",
      summaryLines,
      "",
      "Return plain text only in this exact format:",
      "TITLE: ...",
      "SUBTITLE: ...",
      "ANGLE: ...",
      "H1: ...",
      "H2: ...",
      "H3: ...",
      "Rules:",
      "- title must be original and specific",
      "- subtitle must state the core thesis in one sentence",
      "- angle must describe the main argument",
      "- headings must be dynamic and topic-specific",
      "- do not use generic headings like What happened or Why it matters",
    ].join("\n"),
    [
      `Topic: ${run.config.brief}`,
      "Return TITLE, SUBTITLE, ANGLE, H1, H2, H3.",
      "No markdown.",
      summaryLines,
    ].join("\n"),
  ];

  for (const promptText of promptVariants) {
    try {
      const prepared = await preparePromptInputs(aiSession, promptText);
      if (isPromptTooLongForSession(aiSession, prepared)) {
        continue;
      }
      const generatedText = await generateFromPreparedInputs(aiSession, prepared, {
        maxNewTokens: 220,
        timeoutMs: 150000,
      });
      const frame = parseNewsletterFrameText(generatedText, run);
      if (frame) {
        return frame;
      }
    } catch (error) {
      console.warn("[Chronicle] Newsletter frame generation failed.", error);
    }
  }

  return buildFallbackNewsletterFrame(run);
}

async function generateNewsletterBody(aiSession, run, frame) {
  const opening = await generateNewsletterOpening(aiSession, run, frame);
  if (!opening) {
    return "";
  }

  const sections = [];
  for (const heading of frame.headings.slice(0, 3)) {
    const section = await generateNewsletterSection(aiSession, run, frame, heading, sections);
    if (!section) {
      return "";
    }
    sections.push(section);
  }

  return [
    `# ${frame.title}`,
    "",
    `_${frame.subtitle}_`,
    "",
    opening,
    "",
    ...sections,
  ].join("\n");
}

async function generateNewsletterOpening(aiSession, run, frame) {
  const evidence = selectEvidenceForHeading(run, frame.angle, WRITER_POLICY.openingEvidenceCount);
  const promptVariants = [
    buildOpeningPrompt(run, frame, evidence, false),
    buildOpeningPrompt(run, frame, evidence, true),
  ];

  for (const promptText of promptVariants) {
    try {
      const prepared = await preparePromptInputs(aiSession, promptText);
      if (isPromptTooLongForSession(aiSession, prepared)) {
        continue;
      }
      const generatedText = await generateFromPreparedInputs(aiSession, prepared, {
        maxNewTokens: aiTokenBudget(run, WRITER_POLICY.openingMaxTokens),
        timeoutMs: aiTimeoutBudget(run, WRITER_POLICY.openingAttemptTimeoutMs),
        doSample: true,
        temperature: 0.72,
        topP: 0.92,
      });
      const paragraph = cleanGeneratedProse(generatedText);
      if (isUsableParagraph(paragraph)) {
        return paragraph;
      }
    } catch (error) {
      console.warn("[Chronicle] Opening generation failed.", error);
    }
  }

  return "";
}

async function generateNewsletterSection(aiSession, run, frame, heading, previousSections) {
  const evidence = selectEvidenceForHeading(run, heading, WRITER_POLICY.sectionEvidenceCount);
  const promptVariants = [
    buildSectionPrompt(run, frame, heading, evidence, previousSections, false),
    buildSectionPrompt(run, frame, heading, evidence, previousSections, true),
  ];

  for (const promptText of promptVariants) {
    try {
      const prepared = await preparePromptInputs(aiSession, promptText);
      if (isPromptTooLongForSession(aiSession, prepared)) {
        continue;
      }
      const generatedText = await generateFromPreparedInputs(aiSession, prepared, {
        maxNewTokens: aiTokenBudget(run, WRITER_POLICY.sectionMaxTokens),
        timeoutMs: aiTimeoutBudget(run, WRITER_POLICY.sectionAttemptTimeoutMs),
        doSample: true,
        temperature: 0.74,
        topP: 0.93,
      });
      const sectionMarkdown = normalizeSectionMarkdown(generatedText, heading);
      if (isUsableSectionMarkdown(sectionMarkdown, heading)) {
        return sectionMarkdown;
      }
    } catch (error) {
      console.warn(`[Chronicle] Section generation failed for ${heading}.`, error);
    }
  }

  return "";
}

async function generateWholeNewsletterFallback(aiSession, run, frame) {
  const promptVariants = buildNewsletterPromptVariants(run, frame);

  for (const variant of promptVariants) {
    try {
      const prepared = await preparePromptInputs(aiSession, variant.promptText);
      if (isPromptTooLongForSession(aiSession, prepared)) {
        continue;
      }

      const generatedText = await generateFromPreparedInputs(aiSession, prepared, {
        maxNewTokens: variant.maxNewTokens,
        timeoutMs: variant.timeoutMs,
        doSample: true,
        temperature: 0.74,
        topP: 0.93,
      });

      const cleaned = stripMarkdownFences(generatedText).trim();
      if (isUsableNewsletterMarkdown(cleaned, frame)) {
        return cleaned;
      }
    } catch (error) {
      console.warn("[Chronicle] Whole-issue fallback generation failed.", error);
    }
  }

  return "";
}

function buildEmergencyNewsletterMarkdown(run, frame) {
  const evidence = (run.selectedSummaries?.length ? run.selectedSummaries : run.resultSummaries)
    .slice(0, 6)
    .map((summary) => ({
      ...summary,
      sourceIndex: run.resultSummaries.findIndex((item) => item.result_id === summary.result_id) + 1,
    }));

  const sections = frame.headings.slice(0, 3).map((heading) => {
    const items = selectEvidenceForHeading(run, heading, 2);
    const paragraph = buildEmergencySectionParagraph(items, frame.angle, heading);
    return `## ${heading}\n\n${paragraph}`;
  });

  return [
    `# ${frame.title}`,
    "",
    `_${frame.subtitle}_`,
    "",
    buildEmergencyLeadParagraph(evidence, frame.angle),
    "",
    ...sections,
  ].join("\n");
}

function buildEmergencyLeadParagraph(evidence, angle) {
  const primary = evidence[0];
  const secondary = evidence[1];
  const first = primary ? distillSummary(primary.summary, primary.source_title) : "Chronicle assembled a browser-side evidence packet for this issue.";
  const second = secondary ? distillSummary(secondary.summary, secondary.source_title) : "";
  return [
    trimSentence(`${first} [${primary?.sourceIndex || 1}]`),
    second ? trimSentence(`Taken together, the reporting suggests ${angle.toLowerCase()}. ${second} [${secondary.sourceIndex}]`) : trimSentence(`Taken together, the reporting suggests ${angle.toLowerCase()}.`),
  ].filter(Boolean).join(" ");
}

function buildEmergencySectionParagraph(items, angle, heading) {
  if (!items.length) {
    return `The strongest available evidence around ${heading.toLowerCase()} remains thin, but the reporting still points back to ${angle.toLowerCase()}.`;
  }

  const first = items[0];
  const second = items[1];
  const firstSentence = trimSentence(`${distillSummary(first.summary, first.source_title)} [${first.sourceIndex}]`);
  const secondSentence = second
    ? trimSentence(`${distillSummary(second.summary, second.source_title)} [${second.sourceIndex}]`)
    : "";
  const connective = trimSentence(`The through-line is ${angle.toLowerCase()}, which is why this section matters beyond any single headline.`);

  return [firstSentence, secondSentence, connective].filter(Boolean).join(" ");
}

async function generateStructuredJson(aiSession, promptVariants, options) {
  let lastError = null;

  for (const promptText of promptVariants) {
    try {
      const prepared = await preparePromptInputs(aiSession, promptText);
      if (isPromptTooLongForSession(aiSession, prepared)) {
        continue;
      }

      const generatedText = await generateFromPreparedInputs(aiSession, prepared, {
        maxNewTokens: options.maxNewTokens,
        timeoutMs: options.timeoutMs,
      });
      const parsed = parseJsonBlock(generatedText);
      const validated = options.validate(parsed);
      if (validated) {
        return parsed;
      }
      lastError = new Error("Generated JSON did not pass validation.");
    } catch (error) {
      lastError = error;
    }
  }

  throw new Error(options.errorMessage || lastError?.message || "Chronicle could not generate valid structured output.");
}

function normalizeQueryPlan(value, queryCount) {
  if (!value || typeof value !== "object") {
    return null;
  }

  const title = cleanText(value.title || "");
  const queries = uniqueStrings(value.queries || []).slice(0, queryCount);
  if (!title || queries.length !== queryCount) {
    return null;
  }

  return {
    title,
    queries,
  };
}

function parseSearchPlanText(text, queryCount, brief) {
  const lines = String(stripMarkdownFences(text) || "")
    .split(/\r?\n/)
    .map((line) => cleanText(line))
    .filter(Boolean);

  if (!lines.length) {
    return null;
  }

  let title = "";
  const queries = [];

  lines.forEach((line) => {
    const titleMatch = line.match(/^title\s*:\s*(.+)$/i);
    if (titleMatch && !title) {
      title = cleanText(titleMatch[1]);
      return;
    }

    const queryMatch = line.match(/^q\d+\s*:\s*(.+)$/i);
    if (queryMatch) {
      queries.push(cleanText(queryMatch[1]));
      return;
    }

    if (!title) {
      title = line.replace(/^[-*•]\s*/, "");
      return;
    }

    if (/^[\-\*\u2022]/.test(line) || /\bnews\b|\bweek\b|\blatest\b|\bupdate\b/i.test(line)) {
      queries.push(line.replace(/^[-*•]\s*/, "").replace(/^q\d+\s*:\s*/i, ""));
    }
  });

  const finalTitle = cleanTitle(title || brief);
  const finalQueries = uniqueStrings(queries.map((query) => cleanSearchQuery(query)));
  if (!finalTitle) {
    return null;
  }

  const filledQueries = fillMissingQueries(finalQueries, brief, queryCount);
  if (filledQueries.length !== queryCount) {
    return null;
  }

  return {
    title: finalTitle,
    queries: filledQueries,
  };
}

function buildFallbackQueryPlan(brief, queryCount) {
  return {
    title: cleanTitle(brief),
    queries: fillMissingQueries([], brief, queryCount),
  };
}

function parseNewsletterFrameText(text, run) {
  const lines = String(stripMarkdownFences(text) || "")
    .split(/\r?\n/)
    .map((line) => cleanText(line))
    .filter(Boolean);

  if (!lines.length) {
    return null;
  }

  let title = "";
  let subtitle = "";
  let angle = "";
  const headings = [];

  lines.forEach((line) => {
    const titleMatch = line.match(/^title\s*:\s*(.+)$/i);
    if (titleMatch && !title) {
      title = cleanText(titleMatch[1]);
      return;
    }

    const subtitleMatch = line.match(/^subtitle\s*:\s*(.+)$/i);
    if (subtitleMatch && !subtitle) {
      subtitle = cleanText(subtitleMatch[1]);
      return;
    }

    const angleMatch = line.match(/^angle\s*:\s*(.+)$/i);
    if (angleMatch && !angle) {
      angle = cleanText(angleMatch[1]);
      return;
    }

    const headingMatch = line.match(/^h\d+\s*:\s*(.+)$/i);
    if (headingMatch) {
      headings.push(cleanText(headingMatch[1]));
    }
  });

  title = cleanTitle(title || run.queryPlan?.title || run.config.brief);
  subtitle = cleanText(subtitle);
  angle = cleanText(angle || subtitle);
  const finalHeadings = uniqueStrings(headings.map(cleanText))
    .filter((heading) => !looksGenericHeading(heading))
    .slice(0, 4);

  if (!subtitle || !angle || finalHeadings.length < 2) {
    return null;
  }

  return {
    title,
    subtitle: trimText(subtitle, 180),
    angle: trimText(angle, 160),
    headings: finalHeadings,
  };
}

function buildFallbackNewsletterFrame(run) {
  const focus = cleanTitle(run.queryPlan?.title || run.config.brief);
  const topicPhrase = trimText(cleanText(run.config.brief), 80);
  return {
    title: focus,
    subtitle: `This issue examines how the latest reporting is reshaping the outlook on ${topicPhrase}.`,
    angle: `Build one coherent argument about the current direction of ${topicPhrase}.`,
    headings: [
      `${focus}: current direction`,
      `Where the evidence converges`,
      `What changes next`,
    ],
  };
}

function looksGenericHeading(heading) {
  const text = cleanText(heading).toLowerCase();
  return [
    "what happened",
    "why it matters",
    "key developments",
    "what surfaced first",
    "what the evidence points to",
    "what to watch next",
  ].includes(text);
}

function normalizeResultSummary(value, rawResult) {
  if (!value || typeof value !== "object") {
    return null;
  }

  const summary = trimText(cleanText(value.summary || ""), 260);
  const signal = trimText(cleanText(value.signal || ""), 90);
  const confidenceNote = trimText(cleanText(value.confidence_note || value.confidenceNote || ""), 140);
  if (!summary || !signal || !confidenceNote) {
    return null;
  }
  if (looksTemplateLanguage(summary) || looksTemplateLanguage(signal)) {
    return null;
  }

  if (looksTooCloseToHeadline(summary, rawResult.title) || looksTooCloseToHeadline(signal, rawResult.title)) {
    return null;
  }

  return {
    result_id: cleanText(value.result_id || rawResult.result_id),
    source_title: cleanText(value.source_title || rawResult.title),
    source_url: cleanText(value.source_url || rawResult.url),
    summary,
    signal,
    confidence_note: confidenceNote,
    model_generated: true,
    query: rawResult.query,
    rank: rawResult.rank,
  };
}

function parseResultSummaryText(text, rawResult) {
  const lines = String(stripMarkdownFences(text) || "")
    .split(/\r?\n/)
    .map((line) => cleanText(line))
    .filter(Boolean);

  if (!lines.length) {
    return null;
  }

  let summary = "";
  let signal = "";
  let confidenceNote = "";

  lines.forEach((line) => {
    const summaryMatch = line.match(/^summary\s*:\s*(.+)$/i);
    if (summaryMatch && !summary) {
      summary = cleanText(summaryMatch[1]);
      return;
    }

    const signalMatch = line.match(/^signal\s*:\s*(.+)$/i);
    if (signalMatch && !signal) {
      signal = cleanText(signalMatch[1]);
      return;
    }

    const confidenceMatch = line.match(/^confidence\s*:\s*(.+)$/i);
    if (confidenceMatch && !confidenceNote) {
      confidenceNote = cleanText(confidenceMatch[1]);
    }
  });

  if (!summary && lines[0]) {
    summary = lines[0];
  }
  if (!signal && lines[1]) {
    signal = lines[1];
  }
  if (!confidenceNote) {
    confidenceNote = "This is headline and snippet level evidence, so details may shift.";
  }

  return normalizeResultSummary(
    {
      result_id: rawResult.result_id,
      source_title: rawResult.title,
      source_url: rawResult.url,
      summary,
      signal,
      confidence_note: confidenceNote,
    },
    rawResult,
  );
}

function buildFallbackResultSummary(rawResult) {
  const snippet = cleanText(rawResult.snippet || "");
  const summary = snippet && !looksTooCloseToHeadline(snippet, rawResult.title)
    ? trimText(snippet, 180)
    : trimText(`Reporting mentions a potentially relevant development tied to ${rawResult.query}.`, 180);
  const signal = summarizeHeadlineSignal(rawResult.title, rawResult.query);
  return {
    result_id: rawResult.result_id,
    source_title: rawResult.title,
    source_url: rawResult.url,
    summary,
    signal,
    confidence_note: "This is headline and snippet level evidence, so details may shift.",
    model_generated: false,
    query: rawResult.query,
    rank: rawResult.rank,
  };
}

function isUsableNewsletterMarkdown(markdown, frame) {
  const text = String(markdown || "").trim();
  if (text.length < 220) {
    return false;
  }
  if (!/^#\s+/m.test(text)) {
    return false;
  }
  const headingCount = (text.match(/^##\s+/gm) || []).length;
  if (headingCount < 2) {
    return false;
  }
  if (looksLikeSourceDump(text)) {
    return false;
  }
  if (looksTemplateLanguage(text)) {
    return false;
  }
  if (frame?.headings?.length) {
    const hasAnyPlannedHeading = frame.headings.some((heading) => text.toLowerCase().includes(heading.toLowerCase()));
    if (!hasAnyPlannedHeading) {
      return false;
    }
  }
  return true;
}

function shouldRunPolishPass(markdown) {
  if (!WRITER_POLICY.polishOnPassOnly) {
    return true;
  }
  return looksLikeSourceDump(markdown) || looksTemplateLanguage(markdown) || looksRepetitiveNewsletter(markdown);
}

function looksLikeSourceDump(text) {
  const lines = String(text || "").split(/\r?\n/).map((line) => cleanText(line)).filter(Boolean);
  const repeatedSourceLines = lines.filter((line) => / - (Reuters|CNBC|Fortune|Economic Times|Bloomberg|AP|BBC|The New York Times|CNN|TechCrunch)\b/i.test(line));
  const titleEchoCount = lines.filter((line) => /\.\s+[A-Z][A-Za-z0-9'’"“”\-]/.test(line) && line.length > 90).length;
  return repeatedSourceLines.length >= 3 || titleEchoCount >= 4;
}

function looksTemplateLanguage(text) {
  const value = cleanText(text).toLowerCase();
  return [
    "this source points to a development related to",
    "the strongest signal in this reporting window",
    "the central shift",
    "how the evidence connects",
    "what to watch next",
    "taken together, the reporting suggests synthesize",
  ].some((pattern) => value.includes(pattern));
}

function looksRepetitiveNewsletter(text) {
  const paragraphs = String(text || "")
    .split(/\n{2,}/)
    .map((part) => cleanText(part).toLowerCase())
    .filter((part) => part && !part.startsWith("#"));
  if (paragraphs.length < 3) {
    return false;
  }
  let nearDuplicatePairs = 0;
  for (let leftIndex = 0; leftIndex < paragraphs.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < paragraphs.length; rightIndex += 1) {
      if (paragraphSimilarity(paragraphs[leftIndex], paragraphs[rightIndex]) >= 0.86) {
        nearDuplicatePairs += 1;
      }
    }
  }
  return nearDuplicatePairs >= 2;
}

function paragraphSimilarity(leftText, rightText) {
  const left = buildKeywordSet(leftText);
  const right = buildKeywordSet(rightText);
  if (!left.size || !right.size) {
    return 0;
  }
  const overlap = countTokenOverlap(left, right);
  const union = new Set([...left, ...right]).size;
  return union ? overlap / union : 0;
}

function selectNarrativeSources(resultSummaries, brief) {
  const topicTokens = buildKeywordSet(brief);
  const hasTopicTokens = topicTokens.size > 0;
  const scored = resultSummaries.map((summary, index) => {
    const text = `${summary.source_title} ${summary.summary} ${summary.signal}`;
    const tokens = buildKeywordSet(text);
    const overlapWithTopic = countTokenOverlap(tokens, topicTokens);
    const pairwiseDensity = resultSummaries.reduce((score, other, otherIndex) => {
      if (index === otherIndex) {
        return score;
      }
      return score + Math.min(countTokenOverlap(tokens, buildKeywordSet(`${other.source_title} ${other.summary} ${other.signal}`)), 4);
    }, 0);
    const outlet = extractOutletName(summary.source_title);
    const quality = computeOutletScore(outlet, summary.source_title);
    return {
      ...summary,
      topicOverlap: overlapWithTopic,
      narrativeScore: overlapWithTopic * 4 + pairwiseDensity + quality,
    };
  });

  const filtered = hasTopicTokens
    ? scored.filter((summary) => summary.topicOverlap > 0)
    : scored;

  return (filtered.length ? filtered : scored)
    .sort((left, right) => right.narrativeScore - left.narrativeScore)
    .slice(0, 6);
}

function prioritizeResultsForSummarization(rawResults, brief, summarizeCount) {
  const topicTokens = buildKeywordSet(brief);
  const seenByOutlet = new Set();
  const ranked = rawResults
    .map((result) => {
      const textTokens = buildKeywordSet(`${result.title} ${result.snippet} ${result.query}`);
      const overlap = countTokenOverlap(textTokens, topicTokens);
      const outlet = extractOutletName(result.title);
      const outletQuality = computeOutletScore(outlet, result.title);
      const rankBonus = Math.max(0, 6 - Number(result.rank || 6));
      const score = overlap * 5 + outletQuality + rankBonus;
      return { ...result, _rankScore: score };
    })
    .sort((left, right) => right._rankScore - left._rankScore);

  const selected = [];
  ranked.forEach((result) => {
    if (selected.length >= summarizeCount) {
      return;
    }
    const outlet = extractOutletName(result.title).toLowerCase();
    if (outlet && seenByOutlet.has(outlet) && selected.length + 1 < summarizeCount) {
      return;
    }
    if (outlet) {
      seenByOutlet.add(outlet);
    }
    selected.push(result);
  });

  if (selected.length < summarizeCount) {
    ranked.forEach((result) => {
      if (selected.length >= summarizeCount) {
        return;
      }
      if (!selected.some((item) => item.result_id === result.result_id)) {
        selected.push(result);
      }
    });
  }

  return selected.slice(0, summarizeCount);
}

function buildKeywordSet(text) {
  const stopwords = new Set([
    "the", "and", "for", "with", "from", "that", "this", "have", "will", "into", "about", "after",
    "latest", "update", "news", "week", "more", "says", "report", "amid", "over", "under", "their",
    "they", "them", "than", "what", "when", "where", "which", "while", "into", "across",
  ]);

  return new Set(
    cleanText(text)
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, " ")
      .split(/\s+/)
      .filter((token) => token.length > 2 && !stopwords.has(token)),
  );
}

function countTokenOverlap(left, right) {
  let count = 0;
  left.forEach((token) => {
    if (right.has(token)) {
      count += 1;
    }
  });
  return count;
}

function extractOutletName(sourceTitle) {
  const parts = String(sourceTitle || "").split(" - ");
  return cleanText(parts[parts.length - 1] || "");
}

function computeOutletScore(outlet, title) {
  const premium = ["Reuters", "BBC", "Bloomberg", "Financial Times", "The New York Times", "Wall Street Journal", "AP", "CNBC", "Fortune"];
  const weak = ["LinkedIn", "MarTech", "Economic Times", "Investor's Business Daily"];
  if (premium.includes(outlet)) {
    return 8;
  }
  if (weak.includes(outlet) || /\bnews updates\b|\blatest ai-powered\b/i.test(title)) {
    return -4;
  }
  return 1;
}

function selectEvidenceForHeading(run, heading, count) {
  const focusTokens = buildKeywordSet(`${heading} ${run.config.brief} ${run.queryPlan?.title || ""}`);
  const evidence = (run.selectedSummaries?.length ? run.selectedSummaries : run.resultSummaries)
    .map((summary) => ({
      ...summary,
      focusScore: countTokenOverlap(buildKeywordSet(`${summary.source_title} ${summary.summary} ${summary.signal}`), focusTokens) + (summary.narrativeScore || 0),
    }))
    .sort((left, right) => right.focusScore - left.focusScore)
    .slice(0, count);

  return evidence.map((summary) => ({
    ...summary,
    sourceIndex: run.resultSummaries.findIndex((item) => item.result_id === summary.result_id) + 1,
  }));
}

function buildOpeningPrompt(run, frame, evidence, compact) {
  const lines = evidence
    .map((summary) => (
      `[${summary.sourceIndex}] ${trimText(summary.source_title, compact ? 72 : 92)}\n` +
      `summary: ${trimText(summary.summary, compact ? 96 : 132)}\n` +
      `signal: ${trimText(summary.signal, 56)}`
    ))
    .join("\n\n");

  return [
    "Write the opening paragraph for a Chronicle newsletter.",
    `Topic: ${run.config.brief}`,
    `Mode: ${run.config.explanation_style}`,
    `Title: ${frame.title}`,
    `Subtitle: ${frame.subtitle}`,
    `Angle: ${frame.angle}`,
    `Style instructions: ${buildPresentationInstructions(run.config)}`,
    "",
    "Evidence:",
    lines,
    "",
    "Rules:",
    "- write 1 paragraph only",
    "- make a real argument, not a summary list",
    "- do not repeat source titles verbatim",
    "- use inline citations like [1] when needed",
    "- write only the paragraph",
  ].join("\n");
}

function buildSectionPrompt(run, frame, heading, evidence, previousSections, compact) {
  const lines = evidence
    .map((summary) => (
      `[${summary.sourceIndex}] ${trimText(summary.source_title, compact ? 72 : 92)}\n` +
      `summary: ${trimText(summary.summary, compact ? 96 : 132)}\n` +
      `signal: ${trimText(summary.signal, 56)}`
    ))
    .join("\n\n");
  const previousContext = previousSections
    .map((section) => cleanGeneratedProse(section).slice(0, 180))
    .join("\n");
  const alreadyCovered = extractPriorClaims(previousSections);

  return [
    "Write one section of a Chronicle newsletter.",
    `Topic: ${run.config.brief}`,
    `Mode: ${run.config.explanation_style}`,
    `Title: ${frame.title}`,
    `Subtitle: ${frame.subtitle}`,
    `Angle: ${frame.angle}`,
    `Section heading: ${heading}`,
    `Style instructions: ${buildPresentationInstructions(run.config)}`,
    previousContext ? `Earlier sections:\n${previousContext}` : "",
    alreadyCovered.length ? `Already covered claims:\n${alreadyCovered.map((line, index) => `${index + 1}. ${line}`).join("\n")}` : "",
    "",
    "Evidence:",
    lines,
    "",
    "Rules:",
    "- start with '## {heading}' using the provided heading",
    "- then write 2 or 3 short paragraphs",
    "- advance the argument under this heading",
    "- add at least one new insight that is not already covered",
    "- do not paste or mirror headlines",
    "- do not reuse the same sentence pattern from earlier sections",
    "- do not write a bullet list",
    "- use inline citations like [1] when making claims",
    "- write only markdown for this section",
  ].filter(Boolean).join("\n");
}

function extractPriorClaims(previousSections) {
  const claims = [];
  previousSections.forEach((section) => {
    const lines = cleanGeneratedProse(section)
      .split(/(?<=[.!?])\s+/)
      .map((line) => trimText(cleanText(line), 120))
      .filter((line) => line.length > 40);
    lines.slice(0, 2).forEach((line) => claims.push(line));
  });
  return uniqueStrings(claims).slice(0, 4);
}

function cleanGeneratedProse(text) {
  return cleanText(
    stripMarkdownFences(text)
      .replace(/^#+\s+/gm, "")
      .replace(/^[_*]+|[_*]+$/g, "")
  );
}

function isUsableParagraph(text) {
  const value = String(text || "").trim();
  if (value.length < 120) {
    return false;
  }
  if (looksLikeSourceDump(value)) {
    return false;
  }
  return true;
}

function normalizeSectionMarkdown(text, heading) {
  let cleaned = stripMarkdownFences(text).trim();
  cleaned = cleaned.replace(/^##\s+.+$/m, "").trim();
  return `## ${heading}\n\n${cleaned}`.trim();
}

function isUsableSectionMarkdown(markdown, heading) {
  const value = String(markdown || "").trim();
  if (!value.startsWith(`## ${heading}`)) {
    return false;
  }
  if (value.length < 180) {
    return false;
  }
  if (looksLikeSourceDump(value)) {
    return false;
  }
  if (looksRepetitiveNewsletter(value)) {
    return false;
  }
  return true;
}

async function polishNewsletterDraft(aiSession, run, frame, markdownDraft) {
  const promptVariants = [
    buildPolishPrompt(run, frame, markdownDraft, false),
    buildPolishPrompt(run, frame, markdownDraft, true),
  ];

  for (const promptText of promptVariants) {
    try {
      const prepared = await preparePromptInputs(aiSession, promptText);
      if (isPromptTooLongForSession(aiSession, prepared)) {
        continue;
      }
      const generatedText = await generateFromPreparedInputs(aiSession, prepared, {
        maxNewTokens: aiTokenBudget(run, WRITER_POLICY.polishMaxTokens),
        timeoutMs: aiTimeoutBudget(run, WRITER_POLICY.polishTimeoutMs),
        doSample: true,
        temperature: 0.68,
        topP: 0.9,
      });
      const cleaned = stripMarkdownFences(generatedText).trim();
      if (isUsableNewsletterMarkdown(cleaned, frame) && !looksRepetitiveNewsletter(cleaned)) {
        return cleaned;
      }
    } catch (error) {
      console.warn("[Chronicle] Newsletter polish pass failed.", error);
    }
  }
  return "";
}

function buildPolishPrompt(run, frame, markdownDraft, compact) {
  const draft = trimText(stripSourcesSection(markdownDraft), compact ? 2200 : 3600);
  return [
    "You are Chronicle's final editor.",
    "Rewrite the draft into a publication-quality newsletter with strong flow and minimal repetition.",
    `Topic: ${run.config.brief}`,
    `Mode: ${run.config.explanation_style}`,
    `Style instructions: ${buildPresentationInstructions(run.config)}`,
    `Title must stay: ${frame.title}`,
    "",
    "Draft:",
    draft,
    "",
    "Rules:",
    "- keep factual claims grounded in the cited evidence",
    "- preserve inline citations like [1], [2]",
    "- avoid repeating the same statistic in multiple sections unless necessary",
    "- remove template-like phrasing and headline echoes",
    "- keep one H1 and three H2 sections",
    "- write only markdown and do not include a Sources section",
  ].join("\n");
}

async function ensureBrowserSession() {
  if (state.browserSession?.worker) {
    return state.browserSession;
  }

  if (state.browserSessionPromise) {
    return state.browserSessionPromise;
  }

  state.browserSessionPromise = loadBrowserSession()
    .then(async (session) => {
      state.browserSession = session;
      state.browserRuntimeStatus = "ready";
      const sliceMeta = session.profile?.hasSlices
        ? ` · ${session.profile.label} · num_slices=${session.profile.sliceCount}`
        : "";
      state.browserRuntimeMessage = `${session.runtimeKind === "worker" || session.runtimeKind === "text" ? "Text-only" : "Multimodal"} local model${sliceMeta} · ${runtimeLabelForSession(session)} · self-test passed`;
      state.browserRuntimeProgress = 1;
      state.browserRuntimeProgressText = "Warmup complete. Chronicle is ready.";
      renderHeaderStatus();
      return session;
    })
    .catch((error) => {
      state.browserRuntimeStatus = "error";
      state.browserRuntimeMessage = error.message || "Chronicle could not initialize the browser model.";
      state.browserRuntimeProgressText = "Model warmup failed.";
      renderHeaderStatus();
      throw error;
    })
    .finally(() => {
      state.browserSessionPromise = null;
    });

  return state.browserSessionPromise;
}

async function validateModelSession(aiSession) {
  let lastReply = "";
  try {
    const prepared = await preparePromptInputs(aiSession, "Ready?");
    const reply = await generateFromPreparedInputs(aiSession, prepared, {
      maxNewTokens: 1,
      timeoutMs: 6000,
      doSample: false,
    });
    lastReply = cleanText(reply);
  } catch (error) {
    lastReply = cleanText(error?.message || "");
  }

  if (!lastReply) {
    throw new Error("Model loaded but validation generation failed.");
  }
}

async function loadBrowserSession() {
  const candidates = buildBrowserCandidates();
  let lastError = null;

  for (const candidate of candidates) {
    try {
      beginBrowserLoad(candidate);
      const progressCallback = createBrowserLoadProgressHandler(candidate);
      const session = await withTimeout(
        loadTextOnlyBrowserSession(candidate, progressCallback),
        getCandidateLoadTimeoutMs(candidate),
        `Warmup timed out for ${candidate.label}.`,
      );
      await validateModelSession(session);
      return session;
    } catch (error) {
      lastError = error;
      resetModelWorker();
      state.browserRuntimeMessage = `Retrying with a lighter browser profile after ${candidate.label} failed.`;
      state.browserRuntimeProgressText = "Switching runtime profile…";
      state.browserRuntimeProgress = 0.08;
      renderHeaderStatus();
      console.warn(`[Chronicle] Failed browser candidate ${candidate.label}`, error);
    }
  }

  throw new Error(lastError?.message || "No local browser model bundle could be initialized.");
}

function getCandidateLoadTimeoutMs(candidate) {
  if (!candidate?.hasSlices && candidate?.device === "webgpu") {
    return 60000;
  }
  if (!candidate?.hasSlices && candidate?.device === "wasm") {
    return 180000;
  }
  if (candidate?.device === "webgpu") {
    return 300000;
  }
  return 360000;
}

async function loadTextOnlyBrowserSession(candidate, progressCallback) {
  state.browserRuntimeMessage = `Local model text runtime · ${candidate.label}`;
  state.browserRuntimeProgressText = "Opening text tokenizer…";
  renderHeaderStatus();
  const worker = getOrCreateModelWorker(progressCallback);
  await sendWorkerRequest("init", {
    model: candidate.model,
    device: candidate.device,
    dtype: candidate.textModelOptions?.dtype,
    modelKwargs: candidate.textModelOptions?.model_kwargs || {},
    numThreads: Math.min(4, navigator.hardwareConcurrency || 2),
  }, Math.max(60000, getCandidateLoadTimeoutMs(candidate)));
  return {
    runtimeKind: "worker",
    worker,
    profile: candidate,
    activeSliceCount: candidate.sliceCount || 1,
  };
}

function getOrCreateModelWorker(progressCallback) {
  if (state.modelWorker) {
    return state.modelWorker;
  }

  const worker = new Worker("/static/chronicle_worker.js", { type: "module" });
  worker.onmessage = (event) => {
    const payload = event.data || {};
    if (payload.type === "progress") {
      progressCallback(payload.progress || {});
      return;
    }
    if (payload.type === "error" && !payload.id) {
      console.error("[Chronicle] Worker error:", payload.message || "Unknown worker error");
      return;
    }
    if (!payload.id) {
      return;
    }
    const pending = state.workerPending.get(payload.id);
    if (!pending) {
      return;
    }
    state.workerPending.delete(payload.id);
    if (payload.type === "result" || payload.type === "ready") {
      pending.resolve(payload);
      return;
    }
    pending.reject(new Error(payload.message || "Worker request failed."));
  };
  worker.onerror = (event) => {
    console.error("[Chronicle] Worker crashed.", event);
  };

  state.modelWorker = worker;
  return worker;
}

function resetModelWorker() {
  if (state.modelWorker) {
    state.modelWorker.terminate();
  }
  state.modelWorker = null;
  state.workerPending = new Map();
}

function sendWorkerRequest(type, payload, timeoutMs = 90000) {
  if (!state.modelWorker) {
    throw new Error("Chronicle worker is not initialized.");
  }
  const id = `req-${++state.workerRequestSeq}`;
  const responsePromise = new Promise((resolve, reject) => {
    state.workerPending.set(id, { resolve, reject });
  });
  state.modelWorker.postMessage({ type, id, ...payload });
  return withTimeout(
    responsePromise,
    timeoutMs,
    `Worker request timed out for ${type}.`,
  ).finally(() => {
    state.workerPending.delete(id);
  });
}

async function loadTransformersRuntime() {
  if (state.transformersRuntimePromise) {
    return state.transformersRuntimePromise;
  }

  state.transformersRuntimePromise = import(TRANSFORMERS_CDN)
    .then((runtime) => {
      runtime.env.allowRemoteModels = true;
      runtime.env.allowLocalModels = false;
      runtime.env.useBrowserCache = false;

      if (runtime.env.backends?.onnx?.wasm) {
        runtime.env.backends.onnx.wasm.numThreads = Math.min(2, navigator.hardwareConcurrency || 2);
      }

      return runtime;
    })
    .catch((error) => {
      state.transformersRuntimePromise = null;
      throw error;
    });

  return state.transformersRuntimePromise;
}

function warmBrowserSessionInBackground() {
  if (state.browserWarmStarted) {
    return;
  }

  state.browserWarmStarted = true;
  state.browserRuntimeStatus = "warming";
  state.browserRuntimeMessage = "Preparing the browser model so Chronicle can run fully on device.";
  state.browserRuntimeProgressText = "Loading model files…";
  renderHeaderStatus();
  ensureBrowserSession().catch((error) => {
    console.warn("[Chronicle] Browser warmup failed.", error);
  });
}

function detectBrowserCapabilities() {
  const hasWebGPU = typeof navigator !== "undefined" && Boolean(navigator.gpu);
  const deviceMemory = Number(navigator.deviceMemory || 0);
  const hardwareConcurrency = Number(navigator.hardwareConcurrency || 0);
  const userAgent = typeof navigator !== "undefined" ? String(navigator.userAgent || "") : "";
  const isMobile = typeof navigator !== "undefined"
    && /Android|iPhone|iPad|iPod|Mobile/i.test(userAgent);
  const isHeadlessChrome = /HeadlessChrome/i.test(userAgent);

  return {
    hasWebGPU,
    webgpuLikelyUsable: hasWebGPU && !isHeadlessChrome,
    deviceMemory,
    hardwareConcurrency,
    isMobile,
    availableMemoryGiB: estimateAvailableMemoryGiB(),
  };
}

function calculateBrowserProfile(config) {
  const maxSlices = 36;
  let sliceCount = 1;
  const availableGiB = config.availableMemoryGiB || 0;
  const effectiveGiB = availableGiB > 0
    ? Math.max(1, Math.floor(Math.max(0, availableGiB - 1.5) * 0.75))
    : config.deviceMemory;

  if (config.hasWebGPU) {
    if (config.isMobile) {
      sliceCount = effectiveGiB >= 8 ? 12 : effectiveGiB >= 6 ? 10 : effectiveGiB >= 4 ? 8 : 6;
    } else if (effectiveGiB >= 24 && config.hardwareConcurrency >= 12) {
      sliceCount = 20;
    } else if (effectiveGiB >= 16 && config.hardwareConcurrency >= 8) {
      sliceCount = 16;
    } else if (effectiveGiB >= 12 && config.hardwareConcurrency >= 8) {
      sliceCount = 14;
    } else if (effectiveGiB >= 8) {
      sliceCount = 12;
    } else if (effectiveGiB >= 6) {
      sliceCount = 10;
    } else if (effectiveGiB >= 4) {
      sliceCount = 8;
    } else {
      sliceCount = 6;
    }
  } else {
    sliceCount = effectiveGiB >= 8 ? 8 : effectiveGiB >= 6 ? 7 : effectiveGiB >= 4 ? 6 : 4;
  }

  sliceCount = Math.max(1, Math.min(sliceCount, maxSlices));
  const percentage = Math.round((sliceCount / maxSlices) * 100);
  const sliceRatio = sliceCount / maxSlices;

  return {
    maxSlices,
    sliceCount,
    percentage,
    label: `${percentage}% slice (${sliceCount}/${maxSlices})`,
    availableMemoryGiB: availableGiB,
    reservedMemoryGiB: availableGiB > 0 ? Math.max(0, availableGiB - effectiveGiB) : 0,
    maxInputTokens: sliceRatio >= 0.4 ? 1480 : sliceRatio >= 0.25 ? 1320 : 1160,
    maxNewTokens: sliceRatio >= 0.4 ? 880 : sliceRatio >= 0.25 ? 760 : 660,
    generationTimeoutMs: config.hasWebGPU ? 240000 : 300000,
  };
}

function buildBrowserCandidates() {
  const candidates = [];
  const baseProfile = state.browserProfile || calculateBrowserProfile(state.browserCapabilities || detectBrowserCapabilities());

  MODEL_CANDIDATES.forEach((modelId) => {
    if (isGemma3nModel(modelId)) {
      const sliceFallbacks = buildSliceFallbackChain(baseProfile.sliceCount, baseProfile.maxSlices);
      if (state.browserCapabilities.webgpuLikelyUsable) {
        sliceFallbacks.forEach((sliceCount) => {
          candidates.push(buildBrowserCandidate(modelId, "webgpu", sliceCount, baseProfile));
        });
      }
      const wasmTargetSlices = Math.max(1, Math.min(baseProfile.sliceCount, 2));
      buildSliceFallbackChain(wasmTargetSlices, baseProfile.maxSlices).forEach((sliceCount) => {
        candidates.push(buildBrowserCandidate(modelId, "wasm", sliceCount, baseProfile));
      });
      return;
    }

    // For non-Gemma, try WebGPU first for faster warmup on capable devices, then fallback to WASM.
    if (state.browserCapabilities.webgpuLikelyUsable) {
      candidates.push(buildBrowserCandidate(modelId, "webgpu", 1, baseProfile));
    }
    candidates.push(buildBrowserCandidate(modelId, "wasm", 1, baseProfile));
  });

  return dedupeCandidates(candidates);
}

function buildBrowserCandidate(modelId, device, sliceCount, baseProfile) {
  const sliceLabel = `${Math.round((sliceCount / baseProfile.maxSlices) * 100)}% slice (${sliceCount}/${baseProfile.maxSlices})`;
  const hasSlices = isGemma3nModel(modelId);
  const nonGemmaDtype = resolveNonGemmaDtype(modelId, device);
  const textModelOptions = hasSlices
    ? {
      device,
      dtype: device === "wasm" ? GEMMA3N_TEXT_WASM_DTYPE_MAP : GEMMA3N_TEXT_DTYPE_MAP,
      model_kwargs: { num_slices: sliceCount },
    }
    : { device, dtype: nonGemmaDtype };
  const multimodalModelOptions = hasSlices
    ? {
      device,
      dtype: GEMMA3N_DTYPE_MAP,
      model_kwargs: { num_slices: sliceCount },
    }
    : { device, dtype: nonGemmaDtype };

  return {
    device,
    model: modelId,
    label: hasSlices ? `${device === "webgpu" ? "WebGPU" : "WASM"} ${sliceLabel}` : `${device === "webgpu" ? "WebGPU" : "WASM"}`,
    hasSlices,
    maxSlices: baseProfile.maxSlices,
    sliceCount,
    maxInputTokens: hasSlices ? baseProfile.maxInputTokens : 256,
    maxNewTokens: hasSlices ? baseProfile.maxNewTokens : 32,
    generationTimeoutMs: hasSlices ? baseProfile.generationTimeoutMs : 20000,
    textModelOptions,
    multimodalModelOptions,
  };
}

function resolveNonGemmaDtype(modelId, device) {
  if (device !== "wasm") {
    return "q4";
  }
  if (String(modelId || "").includes("tiny-random-gpt2-ONNX")) {
    return "fp16";
  }
  return "fp32";
}

function buildSliceFallbackChain(targetSlices, maxSlices) {
  const chain = [];
  [targetSlices, Math.ceil(targetSlices * 0.5), 1].forEach((value) => {
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

function getBrowserWarmupTimeoutMs() {
  if (state.browserSession?.worker) {
    return 30000;
  }
  return state.browserCapabilities?.hasWebGPU ? 480000 : 600000;
}

async function waitForBrowserSessionReady() {
  const loadPromise = ensureBrowserSession();
  const deadline = Date.now() + getBrowserWarmupTimeoutMs();

  while (true) {
    const outcome = await Promise.race([
      loadPromise.then((session) => ({ done: true, session })),
      sleep(250).then(() => ({ done: false })),
    ]);

    if (outcome.done) {
      return outcome.session;
    }

    if (Date.now() >= deadline) {
      throw new Error("The model is taking too long to load on this device. Keep the tab open a little longer, then retry.");
    }

    const stalledForMs = Date.now() - (state.browserLastProgressAt || 0);
    if (
      state.browserRuntimeStatus === "warming"
      && state.browserRuntimeProgress >= 0.96
      && stalledForMs > 7000
    ) {
      state.browserRuntimeProgressText = "Files loaded. Compiling model in browser memory (this can take 1-2 minutes).";
      renderHeaderStatus();
    }

    const now = Date.now();
    if (now - (state.lastWarmupUiUpdateAt || 0) >= 800) {
      const detail = getWarmupProgressLabel();
      updateTurnStatus(`Loading Chronicle brain (${detail})`);
      if (state.currentTurn?.stageNode && state.activeStageKey === "query") {
        setStageState("query", "active", detail);
      }
      state.lastWarmupUiUpdateAt = now;
    }
  }
}

function beginBrowserLoad(candidate) {
  state.browserLoadCandidate = candidate.label;
  state.browserLoadProgressEntries = {};
  state.browserLastProgressAt = Date.now();
  state.browserLoadStartedAt = Date.now();
  state.browserRuntimeStatus = "warming";
  state.browserRuntimeProgress = 0;
  state.browserRuntimeMessage = `Local model · ${candidate.label}`;
  state.browserRuntimeProgressText = "Waiting for download telemetry…";
  renderHeaderStatus();
}

function createBrowserLoadProgressHandler(candidate) {
  return (progressInfo) => {
    updateBrowserLoadProgress(candidate, progressInfo);
  };
}

function updateBrowserLoadProgress(candidate, progressInfo = {}) {
  if (state.browserLoadCandidate !== candidate.label) {
    beginBrowserLoad(candidate);
  }

  const key = cleanText(progressInfo.file || progressInfo.name || "");
  const ratio = resolveLoadRatio(progressInfo);
  const status = cleanText(progressInfo.status || "").toLowerCase();
  const loaded = Number(progressInfo.loaded);
  const total = Number(progressInfo.total);

  if (key) {
    const previous = state.browserLoadProgressEntries[key];
    const prevObj = previous && typeof previous === "object" ? previous : {};
    const nextObj = {
      ratio: Number.isFinite(ratio) ? ratio : (Number.isFinite(prevObj.ratio) ? prevObj.ratio : null),
      loaded: Number.isFinite(loaded) ? Math.max(Number(prevObj.loaded || 0), loaded) : Number(prevObj.loaded || 0),
      total: Number.isFinite(total) && total > 0 ? Math.max(Number(prevObj.total || 0), total) : Number(prevObj.total || 0),
    };
    state.browserLoadProgressEntries[key] = nextObj;
  }
  if (
    key
    || Number.isFinite(ratio)
    || (Number.isFinite(loaded) && Number.isFinite(total) && total > 0)
    || ["initiate", "progress", "download", "done", "ready"].includes(status)
  ) {
    state.browserLastProgressAt = Date.now();
  }

  const telemetry = summarizeLoadTelemetry(state.browserLoadProgressEntries);
  const progressRatio = Number.isFinite(telemetry.ratio) ? clampNumber(telemetry.ratio, 0, 1) : null;
  const phaseLabel = describeLoadProgress(progressInfo);

  state.browserRuntimeStatus = "warming";
  state.browserRuntimeMessage = `Local model · ${candidate.label}`;

  if (progressRatio !== null) {
    state.browserRuntimeProgress = progressRatio;
    const percent = Math.round(progressRatio * 100);
    if (status === "done" || status === "ready" || progressRatio >= 0.999) {
      if (telemetry.hasByteTotals) {
        state.browserRuntimeProgressText = `100% files (${formatBytes(telemetry.loadedBytes)} / ${formatBytes(telemetry.totalBytes)}) · Initializing runtime…`;
      } else {
        state.browserRuntimeProgressText = "100% files · Initializing runtime…";
      }
      renderHeaderStatus();
      return;
    }

    if (telemetry.hasByteTotals) {
      const etaText = estimateEtaFromTransferredBytes(state.browserLoadStartedAt, telemetry.loadedBytes, telemetry.totalBytes);
      state.browserRuntimeProgressText = `${percent}% files (${formatBytes(telemetry.loadedBytes)} / ${formatBytes(telemetry.totalBytes)}) · ${phaseLabel}${etaText ? ` · ETA ${etaText}` : ""}`;
    } else {
      const etaText = estimateEtaFromProgress(state.browserLoadStartedAt, progressRatio);
      state.browserRuntimeProgressText = `${percent}% files · ${phaseLabel}${etaText ? ` · ETA ${etaText}` : ""}`;
    }
    renderHeaderStatus();
    return;
  }

  if (status === "done" || status === "ready") {
    state.browserRuntimeProgress = 1;
    state.browserRuntimeProgressText = "File load complete · Initializing runtime…";
  } else {
    state.browserRuntimeProgressText = `${phaseLabel}…`;
  }
  renderHeaderStatus();
}

function resolveLoadRatio(progressInfo) {
  if (Number.isFinite(progressInfo.progress)) {
    return progressInfo.progress > 1 ? progressInfo.progress / 100 : progressInfo.progress;
  }

  if (Number.isFinite(progressInfo.loaded) && Number.isFinite(progressInfo.total) && progressInfo.total > 0) {
    return progressInfo.loaded / progressInfo.total;
  }

  if (["done", "ready"].includes(cleanText(progressInfo.status).toLowerCase())) {
    return 1;
  }

  return null;
}

function describeLoadProgress(progressInfo) {
  const status = cleanText(progressInfo.status || "").toLowerCase();
  const fileName = basename(cleanText(progressInfo.file || progressInfo.name || ""));
  if (fileName) {
    return fileName;
  }
  if (status === "initiate") {
    return "Opening local model files";
  }
  if (status === "progress" || status === "download") {
    return "Loading model shards";
  }
  if (status === "done" || status === "ready") {
    return "Finalizing runtime";
  }
  return "Loading model files";
}

function advanceWarmupTailProgress() {
  // Intentionally no synthetic progress increments.
}

function getWarmupProgressLabel() {
  const percent = Math.round(clampNumber(state.browserRuntimeProgress || 0, 0, 1) * 100);
  const detail = state.browserRuntimeProgressText || "Loading model files";
  return `${percent}% · ${detail.replace(/^\d+%\s·\s/, "")}`;
}

async function preparePromptInputs(aiSession, promptText) {
  if (aiSession.runtimeKind === "worker") {
    return {
      prompt: promptText,
      inputs: null,
      inputLength: estimatePromptTokens(promptText),
    };
  }

  const messages = aiSession.runtimeKind === "text"
    ? [
      {
        role: "user",
        content: promptText,
      },
    ]
    : [
      {
        role: "user",
        content: [{ type: "text", text: promptText }],
      },
    ];
  const prompt = typeof aiSession.codec?.apply_chat_template === "function"
    ? aiSession.codec.apply_chat_template(messages, {
      add_generation_prompt: true,
      tokenize: false,
    })
    : `User: ${String(promptText || "").trim()}\nAssistant:`;
  const inputs = aiSession.runtimeKind === "text"
    ? await aiSession.tokenizer(prompt, { add_special_tokens: false, return_tensor: true })
    : await aiSession.processor(prompt, null, null, { add_special_tokens: false });

  return {
    prompt,
    inputs,
    inputLength: resolveInputLength(inputs.input_ids),
  };
}

async function generateFromPreparedInputs(aiSession, prepared, options) {
  if (aiSession.runtimeKind === "none") {
    throw new Error("Model runtime unavailable for this run.");
  }

  if (aiSession.runtimeKind === "worker") {
    const baseMaxNewTokens = options.maxNewTokens || aiSession.profile.maxNewTokens;
    const baseTimeoutMs = options.timeoutMs || aiSession.profile.generationTimeoutMs;
    const requestGeneration = async (requestOptions, timeoutMs) => sendWorkerRequest(
      "generate",
      {
        prompt: prepared.prompt,
        maxNewTokens: requestOptions.maxNewTokens,
        timeoutMs,
        doSample: Boolean(requestOptions.doSample),
        temperature: requestOptions.doSample ? requestOptions.temperature : undefined,
        topP: requestOptions.doSample ? requestOptions.topP : undefined,
        repetitionPenalty: requestOptions.repetitionPenalty,
      },
      timeoutMs,
    );

    await yieldToBrowser();
    let result;
    try {
      result = await requestGeneration(
        {
          maxNewTokens: baseMaxNewTokens,
          doSample: Boolean(options.doSample),
          temperature: options.doSample ? options.temperature || 0.7 : undefined,
          topP: options.doSample ? options.topP || 0.92 : undefined,
          repetitionPenalty: options.repetitionPenalty || 1.08,
        },
        baseTimeoutMs,
      );
    } catch (error) {
      const message = cleanText(error?.message || "").toLowerCase();
      const retriable = message.includes("timed out") || message.includes("empty response");
      if (!retriable) {
        throw error;
      }
      console.warn("[Chronicle] Retrying slow worker generation with conservative settings.");
      await yieldToBrowser();
      result = await requestGeneration(
        {
          maxNewTokens: Math.max(48, Math.round(baseMaxNewTokens * 0.7)),
          doSample: true,
          temperature: 0.68,
          topP: 0.9,
          repetitionPenalty: 1.08,
        },
        Math.max(baseTimeoutMs, 150000),
      );
    }

    const generatedText = cleanText(result.text || "");
    if (!generatedText) {
      throw new Error("The browser model returned an empty response.");
    }
    await yieldToBrowser();
    return generatedText;
  }

  await yieldToBrowser();
  const output = await withTimeout(
    aiSession.model.generate({
      ...prepared.inputs,
      max_new_tokens: options.maxNewTokens || aiSession.profile.maxNewTokens,
      do_sample: Boolean(options.doSample),
      temperature: options.doSample ? options.temperature || 0.7 : undefined,
      top_p: options.doSample ? options.topP || 0.92 : undefined,
      repetition_penalty: options.repetitionPenalty || 1.08,
    }),
    options.timeoutMs || aiSession.profile.generationTimeoutMs,
    "Browser generation timed out before Chronicle could finish the step.",
  );

  const generatedTokens = output.slice(null, [prepared.inputLength, null]);
  const decoded = aiSession.codec.batch_decode(generatedTokens, {
    skip_special_tokens: true,
  });
  const generatedText = Array.isArray(decoded) ? decoded[0] : decoded;

  if (!generatedText) {
    throw new Error("The browser model returned an empty response.");
  }

  await yieldToBrowser();
  return generatedText;
}

function buildSyntheticRawResults(queryPlan, payload, count) {
  const topic = cleanText(payload?.brief || "the requested topic");
  const queries = Array.isArray(queryPlan?.queries) && queryPlan.queries.length
    ? queryPlan.queries
    : [topic];
  const total = Math.max(1, Number(count || 1));
  const items = [];
  for (let index = 0; index < total; index += 1) {
    const query = queries[index % queries.length] || topic;
    items.push({
      result_id: `synthetic-${index + 1}`,
      query,
      rank: index + 1,
      title: `${cleanTitle(topic)} briefing placeholder`,
      url: `local://synthetic/${index + 1}`,
      snippet: `No live result was available for "${query}". Chronicle generated a deterministic placeholder summary from the query intent.`,
    });
  }
  return items;
}

function estimatePromptTokens(promptText) {
  const text = cleanText(promptText);
  if (!text) {
    return 0;
  }
  const words = text.split(/\s+/).length;
  return Math.ceil(words * 1.4);
}

function isPromptTooLongForSession(aiSession, prepared) {
  if (aiSession?.runtimeKind === "worker") {
    // Worker mode uses an estimate that can be pessimistic; let the model decide at runtime.
    return false;
  }
  return !prepared.inputLength || prepared.inputLength > aiSession.profile.maxInputTokens;
}

function finalizeNewsletterMarkdown(markdown, run) {
  let text = stripMarkdownFences(markdown).trim();
  text = stripSourcesSection(text);

  if (!/^#\s+/m.test(text)) {
    text = `# ${run.queryPlan.title}\n\n${text}`;
  }

  const sourceSummaries = run.selectedSummaries?.length ? run.selectedSummaries : run.resultSummaries;
  return `${text.trim()}\n\n${buildSourcesSection(sourceSummaries, run.resultSummaries)}`.trim();
}

function buildSourcesSection(resultSummaries, allSummaries = resultSummaries) {
  const seen = new Set();
  const lines = [];

  resultSummaries.forEach((summary) => {
    const key = `${summary.source_title}::${summary.source_url}`;
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    const index = allSummaries.findIndex((item) => item.result_id === summary.result_id) + 1;
    lines.push(`[${index}]: ${summary.source_title} - ${summary.source_url}`);
  });

  return `## Sources\n${lines.join("\n")}`;
}

function appendResultCard(result) {
  const turn = ensureCurrentTurn();
  if (turn.completed) {
    return;
  }

  const card = ensureResultNode();
  card.querySelector(".result-title").textContent = result.title;
  card.querySelector(".message-body").innerHTML = "<p>Newsletter is ready and can be opened.</p>";

  const actions = card.querySelector(".result-actions");
  actions.innerHTML = `
    <button type="button" class="result-link result-link--primary" data-action="open">Click here to open</button>
    <button type="button" class="result-link" data-action="download">Download HTML</button>
  `;

  actions.querySelector('[data-action="open"]').addEventListener("click", () => {
    try {
      openHtmlIssue(result);
    } catch (error) {
      appendSystemMessage(error.message || "Chronicle could not open the HTML issue.");
    }
  });

  actions.querySelector('[data-action="download"]').addEventListener("click", () => {
    downloadHtmlIssue(result);
  });

  turn.completed = true;
  scrollThreadToBottom();
}

function openHtmlIssue(result) {
  const popup = window.open("", "_blank");
  if (!popup) {
    throw new Error("The browser blocked the issue window. Allow pop-ups to open the HTML issue.");
  }

  popup.document.open();
  popup.document.write(result.htmlDocument);
  popup.document.close();
}

function downloadHtmlIssue(result) {
  const blob = new Blob([result.htmlDocument], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = result.downloadName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function renderEditableIssueHtml(title, markdown, storageKey) {
  const renderedBody = markdownToHtml(markdown);
  const safeTitle = escapeHtml(title);
  const description = escapeHtml(extractPlainText(markdown).slice(0, 160));
  const editionLabel = new Date().toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${safeTitle}</title>
  <meta name="description" content="${description}">
  <style>
    :root {
      --bg: #09111d;
      --panel: rgba(10, 18, 31, 0.94);
      --panel-soft: rgba(18, 28, 46, 0.92);
      --line: rgba(255, 255, 255, 0.09);
      --ink: #eef3fb;
      --muted: #91a4bf;
      --accent: #ff925d;
      --accent-2: #ffd07f;
      --display: "Ivar Display", "Noe Display", Georgia, serif;
      --ui: "Satoshi", "Avenir Next", sans-serif;
      --mono: "SFMono-Regular", Menlo, monospace;
    }

    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }

    body {
      color: var(--ink);
      font-family: var(--ui);
      background:
        radial-gradient(circle at 12% 14%, rgba(255, 146, 93, 0.18), transparent 24%),
        radial-gradient(circle at 84% 10%, rgba(111, 232, 255, 0.16), transparent 22%),
        linear-gradient(180deg, #07101b 0%, #0a1320 100%);
    }

    .shell {
      width: min(1180px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 18px 0 42px;
    }

    .toolbar {
      position: sticky;
      top: 18px;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 18px;
      border-radius: 22px;
      border: 1px solid var(--line);
      background: rgba(6, 11, 19, 0.84);
      backdrop-filter: blur(18px);
    }

    .toolbar-copy {
      display: grid;
      gap: 4px;
    }

    .kicker {
      margin: 0;
      color: var(--muted);
      font-size: 0.75rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }

    .toolbar-title {
      margin: 0;
      font-size: 1rem;
      font-weight: 700;
    }

    .toolbar-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    button {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 11px 16px;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.05);
      font: inherit;
      cursor: pointer;
    }

    button.primary {
      color: #180f07;
      font-weight: 800;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      border-color: rgba(255, 146, 93, 0.38);
    }

    .card {
      margin-top: 18px;
      border-radius: 30px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, var(--panel), var(--panel-soft));
      overflow: hidden;
    }

    .meta {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      padding: 20px 24px 0;
      color: var(--muted);
      font-size: 0.84rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .editor {
      min-height: 72vh;
      padding: 24px;
      outline: none;
      line-height: 1.72;
      font-size: 1.05rem;
    }

    .editor h1, .editor h2, .editor h3 {
      font-family: var(--display);
      line-height: 1.02;
      letter-spacing: -0.04em;
      margin: 0 0 16px;
    }

    .editor h1 { font-size: clamp(2.8rem, 7vw, 5rem); }
    .editor h2 { font-size: clamp(1.5rem, 3vw, 2.2rem); margin-top: 34px; }
    .editor h3 { font-size: 1.2rem; margin-top: 24px; }
    .editor p { margin: 0 0 16px; color: rgba(238, 243, 251, 0.94); }
    .editor ul { margin: 0 0 18px 20px; padding: 0; }
    .editor li { margin-bottom: 10px; }
    .editor code { font-family: var(--mono); background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 6px; }

    .foot {
      padding: 0 24px 22px;
      color: var(--muted);
      font-size: 0.92rem;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="toolbar">
      <div class="toolbar-copy">
        <p class="kicker">Chronicle HTML editor</p>
        <p class="toolbar-title">${safeTitle}</p>
      </div>
      <div class="toolbar-actions">
        <button type="button" id="restore-button">Restore saved draft</button>
        <button type="button" id="download-button" class="primary">Download HTML</button>
      </div>
    </div>

    <div class="card">
      <div class="meta">
        <span>${escapeHtml(editionLabel)}</span>
        <span>Editable issue</span>
      </div>
      <article id="editor" class="editor" contenteditable="true" spellcheck="true">${renderedBody}</article>
      <div class="foot">This issue lives entirely in your browser. Changes are stored locally on this device.</div>
    </div>
  </div>

  <script>
    const storageKey = ${JSON.stringify(storageKey)};
    const downloadName = ${JSON.stringify(`${slugify(title)}-editable.html`)};
    const editor = document.getElementById("editor");
    const restoreButton = document.getElementById("restore-button");
    const downloadButton = document.getElementById("download-button");

    const cached = window.localStorage.getItem(storageKey);
    if (cached) {
      editor.innerHTML = cached;
    }

    editor.addEventListener("input", () => {
      window.localStorage.setItem(storageKey, editor.innerHTML);
    });

    restoreButton.addEventListener("click", () => {
      const saved = window.localStorage.getItem(storageKey);
      if (saved) {
        editor.innerHTML = saved;
      }
    });

    downloadButton.addEventListener("click", () => {
      const htmlText = document.documentElement.outerHTML;
      const blob = new Blob([htmlText], { type: "text/html;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = downloadName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    });
  </script>
</body>
</html>`;
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
  dismissEmptyState();
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

function appendStageCard() {
  dismissEmptyState();
  const article = document.createElement("article");
  article.className = "message message--assistant";
  article.innerHTML = `
    <div class="stage-card">
      <div class="stage-list">
        ${TURN_STAGES.map((stage) => `
          <div class="stage-item is-pending" data-stage="${stage.key}">
            <div class="stage-index">${stage.index}</div>
            <div class="stage-copy">
              <p class="stage-name">${stage.title}</p>
              <p class="stage-detail">Waiting</p>
            </div>
            <div class="stage-badge">Pending</div>
          </div>
        `).join("")}
      </div>
    </div>
  `;
  elements.chatThread.appendChild(article);
  scrollThreadToBottom();
  return article;
}

function setStageState(stageKey, status, detail = "") {
  const turn = ensureCurrentTurn();
  if (!turn.stageNode) {
    return;
  }

  const row = turn.stageNode.querySelector(`[data-stage="${stageKey}"]`);
  if (!row) {
    return;
  }

  row.classList.remove("is-pending", "is-active", "is-complete", "is-error");
  row.classList.add(`is-${status}`);

  const detailNode = row.querySelector(".stage-detail");
  const badgeNode = row.querySelector(".stage-badge");
  const presets = {
    pending: { detail: "Waiting", badge: "Pending" },
    active: { detail: "Working", badge: "Live" },
    complete: { detail: "Complete", badge: "Done" },
    error: { detail: "Blocked", badge: "Error" },
  };
  const preset = presets[status] || presets.pending;
  detailNode.textContent = detail || preset.detail;
  badgeNode.textContent = preset.badge;
  scrollThreadToBottom();
}

function ensureCurrentTurn() {
  if (!state.currentTurn) {
    startNewTurn("Chronicle");
  }
  return state.currentTurn;
}

function ensureResultNode() {
  const turn = ensureCurrentTurn();
  if (turn.resultNode) {
    return turn.resultNode;
  }

  dismissEmptyState();
  const card = document.createElement("article");
  card.className = "message message--assistant";
  card.innerHTML = `
    <div class="message-card result-card">
      <p class="message-label">Newsletter</p>
      <p class="result-title"></p>
      <div class="result-actions"></div>
      <div class="message-body"></div>
    </div>
  `;
  elements.chatThread.appendChild(card);
  turn.resultNode = card;
  scrollThreadToBottom();
  return card;
}

function finishTurnWithError(message) {
  setGenerateBusy(false);
  if (state.currentTurn?.statusNode) {
    updateTurnStatus("Run failed.");
  }
  if (state.currentTurn?.stageNode && state.activeStageKey) {
    setStageState(state.activeStageKey, "error", "Blocked");
  }
  appendSystemMessage(message);
}

function setGenerateBusy(isBusy) {
  elements.generateButton.disabled = isBusy;
  elements.generateButton.textContent = isBusy ? "Working…" : "Send";
}

function toggleCustomStyleField(visible) {
  elements.customStyleField.classList.toggle("is-hidden", !visible);
}

function setStatusPill(text, modifierClass = "") {
  elements.statusPill.textContent = text;
  elements.statusPill.classList.remove("is-warm", "is-danger");
  if (modifierClass) {
    elements.statusPill.classList.add(modifierClass);
  }
}

function dismissEmptyState() {
  if (elements.emptyState) {
    elements.emptyState.remove();
    elements.emptyState = null;
  }
}

function onChatThreadScroll() {
  state.autoScrollToBottom = isChatThreadNearBottom();
}

function isChatThreadNearBottom() {
  if (!elements.chatThread) {
    return true;
  }
  const remaining = elements.chatThread.scrollHeight - elements.chatThread.scrollTop - elements.chatThread.clientHeight;
  return remaining < 96;
}

function scrollThreadToBottom(force = false) {
  if (!elements.chatThread) {
    return;
  }
  if (!force && !state.autoScrollToBottom) {
    return;
  }
  elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
}

function markdownToHtml(markdown) {
  const lines = stripMarkdownFences(markdown).split(/\r?\n/);
  const html = [];
  let inList = false;

  function closeList() {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
  }

  lines.forEach((line) => {
    const stripped = line.trim();
    if (!stripped) {
      closeList();
      return;
    }

    if (stripped.startsWith("# ")) {
      closeList();
      html.push(`<h1>${formatInlineMarkdown(stripped.slice(2).trim())}</h1>`);
      return;
    }

    if (stripped.startsWith("## ")) {
      closeList();
      html.push(`<h2>${formatInlineMarkdown(stripped.slice(3).trim())}</h2>`);
      return;
    }

    if (stripped.startsWith("### ")) {
      closeList();
      html.push(`<h3>${formatInlineMarkdown(stripped.slice(4).trim())}</h3>`);
      return;
    }

    if (stripped.startsWith("- ")) {
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${formatInlineMarkdown(stripped.slice(2).trim())}</li>`);
      return;
    }

    closeList();
    html.push(`<p>${formatInlineMarkdown(stripped)}</p>`);
  });

  closeList();
  return html.join("\n");
}

function formatInlineMarkdown(text) {
  let value = escapeHtml(text);
  value = value.replace(/`([^`]+)`/g, "<code>$1</code>");
  value = value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  value = value.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return value;
}

function stripSourcesSection(markdown) {
  return String(markdown || "").replace(/\n##\s+Sources[\s\S]*$/i, "").trim();
}

function stripMarkdownFences(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed.startsWith("```")) {
    return trimmed;
  }
  return trimmed.replace(/^```[a-zA-Z0-9_-]*\s*/, "").replace(/\s*```$/, "").trim();
}

function extractTitleFromMarkdown(markdown, fallbackTitle) {
  const match = String(markdown || "").match(/^#\s+(.+)$/m);
  return match ? cleanText(match[1]) : fallbackTitle;
}

function extractPlainText(markdown) {
  return cleanText(
    stripMarkdownFences(markdown)
      .replace(/^#{1,6}\s+/gm, "")
      .replace(/\n+/g, " "),
  );
}

function parseJsonBlock(text) {
  const cleaned = stripMarkdownFences(text).trim();
  const startIndex = cleaned.indexOf("{");
  const endIndex = cleaned.lastIndexOf("}");
  if (startIndex === -1 || endIndex === -1 || endIndex <= startIndex) {
    throw new Error("Chronicle did not return a JSON object.");
  }

  const jsonText = cleaned.slice(startIndex, endIndex + 1);
  return JSON.parse(jsonText);
}

function extractSnippetFromDescription(descriptionHtml, title) {
  if (!descriptionHtml) {
    return "";
  }

  const wrapper = document.createElement("div");
  wrapper.innerHTML = descriptionHtml;
  const snippet = cleanText(wrapper.textContent || "");
  if (!snippet || looksTooCloseToHeadline(snippet, title)) {
    return "";
  }
  return snippet;
}

function uniqueStrings(values) {
  const seen = new Set();
  return values
    .map((value) => cleanText(value))
    .filter((value) => {
      if (!value) {
        return false;
      }
      const key = value.toLowerCase();
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
}

function cleanText(value) {
  return String(value || "")
    .replace(/\s+/g, " ")
    .trim();
}

function trimText(text, maximumLength) {
  const value = String(text || "");
  if (value.length <= maximumLength) {
    return value;
  }
  return `${value.slice(0, maximumLength - 1).trim()}…`;
}

function cleanTitle(value) {
  const text = cleanText(value)
    .replace(/^topic\s*:\s*/i, "")
    .replace(/[.]+$/g, "");
  if (!text) {
    return "Chronicle Issue";
  }
  return text
    .split(/\s+/)
    .slice(0, 8)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function cleanSearchQuery(value) {
  return cleanText(value)
    .replace(/^["'`]+|["'`]+$/g, "")
    .replace(/\s+/g, " ")
    .replace(/[.]+$/g, "");
}

function fillMissingQueries(existingQueries, brief, queryCount) {
  const topic = cleanSearchQuery(brief);
  const keywords = topic
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 2)
    .slice(0, 6);
  const keywordStem = keywords.join(" ");
  const fallbackPool = [
    `${topic} latest news`,
    `${topic} this week`,
    `${topic} latest developments`,
    `${topic} analysis`,
    `${topic} outlook`,
    keywordStem ? `${keywordStem} policy economy` : "",
    keywordStem ? `${keywordStem} Reuters OR AP OR Bloomberg` : "",
    keywordStem ? `${keywordStem} government business technology` : "",
  ].map((query) => cleanSearchQuery(query));

  const merged = uniqueStrings([...existingQueries, ...fallbackPool]).slice(0, queryCount);
  while (merged.length < queryCount) {
    merged.push(`${topic} update ${merged.length + 1}`);
  }
  return merged;
}

function resolveInputLength(inputIds) {
  if (!inputIds) {
    return 0;
  }
  if (typeof inputIds.dims?.at === "function") {
    return inputIds.dims.at(-1) || 0;
  }
  if (Array.isArray(inputIds)) {
    if (Array.isArray(inputIds[0])) {
      return inputIds[0].length || 0;
    }
    return inputIds.length || 0;
  }
  return 0;
}

function slugify(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "chronicle-issue";
}

function normalizeForComparison(value) {
  return cleanText(value)
    .toLowerCase()
    .replace(/\b(reuters|bbc|cnbc|fortune|economic times|the economic times|ap|bloomberg|cnn|techcrunch|dw\.com|dw|nytimes|the new york times)\b/g, " ")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function looksTooCloseToHeadline(candidate, title) {
  const left = normalizeForComparison(candidate);
  const right = normalizeForComparison(title);
  if (!left || !right) {
    return false;
  }
  if (left === right) {
    return true;
  }
  return left.length >= right.length * 0.75 && (left.includes(right) || right.includes(left));
}

function summarizeHeadlineSignal(title, query) {
  const normalized = normalizeForComparison(title);
  const keywords = uniqueStrings(normalized.split(/\s+/)).slice(0, 6);
  if (keywords.length) {
    return trimText(keywords.join(" "), 72);
  }
  return trimText(cleanSearchQuery(query), 72);
}

function distillSummary(summary, sourceTitle) {
  let text = cleanText(summary);
  if (looksTooCloseToHeadline(text, sourceTitle)) {
    text = `This source indicates a material development connected to ${extractOutletName(sourceTitle)}'s reporting.`;
  }
  return trimSentence(text);
}

function trimSentence(text) {
  const value = cleanText(text).replace(/\s+/g, " ");
  if (!value) {
    return "";
  }
  return /[.!?]$/.test(value) ? value : `${value}.`;
}

function basename(value) {
  const text = String(value || "").trim();
  if (!text) {
    return "";
  }
  const parts = text.split(/[\\/]/);
  return parts[parts.length - 1];
}

function runtimeLabelForSession(session) {
  const deviceLabel = session.profile?.device === "webgpu" ? "WebGPU" : "WASM";
  if (!session.profile?.hasSlices) {
    return deviceLabel;
  }
  const activeSlices = session.activeSliceCount || session.profile?.sliceCount || 1;
  const maxSlices = session.profile?.maxSlices || state.browserProfile?.maxSlices || 36;
  return `${deviceLabel} slice ${activeSlices}/${maxSlices}`;
}

function clampNumber(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, Number(value) || 0));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function aiTokenBudget(run, fallback) {
  return Math.min(fallback, state.browserSession?.profile?.maxNewTokens || state.browserProfile?.maxNewTokens || fallback);
}

function aiTimeoutBudget(run, fallback) {
  void run;
  return Math.min(fallback, state.browserSession?.profile?.generationTimeoutMs || state.browserProfile?.generationTimeoutMs || fallback);
}

function estimateLoopEta(startMs, completed, total) {
  if (!startMs || completed <= 0 || total <= completed) {
    return "";
  }
  const elapsedMs = Math.max(1, performance.now() - startMs);
  const avgPerUnitMs = elapsedMs / completed;
  const remainingMs = Math.max(0, Math.round(avgPerUnitMs * (total - completed)));
  return formatEta(remainingMs);
}

function estimateNewsletterEta(startMs, completedSteps, totalSteps) {
  const eta = estimateLoopEta(startMs, completedSteps, totalSteps);
  return eta ? ` · ETA ${eta}` : "";
}

function estimateEtaFromProgress(startedAtMs, progress) {
  const ratio = clampNumber(progress, 0, 1);
  if (!startedAtMs || ratio <= 0.02 || ratio >= 0.995) {
    return "";
  }
  const elapsedMs = Math.max(1, Date.now() - startedAtMs);
  const projectedTotalMs = elapsedMs / ratio;
  const remainingMs = Math.max(0, Math.round(projectedTotalMs - elapsedMs));
  return formatEta(remainingMs);
}

function estimateEtaFromTransferredBytes(startedAtMs, loadedBytes, totalBytes) {
  if (!startedAtMs || !Number.isFinite(loadedBytes) || !Number.isFinite(totalBytes) || loadedBytes <= 0 || totalBytes <= 0 || loadedBytes >= totalBytes) {
    return "";
  }
  const elapsedMs = Math.max(1, Date.now() - startedAtMs);
  const bytesPerMs = loadedBytes / elapsedMs;
  if (!Number.isFinite(bytesPerMs) || bytesPerMs <= 0) {
    return "";
  }
  const remainingBytes = Math.max(0, totalBytes - loadedBytes);
  const remainingMs = Math.round(remainingBytes / bytesPerMs);
  return formatEta(remainingMs);
}

function summarizeLoadTelemetry(entries) {
  const values = Object.values(entries || {});
  let loadedBytes = 0;
  let totalBytes = 0;
  const ratios = [];

  values.forEach((entry) => {
    if (Number.isFinite(entry)) {
      ratios.push(entry);
      return;
    }
    if (!entry || typeof entry !== "object") {
      return;
    }
    if (Number.isFinite(entry.ratio)) {
      ratios.push(entry.ratio);
    }
    if (Number.isFinite(entry.total) && entry.total > 0) {
      totalBytes += entry.total;
      loadedBytes += Math.min(Math.max(0, Number(entry.loaded || 0)), entry.total);
    }
  });

  if (totalBytes > 0) {
    return {
      hasByteTotals: true,
      loadedBytes,
      totalBytes,
      ratio: clampNumber(loadedBytes / totalBytes, 0, 1),
    };
  }

  if (ratios.length) {
    const avgRatio = ratios.reduce((sum, ratio) => sum + ratio, 0) / ratios.length;
    return {
      hasByteTotals: false,
      loadedBytes: 0,
      totalBytes: 0,
      ratio: clampNumber(avgRatio, 0, 1),
    };
  }

  return {
    hasByteTotals: false,
    loadedBytes: 0,
    totalBytes: 0,
    ratio: null,
  };
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const rounded = size >= 100 ? Math.round(size) : size >= 10 ? size.toFixed(1) : size.toFixed(2);
  return `${rounded} ${units[unitIndex]}`;
}

function formatEta(milliseconds) {
  const seconds = Math.max(0, Math.round(milliseconds / 1000));
  if (!seconds) {
    return "";
  }
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
}

function estimateAvailableMemoryGiB() {
  try {
    const perfMemory = performance?.memory;
    if (!perfMemory) {
      return 0;
    }
    const limit = Number(perfMemory.jsHeapSizeLimit || 0);
    const used = Number(perfMemory.usedJSHeapSize || 0);
    if (!limit || !Number.isFinite(limit)) {
      return 0;
    }
    const freeBytes = Math.max(0, limit - used);
    return Number((freeBytes / (1024 ** 3)).toFixed(2));
  } catch (_error) {
    return 0;
  }
}

function withTimeout(promise, timeoutMs, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      window.setTimeout(() => reject(new Error(message)), timeoutMs);
    }),
  ]);
}

function sleep(milliseconds) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}

function yieldToBrowser() {
  return new Promise((resolve) => {
    if (typeof window.requestAnimationFrame === "function") {
      window.requestAnimationFrame(() => resolve());
      return;
    }
    window.setTimeout(resolve, 0);
  });
}

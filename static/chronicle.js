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
    reasoningNode: null,
    resultNode: null,
    completed: false,
  };
}

function startRecoveredTurn() {
  state.currentTurn = {
    statusNode: appendAssistantMessage("Resuming the current local run…"),
    logContainer: appendLogCard(),
    displayedJobLogCount: 0,
    reasoningNode: null,
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
  const resultTitle = run?.title || "Newsletter ready";
  card.querySelector(".result-title").textContent = resultTitle;
  card.querySelector(".message-body").textContent = visibleText;
  card.querySelector(".result-actions").innerHTML = `
    ${run?.html_url ? `<a class="result-link result-link--primary" href="${escapeHtml(run.html_url)}" target="_blank" rel="noreferrer">Open HTML issue</a>` : ""}
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

function updateLiveDraft(text) {
  const card = ensureResultNode();
  card.querySelector(".result-title").textContent = "Draft in progress";
  card.querySelector(".message-body").textContent = text || "Drafting…";
  scrollThreadToBottom();
}

function upsertReasoningSummary(text) {
  const turn = ensureCurrentTurn();
  if (!turn.reasoningNode) {
    turn.reasoningNode = appendAssistantMessage(text, "Reasoning summary");
    return;
  }
  turn.reasoningNode.querySelector(".message-body").textContent = text;
  scrollThreadToBottom();
}

function finishTurnWithError(message) {
  setGenerateBusy(false);
  stopPolling();
  updateTurnStatus("Run failed.");
  appendSystemMessage(message);
}

async function runBrowserGeneration(payload) {
  updateTurnStatus("Fast research stage: planning and Google search…");
  const researchResponse = await fetchJSON("/api/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const research = researchResponse.research;
  appendLogLines(research.logs || buildResearchLogs(research));
  const packet = buildEditorialPacket(research);
  upsertReasoningSummary(buildReasoningSummary(packet, "Evidence selection complete."));
  appendLogLines([
    `Evidence selected: ${packet.selectedSources.length} unique source(s)`,
    `Working thesis: ${packet.workingThesis}`,
    `Editorial lead: ${packet.leadAngle}`,
    `Coverage gaps: ${packet.coverageGap}`,
  ]);

  let markdown = finalizeNewsletterMarkdown(renderFallbackNewsletter(packet), packet);
  let usedFallbackDraft = false;
  let editorialMemo = buildEditorialMemoText(packet.editorialBrief);

  updateLiveDraft(markdown);
  updateTurnStatus("Fast draft prepared. Optional local model refinement is starting now.");
  appendLogLines([
    "Fast draft prepared from the selected evidence.",
    "Loading browser model for optional refinement.",
  ]);
  upsertReasoningSummary(buildReasoningSummary(packet, "Loading browser model for refinement."));

  try {
    const aiSession = await withTimeout(
      ensureBrowserSession(),
      20000,
      "Browser model loading took too long. Using the fast draft instead.",
    );
    updateTurnStatus("Local reasoning stage: building the editorial memo before drafting.");
    appendLogLines([
      `Loading local model: ${aiSession.profile.label}`,
      `Generation backend: ${aiSession.profile.device.toUpperCase()}`,
      `Reasoning budget: ${Math.round(aiSession.profile.reasoningTimeoutMs / 1000)}s`,
      "Local model is building an editorial memo from the collected evidence.",
    ]);
    upsertReasoningSummary(buildReasoningSummary(packet, "Building the local editorial reasoning memo."));

    try {
      editorialMemo = await generateEditorialMemo(research, packet, aiSession);
      appendLogLines([
        "Local reasoning memo complete.",
        `Editorial memo: ${trimText(editorialMemo, 220)}`,
      ]);
      upsertReasoningSummary(buildReasoningSummary(packet, "Editorial reasoning complete. Drafting is starting now.", editorialMemo));
    } catch (memoError) {
      appendLogLines([
        `Editorial memo fallback: ${memoError.message || "Unknown error"}`,
        "Using Chronicle's heuristic editorial brief for the drafting pass.",
      ]);
      editorialMemo = buildEditorialMemoText(packet.editorialBrief);
    }

    updateTurnStatus(`Generating newsletter now. This is the slow stage: ${aiSession.profile.label}.`);
    appendLogLines([
      `Generation budget: ${Math.round(aiSession.profile.generationTimeoutMs / 1000)}s`,
      "Generation started. Waiting here is expected until the draft appears.",
    ]);
    upsertReasoningSummary(buildReasoningSummary(packet, "Generating a polished newsletter draft locally."));

    const generatedMarkdown = await generateNewsletterMarkdown(research, packet, editorialMemo, aiSession, (partialText) => {
      if (partialText.trim()) {
        updateLiveDraft(partialText.trim());
        upsertReasoningSummary(
          buildReasoningSummary(packet, describeGenerationStage(packet, partialText.trim()), editorialMemo),
        );
      }
    });
    markdown = finalizeNewsletterMarkdown(stripMarkdownFences(generatedMarkdown), packet);
  } catch (error) {
    appendLogLines([
      `Model drafting failed: ${error.message || "Unknown error"}`,
      "Switching to deterministic drafting from the collected evidence.",
    ]);
    usedFallbackDraft = true;
    updateLiveDraft(markdown);
    upsertReasoningSummary(buildReasoningSummary(packet, "Local model was too slow, so Chronicle is saving the fast evidence-based draft."));
  }

  const normalizedMarkdown = finalizeNewsletterMarkdown(markdown, packet);
  const title = extractTitleFromMarkdown(normalizedMarkdown, packet.title);

  updateTurnStatus("Saving the newsletter…");
  upsertReasoningSummary(buildReasoningSummary(packet, "Saving the newsletter files now."));
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
  const selectedSources = selectTopSources(research.sources || [], 6);
  const themeTerms = extractThemeTerms(selectedSources, 6);
  const keyEvidence = selectedSources
    .slice(0, 3)
    .map((source) => cleanupSourceTitle(source.title));
  const workingThesis = buildWorkingThesis(research, selectedSources, themeTerms);
  const coverageGap = buildCoverageGap(selectedSources);
  const leadAngle = buildLeadAngle(research, selectedSources, workingThesis);
  const sectionAngles = buildSectionAngles(research.plan?.sections || [], selectedSources);
  const closingInsight = buildClosingInsight(themeTerms, coverageGap);
  const styleGuidance = buildStyleGuidance(research.explanation_style, research.style_instructions || "");
  const editorialBrief = buildHeuristicEditorialBrief(
    research,
    selectedSources,
    workingThesis,
    leadAngle,
    coverageGap,
    themeTerms,
  );

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
    leadAngle,
    sectionAngles,
    closingInsight,
    styleGuidance,
    editorialBrief,
  };
}

function selectTopSources(sources, limit) {
  const deduped = [];
  const seen = new Set();

  sources
    .map((source, index) => ({
      ...source,
      sourceText: trimText(extractEvidenceText(source), 320),
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
  if (source.snippet) {
    score += 2;
  }
  if (source.source_text) {
    score += Math.min(2, source.source_text.length / 220);
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

function buildReasoningSummary(packet, currentStage = "", editorialMemo = "") {
  const editorialPlan = buildEditorialPlanSummary(packet);
  const reasoningMemo = editorialMemo || buildEditorialMemoText(packet.editorialBrief);
  return [
    `Focus: ${packet.brief}`,
    currentStage ? `Current stage: ${currentStage}` : "",
    `Working thesis: ${packet.editorialBrief.coreThesis}`,
    `Hidden pattern: ${packet.editorialBrief.hiddenPattern}`,
    `Lead angle: ${packet.leadAngle}`,
    editorialPlan ? `Editorial plan: ${editorialPlan}` : "",
    reasoningMemo ? `Reasoning memo: ${reasoningMemo}` : "",
    `Evidence selected: ${packet.selectedSources.length} unique source(s)`,
    `Key evidence: ${packet.keyEvidence.join(" | ") || "No source headlines yet"}`,
    `Coverage gap: ${packet.coverageGap}`,
  ].filter(Boolean).join("\n");
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
    packet.editorialBrief.coreThesis,
    "",
    packet.editorialBrief.hiddenPattern,
    "",
  ];

  packet.sections.forEach((sectionName, index) => {
    lines.push(`## ${sectionName}`);
    lines.push(buildSectionLead(packet, sectionName, index));
    lines.push("");
    lines.push(buildSectionParagraph(packet, sectionName, index));
    lines.push("");
    lines.push(buildSectionImplication(packet, sectionName, index));
    lines.push("");
  });

  lines.push("## Closing note");
  lines.push(packet.editorialBrief.killerInsight);
  lines.push("");
  lines.push(packet.closingInsight);
  lines.push("");
  lines.push(buildSourcesSection(packet));
  return lines.join("\n");
}

function buildLeadAngle(research, selectedSources, workingThesis) {
  if (!selectedSources.length) {
    return `This briefing is running with limited verification, so the safest usable angle is to frame what is known, what is not, and why the gap matters.`;
  }
  return `${workingThesis} The goal of this issue is not to list every headline, but to isolate the directional shift that matters most for readers tracking ${research.brief}.`;
}

function buildSectionAngles(sections, selectedSources) {
  return sections.map((sectionName, index) => {
    const source = selectedSources[index] || selectedSources[0];
    if (!source) {
      return `The most honest move in ${sectionName.toLowerCase()} is to stay explicit about what the evidence does and does not support.`;
    }
    return `${cleanupSourceTitle(source.title)} suggests the section should focus on the highest-signal implication rather than repeating the headline.`;
  });
}

function buildClosingInsight(themeTerms, coverageGap) {
  if (themeTerms.length >= 3) {
    return `The deeper pattern is the clustering of ${themeTerms.slice(0, 3).join(", ")} in the same reporting window. ${coverageGap}`;
  }
  return `The most useful takeaway is the directional signal, not the volume of headlines. ${coverageGap}`;
}

function buildEditorialPlanSummary(packet) {
  const sectionPlan = packet.sections
    .slice(0, 3)
    .map((sectionName, index) => `${sectionName}: ${packet.sectionAngles[index] || packet.workingThesis}`)
    .join(" | ");
  return sectionPlan;
}

function buildEditorialMemoText(editorialBrief) {
  if (!editorialBrief) {
    return "";
  }
  return [
    `Core thesis: ${editorialBrief.coreThesis}`,
    `Hidden pattern: ${editorialBrief.hiddenPattern}`,
    `Killer insight: ${editorialBrief.killerInsight}`,
    `Writing approach: ${editorialBrief.writingApproach}`,
  ].join(" | ");
}

function describeGenerationStage(packet, partialText) {
  const headingMatches = partialText.match(/^##\s+/gm) || [];
  const headingCount = headingMatches.length;
  const cleaned = partialText.trim();

  if (!cleaned) {
    return "Preparing the local writing pass.";
  }
  if (headingCount === 0 && cleaned.length < 220) {
    return "Writing the headline and opening angle.";
  }
  if (headingCount < Math.max(1, Math.min(packet.sections.length, 2))) {
    return `Drafting the first body section around ${packet.sections[0] || "the lead theme"}.`;
  }
  if (headingCount < packet.sections.length) {
    return `Drafting the middle sections. ${headingCount}/${packet.sections.length} section headings are in place.`;
  }
  if (!/\n##\s+Closing note\b/i.test(partialText)) {
    return "Writing the closing synthesis and tightening the throughline.";
  }
  if (!/\n##\s+Sources\b/i.test(partialText)) {
    return "Assembling the source notes and final structure.";
  }
  return "Finalizing the local draft and checking citation structure.";
}

function extractEvidenceText(source) {
  const raw = String(source?.source_text || source?.article_text || source?.snippet || "").trim();
  if (!raw) {
    return cleanupSourceTitle(source?.title || "");
  }

  const snippetMatch = raw.match(/Snippet:\s*([\s\S]+)/i);
  if (snippetMatch?.[1]) {
    return cleanEvidenceText(snippetMatch[1]);
  }

  return cleanEvidenceText(raw);
}

function cleanEvidenceText(text) {
  return String(text || "")
    .replace(/^Title:\s*/gim, "")
    .replace(/^URL:\s*\S+\s*$/gim, "")
    .replace(/^Snippet:\s*/gim, "")
    .replace(/\s+/g, " ")
    .trim();
}

function buildStyleGuidance(explanationStyle, customStyleInstructions) {
  if (explanationStyle === "custom" && customStyleInstructions.trim()) {
    return customStyleInstructions.trim();
  }
  if (explanationStyle === "feynman") {
    return "Explain the story with plain language, clean causality, and minimal jargon. Make complexity feel understandable.";
  }
  if (explanationStyle === "soc") {
    return "Lead with the right questions, answer them step by step, and let each section deepen the reader's understanding.";
  }
  return "Be concise, selective, and high-signal. Prefer strong interpretation over padded recap.";
}

function buildHeuristicEditorialBrief(research, selectedSources, workingThesis, leadAngle, coverageGap, themeTerms) {
  const strongestSignal = cleanupSourceTitle(selectedSources[0]?.title || research.brief);
  const secondarySignal = cleanupSourceTitle(selectedSources[1]?.title || "");
  const pairedThemes = themeTerms.slice(0, 2).join(" and ");
  const dominantFrame = pairedThemes || research.brief;

  return {
    coreThesis: selectedSources.length
      ? `${leadAngle} The strongest usable reading is that ${strongestSignal} matters because it changes how readers should interpret the broader direction of ${research.brief}.`
      : workingThesis,
    hiddenPattern: secondarySignal
      ? `The pattern beneath the headlines is convergence: ${strongestSignal} and ${secondarySignal} are not random adjacent stories, but signs that ${dominantFrame} is becoming the frame that organizes the week.`
      : `The hidden pattern is that the same topic is appearing across multiple angles, which usually matters more than any single headline on its own.`,
    killerInsight: `The right takeaway is not that the feed was busy. It is that the reporting window points toward a clearer narrative center, and that center is ${dominantFrame}.`,
    writingApproach: buildStyleGuidance(research.explanation_style, research.style_instructions || ""),
    proofPoints: selectedSources
      .slice(0, 3)
      .map((source) => cleanupSourceTitle(source.title)),
    coverageGap,
  };
}

function buildSectionLead(packet, sectionName, index) {
  if (packet.explanationStyle === "soc") {
    return `The right question in ${sectionName.toLowerCase()} is this: what actually moved, and why should the reader care?`;
  }
  if (packet.explanationStyle === "feynman") {
    return `The simplest way to read ${sectionName.toLowerCase()} is to separate the visible headline from the deeper shift underneath it.`;
  }
  return `${packet.sectionAngles[index] || packet.editorialBrief.coreThesis}`;
}

function buildSectionParagraph(packet, sectionName, index) {
  const primarySource = packet.selectedSources[index] || packet.selectedSources[0];
  const secondarySource = packet.selectedSources[index + 1] || packet.selectedSources[1];
  const sectionAngle = packet.sectionAngles[index] || packet.editorialBrief.coreThesis;

  if (!primarySource) {
    return "Chronicle could not verify enough fresh reporting for this section, so this draft stays explicit about that gap.";
  }

  const primaryNote = trimText(primarySource.sourceText || primarySource.snippet || "", 180);
  const primaryCitation = `[${Math.min(index + 1, packet.selectedSources.length)}]`;
  const secondaryCitation = secondarySource ? `[${Math.min(index + 2, packet.selectedSources.length)}]` : "";
  const corroboration = secondarySource
    ? ` A second signal from ${cleanupSourceTitle(secondarySource.title)} sharpens the same picture rather than overturning it. ${secondaryCitation}`
    : "";

  if (packet.explanationStyle === "soc") {
    return `What does the evidence actually say? Start with ${cleanupSourceTitle(primarySource.title)}: ${primaryNote} ${primaryCitation}${corroboration}`.trim();
  }
  if (packet.explanationStyle === "feynman") {
    return `${sectionAngle} In plain terms, ${cleanupSourceTitle(primarySource.title)} suggests ${primaryNote} ${primaryCitation}${corroboration}`.trim();
  }
  return `${sectionAngle} The clearest anchor is ${cleanupSourceTitle(primarySource.title)}, which suggests ${primaryNote} ${primaryCitation}${corroboration}`.trim();
}

function buildSectionImplication(packet, sectionName, index) {
  const primarySource = packet.selectedSources[index] || packet.selectedSources[0];
  const themeA = packet.themeTerms[0] || "policy";
  const themeB = packet.themeTerms[1] || "execution";
  if (!primarySource) {
    return `The implication for ${sectionName.toLowerCase()} is that readers should treat this as a watchlist item until stronger verification arrives.`;
  }

  if (packet.explanationStyle === "soc") {
    return `Why does that matter? Because the real signal is not the isolated event itself, but the way ${themeA} is starting to interact with ${themeB}. That is the frame the rest of the issue should answer.`;
  }
  if (packet.explanationStyle === "feynman") {
    return `Why it matters: this is easier to understand if you treat ${themeA} and ${themeB} as connected pieces of the same story. Once they start moving together, the week stops looking random.`;
  }
  return `Why this matters: the section is less about one isolated update and more about how ${themeA} is beginning to interact with ${themeB}. For readers following ${packet.brief}, that is the durable signal worth carrying forward from this reporting window.`;
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

async function generateNewsletterMarkdown(research, packet, editorialMemo, aiSession, onProgress) {
  const messages = [
    {
      role: "user",
      content: [
        {
          type: "text",
          text: buildNewsletterPrompt(research, packet, editorialMemo),
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
      repetition_penalty: 1.06,
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

async function generateEditorialMemo(research, packet, aiSession) {
  const messages = [
    {
      role: "user",
      content: [
        {
          type: "text",
          text: buildEditorialMemoPrompt(research, packet),
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
    throw new Error("Gemma 3n processor did not return input ids for editorial reasoning.");
  }

  const output = await withTimeout(
    aiSession.model.generate({
      ...inputs,
      max_new_tokens: aiSession.profile.reasoningMaxNewTokens,
      do_sample: false,
      repetition_penalty: 1.04,
    }),
    aiSession.profile.reasoningTimeoutMs,
    "Browser reasoning timed out before Chronicle could finish the editorial memo.",
  );
  const generatedTokens = output.slice(null, [inputLength, null]);
  const decoded = aiSession.processor.batch_decode(generatedTokens, {
    skip_special_tokens: true,
  });
  const generatedText = Array.isArray(decoded) ? decoded[0] : decoded;
  const cleaned = stripMarkdownFences(String(generatedText || "")).trim();
  if (!cleaned) {
    throw new Error("Editorial memo came back empty.");
  }
  return cleaned;
}

function buildEditorialMemoPrompt(research, packet) {
  const sourceBundle = packet.selectedSources
    .map((source, index) => {
      return [
        `[${index + 1}] ${cleanupSourceTitle(source.title)}`,
        `Notes: ${trimText(source.sourceText || source.snippet || "", 220)}`,
      ].join("\n");
    })
    .join("\n\n");

  return `You are Chronicle's editorial strategist.

Before drafting the newsletter, think through the evidence and return a short editorial memo for the writer.

Brief:
${research.brief}

Audience:
${research.plan.audience}

Tone:
${research.plan.tone}

Explanation style:
${research.explanation_style}

Style guidance:
${packet.styleGuidance}

Planned sections:
${JSON.stringify(packet.sections)}

Evidence:
${sourceBundle || "No sources collected."}

Return plain text only in exactly four lines:
Core thesis: ...
Hidden pattern: ...
Killer insight: ...
Writing approach: ...

Rules:
- make the thesis defensible from the evidence
- identify the pattern linking the sources together
- make the killer insight memorable, not generic
- adapt the writing approach to the requested explanation style
- do not mention prompts, tools, or implementation details`;
}

function buildNewsletterPrompt(research, packet, editorialMemo) {
  const sourceBundle = packet.selectedSources
    .map((source, index) => {
      const excerpt = trimText(source.sourceText || "", 260);
      return [
        `[${index + 1}] ${cleanupSourceTitle(source.title)}`,
        `Query: ${source.query}`,
        `Evidence notes: ${excerpt}`,
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

Think through the evidence silently first, then write one complete markdown newsletter.

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

Requested explanation guidance:
${packet.styleGuidance}

${customStyle}

Coverage window:
Last ${packet.days} days

Planned sections:
${JSON.stringify(packet.sections)}

Editorial reasoning memo:
${editorialMemo}

Heuristic editorial brief:
${JSON.stringify(packet.editorialBrief)}

Section plan:
${JSON.stringify(packet.sections.map((section, index) => ({
  section,
  angle: packet.sectionAngles[index] || packet.editorialBrief.coreThesis,
})))}

Structured market data:
${marketData}

Research notes:
${sourceBundle || "No sources were collected."}

Requirements:
- return markdown only
- start with one H1 title
- aim for roughly 700 to 950 words
- write a sharp opening note with a real point of view, not a generic summary
- organize the body around the planned sections using H2 headings
- use the reasoning summary to keep a strong throughline instead of just listing events
- make each section advance the argument, not repeat the headline
- synthesize at least two source notes when the evidence allows it
- keep the writing analytical, premium, and specific
- explain why the developments matter for a reader, not just what happened
- if evidence is thin, say so cleanly instead of inventing details
- do not copy raw source notes, source labels, URLs, or "Title/URL/Snippet" text into the body
- convert evidence into clean prose
- honor the requested explanation style throughout the whole piece
- for concise, compress aggressively and keep paragraphs tight
- for feynman, explain complexity in plain language
- for soc, structure the flow around the right questions and answers
- for custom, follow the custom guidance over the default explanation modes
- cite source notes inline like [1], [2]
- if market data is present, reference it as [M1]
- end with a Sources section
- in Sources, list each source as [id]: title - url
- if market data is present, include: [M1]: CoinGecko Markets API - https://www.coingecko.com/
- do not use markdown code fences
- do not mention system prompts or implementation details
- do not write meta phrases like "here is the newsletter" or "based on the sources"
- do not expose your hidden reasoning process; only output the final newsletter`;
}

function calculateBrowserProfile(config) {
  const maxSlices = Math.max(1, Number(config.maxSlices || 1));
  const supportsSlicing = Boolean(config.supportsSlicing && maxSlices > 1);

  if (!supportsSlicing) {
    return {
      sliceCount: 1,
      percentage: 100,
      label: "Single bundle",
      maxNewTokens: config.hasWebGPU ? 700 : 520,
      reasoningMaxNewTokens: 220,
      reasoningTimeoutMs: config.hasWebGPU ? 90000 : 75000,
      generationTimeoutMs: config.hasWebGPU ? 240000 : 210000,
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
  let maxNewTokens = 520;
  if (percentage >= 50) {
    maxNewTokens = 760;
  } else if (percentage >= 25) {
    maxNewTokens = 620;
  }

  return {
    sliceCount,
    percentage,
    label: `${percentage}% slice (${sliceCount}/${maxSlices})`,
    maxNewTokens,
    reasoningMaxNewTokens: 220,
    reasoningTimeoutMs: percentage >= 50 ? 100000 : 80000,
    generationTimeoutMs: percentage >= 50 ? 240000 : 210000,
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
    ? Math.max(240, recommendedProfile.maxNewTokens - Math.max(recommendedProfile.sliceCount - normalizedSliceCount, 0) * 40)
    : Math.min(220, recommendedProfile.maxNewTokens);
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
      ? Math.max(70000, recommendedProfile.generationTimeoutMs - Math.max(recommendedProfile.sliceCount - normalizedSliceCount, 0) * 10000)
      : Math.min(70000, recommendedProfile.generationTimeoutMs),
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

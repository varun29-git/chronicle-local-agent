const state = {
  runs: [],
  selectedRunId: null,
  activeJobId: null,
  pollTimer: null,
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
    renderJobState({
      status: "queued",
      message: "Queued on this device",
      params: payload,
    });

    try {
      const response = await fetchJSON("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      state.activeJobId = response.job.id;
      renderJobState(response.job);
      startPolling(response.job.id);
    } catch (error) {
      renderJobError(error.message || "Chronicle could not start the run.");
      setGenerateBusy(false);
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

  renderRuntime(statusResponse.runtime);
  renderRuns(runsResponse.runs || []);

  if (statusResponse.active_job) {
    state.activeJobId = statusResponse.active_job.id;
    renderJobState(statusResponse.active_job);
    startPolling(statusResponse.active_job.id);
  } else {
    renderJobIdle();
  }
}

function renderRuntime(runtime) {
  const dependenciesReady = Boolean(runtime.dependencies_ready);
  const modelReady = Boolean(runtime.model_ready);
  if (!dependenciesReady) {
    setBadge(elements.runtimeBadge, "Deps missing", "is-danger");
  } else if (!modelReady) {
    setBadge(elements.runtimeBadge, "Model missing", "is-warm");
  } else {
    setBadge(elements.runtimeBadge, "Ready", "");
  }
  elements.runtimeHeadline.textContent =
    `Chronicle is using ${runtime.hostname} as the newsroom engine.`;
  if (!dependenciesReady) {
    elements.runtimeSubline.textContent =
      `${runtime.dependency_message}. Start Chronicle with the correct local runtime environment on this machine before generating issues.`;
  } else if (!modelReady) {
    elements.runtimeSubline.textContent =
      "The site is live, but the local model path is missing. Chronicle will stay local-first once the model directory is placed on this machine.";
  } else {
    elements.runtimeSubline.textContent =
      "The runtime is pointed at a local model path, so the issue generation pipeline can stay on-device.";
  }

  elements.statBackend.textContent = upper(runtime.runtime_backend);
  elements.statSlice.textContent = runtime.slice_label || "—";
  elements.statMemory.textContent = runtime.memory_total_gb
    ? `${runtime.memory_total_gb.toFixed(1)} GB`
    : "Unknown";
  elements.statModelReady.textContent = modelReady ? "Ready" : "Missing";

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
    "When a run starts, this panel will track it and point the preview to the finished issue.";
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
      ? `Chronicle finished “${result.title}” on this device.`
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
    "The research, drafting, and output writing flow is happening on the same device hosting this site.";
}

function renderJobError(message) {
  setGenerateBusy(false);
  stopPolling();
  setBadge(elements.jobBadge, "Failed", "is-danger");
  elements.jobMessage.textContent = message;
  elements.jobSubcopy.textContent =
    "Fix the issue, then launch another run. Common blockers are missing local model files or runtime dependencies.";
}

function setGenerateBusy(isBusy) {
  elements.generateButton.disabled = isBusy;
  elements.generateButton.textContent = isBusy
    ? "Generating on this device…"
    : "Generate on this device";
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

async function syncAfterJob(job) {
  const [statusResponse, runsResponse] = await Promise.all([
    fetchJSON("/api/status"),
    fetchJSON("/api/runs?limit=30"),
  ]);
  renderRuntime(statusResponse.runtime);
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

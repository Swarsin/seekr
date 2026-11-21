/*
  Seekr local workspace.

  Security rules followed throughout this file:
  - The session token lives only in the `session` closure below. It is never
    written to a URL, the DOM, storage, or browser logs.
  - Every server-provided string (paths, locators, snippets, messages) is
    rendered with textContent / text nodes. Highlighting builds <mark>
    elements from text nodes; no HTML strings are parsed.
  - All requests go to this page's own origin.
*/
(function () {
  "use strict";

  var POLL_MS = 1000;
  var DEFAULT_K = 10;
  var DEFAULT_PER_FILE = 2;

  var session = { token: null, pending: null, limits: null };

  var state = {
    connected: false,
    everConnected: false,
    status: null,
    roots: null,
    rootsError: null,
    job: null,
    jobTimer: null,
    lastJob: null,
    searchSeq: 0,
    searchInFlight: false,
    searchQueued: false,
    searchShown: false,
    searchError: null,
    forcedFromMode: null,
    view: "search",
    addPending: false,
    statusSeq: 0,
    statusError: null,
    rootsSeq: 0,
    newRoot: null,
    pollInFlight: false,
    finishedJobs: Object.create(null),
  };

  var KIND_LABELS = {
    pdf: "PDF",
    markdown: "Markdown",
    text: "Plain text",
    code: "Source code",
    transcript: "Transcript",
  };

  // ---------------------------------------------------------------- DOM utils

  function byId(id) {
    return document.getElementById(id);
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    return node;
  }

  function append(parent) {
    for (var i = 1; i < arguments.length; i += 1) {
      var child = arguments[i];
      if (child === null || child === undefined || child === false) {
        continue;
      }
      if (typeof child === "string") {
        parent.appendChild(document.createTextNode(child));
      } else {
        parent.appendChild(child);
      }
    }
    return parent;
  }

  function clear(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function show(node, visible) {
    node.hidden = !visible;
  }

  function setText(node, text) {
    node.textContent = text === null || text === undefined ? "" : String(text);
    show(node, Boolean(node.textContent));
  }

  function codeText(text) {
    return el("code", null, text);
  }

  function announce(message) {
    var region = byId("announcer");
    region.textContent = "";
    // A separate task so screen readers notice repeated identical messages.
    window.requestAnimationFrame(function () {
      region.textContent = message;
    });
  }

  function button(label, className, onClick) {
    var b = el("button", "button " + (className || "button-quiet"), label);
    b.type = "button";
    b.addEventListener("click", onClick);
    return b;
  }

  function notice(tone, title, bodies, actions) {
    var box = el("div", "notice notice-" + tone);
    if (title) {
      append(box, el("p", "notice-title", title));
    }
    (bodies || []).forEach(function (body) {
      if (body === null || body === undefined || body === "") {
        return;
      }
      var p = el("p", "notice-body");
      if (Array.isArray(body)) {
        body.forEach(function (part) {
          append(p, part);
        });
      } else {
        append(p, body);
      }
      append(box, p);
    });
    if (actions && actions.length) {
      var row = el("div", "notice-actions");
      actions.forEach(function (a) {
        append(row, a);
      });
      append(box, row);
    }
    return box;
  }

  // ---------------------------------------------------------------- formatting

  function str(value) {
    return typeof value === "string" ? value : "";
  }

  function isCount(value) {
    return typeof value === "number" && isFinite(value) && value >= 0;
  }

  function formatCount(value) {
    return isCount(value) ? Math.round(value).toLocaleString() : "";
  }

  function plural(n, one, many) {
    return formatCount(n) + " " + (n === 1 ? one : many);
  }

  function formatBytes(bytes) {
    if (!isCount(bytes)) {
      return "";
    }
    var units = ["bytes", "KB", "MB", "GB", "TB"];
    var value = bytes;
    var unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    return (unit === 0 ? String(value) : value.toFixed(value < 10 ? 1 : 0)) + " " + units[unit];
  }

  function formatElapsed(ms) {
    if (!isCount(ms)) {
      return "";
    }
    if (ms < 1000) {
      return (ms / 1000).toFixed(1) + " s";
    }
    var seconds = Math.floor(ms / 1000);
    if (seconds < 60) {
      return (ms < 10000 ? (ms / 1000).toFixed(1) : String(seconds)) + " s";
    }
    var minutes = Math.floor(seconds / 60);
    var rest = seconds % 60;
    if (minutes < 60) {
      return minutes + " min " + (rest < 10 ? "0" : "") + rest + " s";
    }
    var hours = Math.floor(minutes / 60);
    return hours + " h " + (minutes % 60) + " min";
  }

  function formatTook(ms) {
    if (typeof ms !== "number" || !isFinite(ms) || ms < 0) {
      return "";
    }
    if (ms < 1000) {
      return Math.max(1, Math.round(ms)) + " ms";
    }
    return (ms / 1000).toFixed(1) + " s";
  }

  function baseName(path) {
    var trimmed = path.replace(/[\\/]+$/, "");
    var parts = trimmed.split(/[\\/]/);
    var last = parts[parts.length - 1];
    return last || path;
  }

  // ---------------------------------------------------------------- API layer

  function ApiFailure(kind, status, code, message, fields) {
    this.kind = kind; // "network" | "http" | "aborted" | "protocol"
    this.status = status || 0;
    this.code = code || kind;
    this.message = message || "";
    this.fields = fields && typeof fields === "object" ? fields : null;
  }

  function isAbort(err) {
    return Boolean(err) && (err.name === "AbortError" || err.kind === "aborted");
  }

  async function readJson(response) {
    try {
      return await response.json();
    } catch (err) {
      if (isAbort(err)) {
        throw new ApiFailure("aborted");
      }
      return null;
    }
  }

  function failureFrom(response, data) {
    var info = data && typeof data === "object" && data.error && typeof data.error === "object" ? data.error : null;
    var code = info && typeof info.code === "string" ? info.code : "http_" + response.status;
    var message = info && typeof info.message === "string" ? info.message : "";
    var fields = info && info.fields && typeof info.fields === "object" ? info.fields : null;
    return new ApiFailure("http", response.status, code, message, fields);
  }

  async function bootstrapSession() {
    var response;
    try {
      response = await fetch("/api/session", {
        method: "GET",
        headers: { "X-Seekr-Client": "1", Accept: "application/json" },
        cache: "no-store",
        credentials: "same-origin",
        redirect: "error",
      });
    } catch (err) {
      throw new ApiFailure("network", 0, "network", "");
    }
    var data = await readJson(response);
    if (!response.ok) {
      throw failureFrom(response, data);
    }
    if (!data || typeof data.token !== "string" || data.token === "") {
      throw new ApiFailure("protocol", response.status, "protocol", "The server sent an unexpected session reply.");
    }
    session.token = data.token;
    session.limits = data.limits && typeof data.limits === "object" ? data.limits : null;
    applyLimits();
    markConnected();
  }

  function ensureSession(force) {
    if (force) {
      session.token = null;
    }
    if (session.token) {
      return Promise.resolve();
    }
    if (!session.pending) {
      session.pending = bootstrapSession().finally(function () {
        session.pending = null;
      });
    }
    return session.pending;
  }

  async function api(method, path, body, options) {
    var opts = options || {};
    try {
      await ensureSession(false);
    } catch (err) {
      if (err.kind === "network") {
        markDisconnected();
      }
      throw err;
    }
    var headers = { "X-Seekr-Token": session.token, Accept: "application/json" };
    var init = {
      method: method,
      headers: headers,
      cache: "no-store",
      credentials: "same-origin",
      redirect: "error",
      signal: opts.signal,
    };
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    var response;
    try {
      response = await fetch(path, init);
    } catch (err) {
      if (isAbort(err)) {
        throw new ApiFailure("aborted");
      }
      markDisconnected();
      throw new ApiFailure("network", 0, "network", "");
    }
    markConnected();
    var data = await readJson(response);
    if (response.ok) {
      return { status: response.status, data: data };
    }
    var failure = failureFrom(response, data);
    if (failure.status === 403 && failure.code === "forbidden" && !opts.retried) {
      // The token is only valid for one server process. After a restart,
      // bootstrap a fresh one and retry once; a 403 means nothing ran.
      try {
        await ensureSession(true);
      } catch (err) {
        if (err.kind === "network") {
          markDisconnected();
        }
        throw err;
      }
      return api(method, path, body, { signal: opts.signal, retried: true });
    }
    throw failure;
  }

  // ---------------------------------------------------------------- connection

  function markConnected() {
    state.everConnected = true;
    if (state.connected) {
      return;
    }
    state.connected = true;
    show(byId("connection-banner"), false);
    renderAll();
  }

  function markDisconnected(customText) {
    var wasConnected = state.connected;
    state.connected = false;
    if (state.jobTimer) {
      window.clearTimeout(state.jobTimer);
      state.jobTimer = null;
    }
    var text = byId("connection-text");
    clear(text);
    if (customText) {
      append(text, customText);
    } else {
      append(text, "Seekr isn’t responding. Check that ", codeText("seekr web"), " is still running in your terminal, then choose Reconnect.");
    }
    show(byId("connection-banner"), true);
    if (wasConnected) {
      announce("Lost contact with Seekr.");
    }
    renderAll();
  }

  async function connect(force) {
    try {
      await ensureSession(Boolean(force));
    } catch (err) {
      if (err.kind === "http" && err.status === 403) {
        markDisconnected("Seekr refused this page. Open it from the address printed by seekr web in your terminal.");
      } else if (err.kind === "protocol") {
        markDisconnected(err.message);
      } else {
        markDisconnected();
      }
      return;
    }
    // Connection state is decided by these real requests, not assumed.
    await Promise.all([refreshStatus(), refreshRoots()]);
    if (state.job && !state.jobTimer) {
      pollJob();
    }
  }

  // ---------------------------------------------------------------- status

  function engineBusy() {
    if (state.job) {
      return true;
    }
    var lane = state.status && state.status.activity ? state.status.activity.lane : "idle";
    return lane === "index" || lane === "model_check";
  }

  function indexRunning() {
    if (state.job && state.job.operation === "index") {
      return true;
    }
    var activity = state.status && state.status.activity;
    return Boolean(activity) && activity.lane === "index";
  }

  function configInvalid() {
    return Boolean(state.status && state.status.config && state.status.config.state === "invalid");
  }

  function configPath() {
    if (state.roots && typeof state.roots.config_path === "string") {
      return state.roots.config_path;
    }
    if (state.status && state.status.config && typeof state.status.config.path === "string") {
      return state.status.config.path;
    }
    return "";
  }

  function rootCount() {
    if (state.roots && Array.isArray(state.roots.roots)) {
      return state.roots.roots.length;
    }
    var cfg = state.status && state.status.config;
    return cfg && isCount(cfg.roots) ? cfg.roots : null;
  }

  async function refreshStatus() {
    var seq = ++state.statusSeq;
    try {
      var result = await api("GET", "/api/status");
      if (seq !== state.statusSeq) {
        return;
      }
      var data = result.data;
      if (!data || typeof data !== "object" || !data.index || !data.model || !data.daemon || !data.config) {
        return;
      }
      state.status = data;
      state.statusError = null;
      var activity = data.activity || {};
      if (!state.job && typeof activity.job_id === "string" && activity.job_id &&
          !state.finishedJobs[activity.job_id] &&
          (activity.lane === "index" || activity.lane === "model_check")) {
        // Recover a job started before this page loaded (or in another tab).
        startPolling(activity.job_id, activity.lane);
      }
    } catch (err) {
      if ((err.kind === "http" || err.kind === "protocol") && seq === state.statusSeq) {
        state.status = null;
        state.statusError = err;
      }
    }
    renderAll();
  }

  async function refreshRoots() {
    var seq = ++state.rootsSeq;
    try {
      var result = await api("GET", "/api/roots");
      if (seq !== state.rootsSeq) {
        return;
      }
      var data = result.data;
      if (data && Array.isArray(data.roots)) {
        state.roots = data;
        state.rootsError = null;
      }
    } catch (err) {
      if (err.kind === "http" && seq === state.rootsSeq) {
        state.roots = null;
        state.rootsError = err;
      }
    }
    renderFolders();
    renderControls();
    renderResultsState();
  }

  function setStatusValue(node, text, tone) {
    node.textContent = text;
    node.className = "status-value" + (tone ? " is-" + tone : "");
  }

  function renderStatus() {
    var st = state.status;
    var indexState = byId("index-state");
    var indexFacts = byId("index-facts");
    var indexDetail = byId("index-detail");
    var modelState = byId("model-state");
    var modelName = byId("model-name");
    var modelDetail = byId("model-detail");
    var daemonState = byId("daemon-state");
    var daemonDetail = byId("daemon-detail");
    var configBlock = byId("config-block");

    clear(configBlock);
    show(configBlock, false);

    if (!st) {
      var waiting = state.connected || !state.everConnected ? "Loading…" : "Unknown";
      setStatusValue(indexState, waiting, null);
      setStatusValue(modelState, waiting, null);
      setStatusValue(daemonState, waiting, null);
      show(indexFacts, false);
      [indexDetail, modelName, modelDetail, daemonDetail].forEach(function (n) {
        setText(n, "");
      });
      return;
    }

    if (st.config.state === "invalid") {
      append(configBlock,
        el("h3", "status-label", "Settings file"),
        notice("error", "Can’t be read", [
          str(st.config.message),
          ["Fix this file by hand, then choose Refresh: ", el("span", "path-text", str(st.config.path))],
        ]));
      show(configBlock, true);
    }

    // Index
    var idx = st.index;
    var labels = {
      missing: ["Not created yet", "off"],
      empty: ["Empty", "off"],
      ready: ["Ready", "ok"],
      incompatible: ["Needs rebuilding", "bad"],
      error: ["Can’t be read", "bad"],
    };
    var label = labels[idx.state] || ["Unknown", null];
    setStatusValue(indexState, label[0], label[1]);

    clear(indexFacts);
    var facts = [
      ["Documents", formatCount(idx.documents)],
      ["Passages", formatCount(idx.chunks)],
      ["Size on disk", formatBytes(idx.size_bytes)],
    ];
    if (isCount(idx.last_error_count) && idx.last_error_count > 0) {
      facts.push(["Unreadable files, last run", formatCount(idx.last_error_count)]);
    }
    var anyFact = false;
    if (idx.state === "ready" || idx.state === "empty") {
      facts.forEach(function (f) {
        if (f[1]) {
          append(indexFacts, el("dt", null, f[0]), el("dd", null, f[1]));
          anyFact = true;
        }
      });
    }
    show(indexFacts, anyFact);

    var indexText = "";
    if (idx.state === "missing") {
      indexText = "No index file exists yet. Update index creates it from your folders.";
    } else if (idx.state === "empty") {
      indexText = "The index has no documents yet.";
    } else if (idx.state === "incompatible") {
      indexText = "It was built with a different model or format. Seekr leaves it untouched; rebuild it from a terminal (seekr reindex).";
    } else if (idx.state === "error") {
      indexText = "Run seekr doctor in a terminal to diagnose it.";
    }
    var idxMessage = str(idx.message);
    setText(indexDetail, idxMessage ? (indexText ? indexText + " " : "") + "Details: " + idxMessage : indexText);

    // Model
    var model = st.model;
    var checking = model.state === "loading" || Boolean(state.job && state.job.operation === "model_check");
    var modelLabels = {
      unchecked: ["Not checked", "off"],
      loading: ["Checking…", "warn"],
      ready: ["Ready", "ok"],
      unavailable: ["Unavailable", "bad"],
    };
    var ml = checking ? modelLabels.loading : (modelLabels[model.state] || ["Unknown", null]);
    setStatusValue(modelState, ml[0], ml[1]);
    setText(modelName, str(model.model));
    var modelText = "";
    if (checking) {
      modelText = "Loading the model from this computer to confirm it works.";
    } else if (model.state === "unchecked") {
      modelText = "Exact-words search doesn’t need it. Other search modes load it the first time you use them, or you can check it now.";
    } else if (model.state === "ready") {
      modelText = "Loaded and working.";
    } else if (model.state === "unavailable") {
      var mm = str(model.message);
      modelText = (mm ? "Details: " + mm + " " : "") +
        "Seekr never downloads models from this page. Make the model available on this computer from a terminal, then choose Check model. Exact-words search still works.";
    }
    setText(modelDetail, modelText);

    // Daemon
    if (st.daemon.present) {
      setStatusValue(daemonState, "Running", "ok");
      clear(daemonDetail);
      append(daemonDetail, "A ", codeText("seekr daemon"), " process is running. Searches go through it, and it looks after the index itself.");
      show(daemonDetail, true);
    } else {
      setStatusValue(daemonState, "Not running", "off");
      setText(daemonDetail, "");
    }
  }

  // ---------------------------------------------------------------- controls

  function renderControls() {
    var update = byId("update-index");
    var updateNote = byId("update-note");
    var check = byId("check-model");
    var st = state.status;

    clear(updateNote);
    var reason = null;
    if (!state.connected) {
      reason = ["Reconnect to Seekr first."];
    } else if (!st) {
      reason = ["Waiting for status."];
    } else if (st.daemon.present) {
      reason = ["Turned off while the background service is running, because it maintains the index. To update from this page, stop the ", codeText("seekr daemon"), " process (Ctrl+C in its terminal), then choose Refresh."];
    } else if (configInvalid()) {
      reason = ["Fix the settings file first."];
    } else if (state.job && state.job.operation === "index") {
      reason = null;
    } else if (engineBusy()) {
      reason = ["Available when the current task finishes."];
    } else if (rootCount() === 0) {
      reason = ["Add a folder first."];
    } else if (st.index.state === "incompatible") {
      reason = ["The index needs rebuilding from a terminal first."];
    } else if (st.index.state === "error") {
      reason = ["The index can’t be read. Run seekr doctor first."];
    } else if (st.model.state === "unavailable") {
      reason = ["Updating needs the local model. Check model once it’s available."];
    }
    var updateDisabled = reason !== null || indexRunning();
    update.disabled = updateDisabled;
    if (reason) {
      reason.forEach(function (part) {
        append(updateNote, part);
      });
      show(updateNote, true);
    } else if (!indexRunning()) {
      append(updateNote, "Reads new and changed files in your folders. Large folders can take a while.");
      show(updateNote, true);
    } else {
      show(updateNote, false);
    }

    var modelChecking = Boolean(st && st.model.state === "loading");
    check.disabled = !state.connected || !st || configInvalid() || engineBusy() || modelChecking;

    byId("add-folder-submit").disabled = state.addPending || !state.connected;
    applyModeAvailability();
  }

  function selectedMode() {
    var checked = document.querySelector('input[name="mode"]:checked');
    return checked ? checked.value : "hybrid";
  }

  function setMode(value) {
    var radios = document.querySelectorAll('input[name="mode"]');
    radios.forEach(function (r) {
      r.checked = r.value === value;
    });
  }

  function applyModeAvailability() {
    var running = indexRunning();
    var radios = document.querySelectorAll('input[name="mode"]');
    radios.forEach(function (r) {
      if (r.value === "keyword") {
        return;
      }
      r.disabled = running;
      r.parentElement.classList.toggle("is-disabled", running);
    });
    var note = byId("mode-note");
    if (running) {
      var current = selectedMode();
      if (current !== "keyword") {
        state.forcedFromMode = current;
        setMode("keyword");
      }
      setText(note, "Only exact-words search is available while the index updates. The other options return when it finishes.");
    } else {
      if (state.forcedFromMode && selectedMode() === "keyword") {
        setMode(state.forcedFromMode);
      }
      state.forcedFromMode = null;
      setText(note, "");
    }
  }

  // ---------------------------------------------------------------- jobs

  function startPolling(jobId, operation) {
    var current = state.job;
    if (current && current.id === jobId && current.operation === operation) {
      // A status refresh and a start reply can both discover the same job.
      // Keep the object the active poll belongs to so its identity check still
      // matches, and only start a poll when nothing is pending.
      if (!state.pollInFlight && !state.jobTimer && state.connected) {
        pollJob();
      }
      return;
    }
    if (state.jobTimer) {
      window.clearTimeout(state.jobTimer);
      state.jobTimer = null;
    }
    state.job = { id: jobId, operation: operation, data: null, lost: false };
    state.lastJob = null;
    renderAll();
    if (state.pollInFlight) {
      // The poll for the replaced job will notice the change when it returns
      // and hand control to this job.
      return;
    }
    pollJob();
  }

  function continueAfterReplacedJob() {
    // A poll that belonged to a replaced job has finished. Release the shared
    // flag, then make sure the job now being tracked gets its own poll.
    state.pollInFlight = false;
    if (state.job && state.connected && !state.jobTimer) {
      pollJob();
    } else {
      renderJob();
    }
  }

  async function pollJob() {
    state.jobTimer = null;
    var job = state.job;
    if (!job || !state.connected || state.pollInFlight) {
      renderJob();
      return;
    }
    state.pollInFlight = true;
    try {
      var result = await api("GET", "/api/jobs/" + encodeURIComponent(job.id));
      if (state.job !== job) {
        continueAfterReplacedJob();
        return;
      }
      var data = result.data;
      if (!data || typeof data.state !== "string") {
        throw new ApiFailure("protocol", 200, "protocol", "The server sent an unexpected task reply.");
      }
      job.data = data;
      if (data.operation === "index" || data.operation === "model_check") {
        job.operation = data.operation;
      }
      if (data.state === "running") {
        state.pollInFlight = false;
        renderAll();
        if (state.connected && state.job === job) {
          state.jobTimer = window.setTimeout(pollJob, POLL_MS);
        }
        return;
      }
      state.pollInFlight = false;
      finishJob(job, data);
    } catch (err) {
      if (state.job !== job) {
        continueAfterReplacedJob();
        return;
      }
      state.pollInFlight = false;
      if (err.kind === "network") {
        renderJob();
        return; // Polling resumes after Reconnect.
      }
      state.job = null;
      state.finishedJobs[job.id] = true;
      if (err.code === "job_not_found") {
        state.lastJob = { operation: job.operation, lost: true };
      } else {
        state.lastJob = { operation: job.operation, failure: err };
      }
      renderAll();
      refreshStatus();
    }
  }

  function finishJob(job, data) {
    state.finishedJobs[job.id] = true;
    state.job = null;
    state.lastJob = { operation: job.operation, data: data };
    announce(jobSummary(job.operation, data));
    renderAll();
    refreshStatus();
    if (job.operation === "index") {
      refreshRoots();
    }
  }

  function jobSummary(operation, data) {
    var took = formatElapsed(data.elapsed_ms);
    if (operation === "model_check") {
      if (data.state === "succeeded") {
        return "Model check finished: the model is ready.";
      }
      return "Model check finished: the model is unavailable.";
    }
    var report = data.report || {};
    if (data.state === "failed") {
      return "Index update failed" + (data.error ? ": " + data.error : ".");
    }
    var n = function (value) {
      return isCount(value) ? value : 0;
    };
    var counts = plural(n(report.added), "file added", "files added") + ", " +
      formatCount(n(report.updated)) + " updated, " + formatCount(n(report.removed)) + " removed";
    if (data.state === "partial") {
      return "Index updated with " + plural(n(report.error_total), "unreadable file", "unreadable files") + ". " + counts + ".";
    }
    return "Index updated" + (took ? " in " + took : "") + ". " + counts + ".";
  }

  function diagList(title, items, total, truncated) {
    var wrap = el("div", "diag");
    var count = isCount(total) ? total : items.length;
    append(wrap, el("p", "diag-title", title + " (" + formatCount(count) + ")"));
    if (truncated || items.length < count) {
      append(wrap, el("p", "status-detail", "Showing the first " + formatCount(items.length) + " of " + formatCount(count) + "."));
    }
    var list = el("ul", "diag-list");
    items.forEach(function (item) {
      if (!item || typeof item !== "object") {
        return;
      }
      var li = el("li");
      append(li, el("span", "diag-path", str(item.path)), el("span", "diag-message", str(item.message)));
      append(list, li);
    });
    append(wrap, list);
    return wrap;
  }

  function renderJob() {
    var box = byId("job");
    var body = byId("job-body");
    var heading = byId("job-heading");
    clear(body);

    var job = state.job;
    var last = state.lastJob;
    if (!job && !last) {
      show(box, false);
      return;
    }
    show(box, true);

    if (job) {
      heading.textContent = "Current task";
      var isIndex = job.operation === "index";
      append(body, el("p", "job-title", isIndex ? "Updating index…" : "Checking model…"));
      if (!state.connected) {
        append(body, el("p", "job-line", "Lost contact with Seekr while this was running. Reconnect to see how it finished."));
        return;
      }
      var elapsed = job.data ? formatElapsed(job.data.elapsed_ms) : "";
      append(body, el("p", "job-line", elapsed ? "Running for " + elapsed + "." : "Started."));
      var bar = el("span", "working");
      bar.setAttribute("aria-hidden", "true");
      append(body, bar);
      if (isIndex) {
        append(body, el("p", "status-detail", "Exact-words search keeps working meanwhile. If you stop seekr web now, it waits for this update to finish before exiting."));
      }
      return;
    }

    heading.textContent = "Last task";
    var op = last.operation;
    if (last.startError) {
      append(body, jobStartError(last.startError, op === "index"));
      return;
    }
    if (last.lost) {
      append(body,
        el("p", "job-title", op === "model_check" ? "Model check" : "Index update"),
        el("p", "job-line", "This task’s record is no longer available, possibly because Seekr restarted. The status above shows the current state."));
      return;
    }
    if (last.failure) {
      append(body,
        el("p", "job-title is-bad", "Couldn’t follow this task"),
        el("p", "job-line", last.failure.message || "The server returned an error while reporting progress."));
      return;
    }
    var data = last.data;
    var took = formatElapsed(data.elapsed_ms);
    if (op === "model_check") {
      if (data.state === "succeeded") {
        var rep = data.report || {};
        append(body, el("p", "job-title", "Model ready"));
        var line = str(rep.model);
        if (isCount(rep.dim)) {
          line += (line ? ", " : "") + formatCount(rep.dim) + " dimensions";
        }
        append(body, el("p", "job-line", line), took ? el("p", "status-detail", "Checked in " + took + ".") : null);
      } else {
        append(body,
          el("p", "job-title is-bad", "Model unavailable"),
          el("p", "job-line", str(data.error) || "The model could not be loaded."),
          el("p", "status-detail", "Seekr never downloads models from this page. Make the model available on this computer, then check again."));
      }
      return;
    }

    if (data.state === "failed") {
      append(body,
        el("p", "job-title is-bad", "Index update failed"),
        el("p", "job-line", str(data.error) || "No report was produced."),
        took ? el("p", "status-detail", "Stopped after " + took + ".") : null);
      return;
    }

    var report = data.report && typeof data.report === "object" ? data.report : {};
    var partial = data.state === "partial";
    append(body, el("p", partial ? "job-title is-warn" : "job-title", partial ? "Index updated, some files couldn’t be read" : "Index updated"));
    if (took) {
      append(body, el("p", "job-line", "Took " + took + "."));
    }
    var counts = el("dl", "counts");
    [["Added", report.added], ["Updated", report.updated], ["Unchanged", report.unchanged], ["Removed", report.removed], ["Skipped", report.skipped]].forEach(function (row) {
      if (isCount(row[1])) {
        append(counts, el("dt", null, row[0]), el("dd", null, formatCount(row[1])));
      }
    });
    append(body, counts);
    var errors = Array.isArray(report.errors) ? report.errors : [];
    var warnings = Array.isArray(report.warnings) ? report.warnings : [];
    if (errors.length || (isCount(report.error_total) && report.error_total > 0)) {
      append(body, diagList("Couldn’t read", errors, report.error_total, report.errors_truncated === true));
    }
    if (warnings.length || (isCount(report.warning_total) && report.warning_total > 0)) {
      append(body, diagList("Warnings", warnings, report.warning_total, report.warnings_truncated === true));
    }
  }

  function jobStartError(err, isIndex) {
    var bodies = [err.message];
    var actions = [];
    var title = isIndex ? "Couldn’t start the index update" : "Couldn’t start the model check";
    if (err.code === "no_roots") {
      bodies = ["There are no folders to read yet."];
      actions.push(button("Add folder", "button-secondary", function () {
        setView("folders", true);
      }));
    } else if (err.code === "daemon_active") {
      bodies = [err.message, ["The background service maintains the index. Stop the ", codeText("seekr daemon"), " process to update from this page."]];
    } else if (err.code === "busy") {
      bodies = ["Another task is already running. Try again when it finishes."];
    } else if (err.code === "config_invalid") {
      bodies = [err.message, ["Settings file: ", el("span", "path-text", configPath())]];
    }
    return notice("error", title, bodies, actions);
  }

  async function startJob(kind) {
    var isIndex = kind === "index";
    var path = isIndex ? "/api/index" : "/api/model/check";
    var trigger = byId(isIndex ? "update-index" : "check-model");
    trigger.disabled = true;
    try {
      var result = await api("POST", path, {});
      var data = result.data;
      if (!data || typeof data.job_id !== "string" || !data.job_id) {
        throw new ApiFailure("protocol", result.status, "protocol", "The server sent an unexpected reply.");
      }
      announce(isIndex ? "Index update started." : "Model check started.");
      startPolling(data.job_id, isIndex ? "index" : "model_check");
    } catch (err) {
      if (err.kind === "network") {
        return;
      }
      state.lastJob = { operation: isIndex ? "index" : "model_check", startError: err };
      renderJob();
      announce((isIndex ? "Index update not started. " : "Model check not started. ") + (err.message || ""));
      refreshStatus();
    } finally {
      renderControls();
    }
  }

  // ---------------------------------------------------------------- search

  function queryTerms(query) {
    var matches = query.toLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];
    var seen = Object.create(null);
    var terms = [];
    matches.forEach(function (term) {
      if (term.length < 2 || seen[term]) {
        return;
      }
      seen[term] = true;
      terms.push(term);
    });
    terms.sort(function (a, b) {
      return b.length - a.length;
    });
    return terms.slice(0, 24);
  }

  function escapeRegExp(text) {
    return text.replace(/[\\^$.*+?()[\]{}|/]/g, "\\$&");
  }

  function appendHighlighted(parent, text, terms) {
    if (!terms.length || !text) {
      append(parent, text);
      return;
    }
    var pattern = new RegExp("(" + terms.map(escapeRegExp).join("|") + ")", "giu");
    text.split(pattern).forEach(function (part, i) {
      if (!part) {
        return;
      }
      if (i % 2 === 1) {
        append(parent, el("mark", null, part));
      } else {
        append(parent, part);
      }
    });
  }

  function intField(id, min, max, fallback, label) {
    var input = byId(id);
    var raw = input.value.trim();
    if (raw === "") {
      input.removeAttribute("aria-invalid");
      return { value: fallback };
    }
    if (!/^\d+$/.test(raw)) {
      input.setAttribute("aria-invalid", "true");
      return { error: label + " must be a whole number from " + min + " to " + max + "." , input: input };
    }
    var n = parseInt(raw, 10);
    if (n < min || n > max) {
      input.setAttribute("aria-invalid", "true");
      return { error: label + " must be from " + min + " to " + max + ".", input: input };
    }
    input.removeAttribute("aria-invalid");
    return { value: n };
  }

  function buildSearchRequest() {
    var query = byId("query").value;
    var body = { query: query, mode: selectedMode() };
    var limits = session.limits || {};
    var k = intField("filter-k", numberOr(limits.k_min, 1), numberOr(limits.k_max, 50), DEFAULT_K, "Most results");
    if (k.error) {
      return { error: k.error, input: k.input };
    }
    var perFile = intField("filter-per-file", numberOr(limits.per_file_min, 1), numberOr(limits.per_file_max, 10), DEFAULT_PER_FILE, "Passages per file");
    if (perFile.error) {
      return { error: perFile.error, input: perFile.input };
    }
    body.k = k.value;
    body.per_file = perFile.value;
    var path = byId("filter-path").value.trim();
    if (path) {
      body.path = path;
    }
    var kind = byId("filter-kind").value;
    if (kind) {
      body.kind = kind;
    }
    var since = byId("filter-since").value;
    if (since === "date") {
      var dateInput = byId("filter-since-date");
      if (!dateInput.value) {
        dateInput.setAttribute("aria-invalid", "true");
        return { error: "Pick a date for “Since date”, or choose another Changed option.", input: dateInput };
      }
      dateInput.removeAttribute("aria-invalid");
      body.since = dateInput.value;
    } else if (since) {
      body.since = since;
    }
    return { body: body };
  }

  function numberOr(value, fallback) {
    return typeof value === "number" && isFinite(value) ? value : fallback;
  }

  function openFilters() {
    byId("filters").open = true;
  }

  function setSearchBusy(busy) {
    var submit = byId("search-submit");
    state.searchInFlight = busy;
    byId("results").setAttribute("aria-busy", busy ? "true" : "false");
    submit.textContent = busy ? "Searching…" : "Search";
    if (busy) {
      // aria-disabled (not disabled) keeps keyboard focus on the button.
      submit.setAttribute("aria-disabled", "true");
    } else {
      submit.removeAttribute("aria-disabled");
    }
  }

  async function runSearch() {
    // One search at a time. Aborting a fetch doesn't stop the server, which
    // would keep the engine busy and make the next search fail with 409, so
    // extra submissions become one follow-up sent when this search finishes.
    if (state.searchInFlight) {
      if (!state.searchQueued) {
        state.searchQueued = true;
        setText(byId("results-meta"), "Searching… Your latest search runs when this one finishes.");
        announce("Still searching. Your latest search runs when this one finishes.");
      }
      return;
    }
    var built = buildSearchRequest();
    var seq = ++state.searchSeq;
    var results = byId("results");

    if (built.error) {
      openFilters();
      showSearchError(notice("error", "Check the filters", [built.error]));
      built.input.focus();
      announce(built.error);
      return;
    }
    var body = built.body;
    if (!body.query.trim()) {
      state.searchShown = false;
      state.searchError = null;
      results.setAttribute("aria-busy", "false");
      clearHits();
      renderResultsState();
      var hint = "Type what you’re looking for, then press Enter.";
      byId("results-state").insertBefore(notice("quiet", null, [hint]), byId("results-state").firstChild);
      byId("query").focus();
      announce(hint);
      return;
    }

    setSearchBusy(true);
    setText(byId("results-meta"), "Searching…");
    try {
      var result = await api("POST", "/api/search", body);
      // Stale guard: skip rendering if a newer search replaced this one.
      if (seq !== state.searchSeq || state.searchQueued) {
        return;
      }
      var data = result.data;
      if (!data || !Array.isArray(data.hits)) {
        throw new ApiFailure("protocol", result.status, "protocol", "The server sent an unexpected search reply.");
      }
      renderHits(body, data);
    } catch (err) {
      if (seq !== state.searchSeq || state.searchQueued || isAbort(err)) {
        return;
      }
      renderSearchFailure(err, body);
    } finally {
      // Always release: only one search is ever in flight.
      setSearchBusy(false);
      if (state.searchQueued) {
        state.searchQueued = false;
        runSearch();
      }
    }
  }

  function clearHits() {
    clear(byId("hit-list"));
    show(byId("hit-list"), false);
    setText(byId("results-meta"), "");
  }

  function showSearchError(node) {
    state.searchShown = false;
    state.searchError = node;
    clearHits();
    var holder = byId("results-state");
    clear(holder);
    append(holder, node);
  }

  function keywordRetry() {
    return button("Search by exact words", "button-secondary", function () {
      setMode("keyword");
      state.forcedFromMode = null;
      runSearch();
    });
  }

  function renderSearchFailure(err, body) {
    var title = "Search didn’t finish";
    var bodies = [err.message];
    var actions = [];
    var code = err.code;
    if (err.kind === "network") {
      title = "Seekr isn’t responding";
      bodies = ["Your search and filters are kept. Reconnect, then search again."];
    } else if (code === "invalid_request") {
      title = "Check the search";
      if (err.fields) {
        Object.keys(err.fields).forEach(function (key) {
          var value = err.fields[key];
          if (typeof value === "string" && value) {
            bodies.push(value);
          }
        });
      }
      openFilters();
    } else if (code === "busy") {
      title = "Seekr is busy";
      bodies = ["Another task is using the search engine. Try again when it finishes. Exact-words search works meanwhile."];
      if (body.mode !== "keyword") {
        actions.push(keywordRetry());
      }
    } else if (code === "model_unavailable") {
      title = "The local model is unavailable";
      bodies = [err.message, "Exact-words search doesn’t need the model."];
      if (body.mode !== "keyword") {
        actions.push(keywordRetry());
      }
    } else if (code === "incompatible_index") {
      title = "The index needs rebuilding";
      bodies = [err.message, "Seekr leaves it untouched. Rebuild it from a terminal (seekr reindex)."];
    } else if (code === "index_unavailable") {
      title = "The index can’t be read";
      bodies = [err.message, "Run seekr doctor in a terminal to diagnose it."];
    } else if (code === "config_invalid") {
      title = "The settings file can’t be read";
      bodies = [err.message, ["Fix this file by hand: ", el("span", "path-text", configPath())]];
    } else if (code === "daemon_failure") {
      title = "The background service didn’t answer";
      bodies = [err.message, ["Restart ", codeText("seekr daemon"), ", or stop it so this page searches the index directly."]];
    } else if (!err.message) {
      bodies = ["The server reported an unexpected problem (" + (err.status || "no status") + "). Try again."];
    }
    showSearchError(notice("error", title, bodies, actions));
    announce(title + ".");
    if (err.kind === "http" && code !== "invalid_request") {
      refreshStatus();
    }
  }

  function renderHits(body, data) {
    state.searchShown = true;
    state.searchError = null;
    var holder = byId("results-state");
    clear(holder);
    var list = byId("hit-list");
    clear(list);
    var terms = queryTerms(body.query);
    var shown = 0;

    data.hits.forEach(function (hit, i) {
      if (!hit || typeof hit !== "object" || typeof hit.path !== "string") {
        return;
      }
      shown += 1;
      append(list, renderHit(hit, i, terms));
    });

    var meta = byId("results-meta");
    var took = formatTook(data.took_ms);
    if (shown === 0) {
      show(list, false);
      setText(meta, "");
      append(holder, notice("quiet", "No passages match", [
        ["Nothing in the index matched ", el("q", null, body.query.trim()), "."],
        hasFilters() ? "Try clearing the filters, using fewer words, or a different match option." : "Try fewer or different words, or a different match option.",
      ], hasFilters() ? [button("Clear filters and search again", "button-quiet", function () {
        clearFilters();
        runSearch();
      })] : []));
      announce("No passages match.");
      return;
    }
    show(list, true);
    var summary = plural(shown, "passage", "passages") + (took ? ", found in " + took : "") + ".";
    setText(meta, summary);
    announce(summary);
  }

  function renderHit(hit, index, terms) {
    var path = hit.path;
    var locator = str(hit.locator);
    var li = el("li", "hit");
    var nameId = "hit-name-" + index;

    var head = el("div", "hit-head");
    var name = el("h3", "hit-name", baseName(path));
    name.id = nameId;
    append(head, name);
    var kind = str(hit.kind);
    if (kind) {
      var kindLabel = Object.prototype.hasOwnProperty.call(KIND_LABELS, kind) ? KIND_LABELS[kind] : kind;
      append(head, el("span", "hit-kind", kindLabel));
    }
    append(li, head);

    append(li, el("p", "hit-path", path));

    if (locator) {
      var loc = el("p", "hit-locator");
      append(loc, el("span", "visually-hidden", "Location: "), locator);
      append(li, loc);
    }

    var snippet = el("p", "hit-snippet");
    appendHighlighted(snippet, str(hit.snippet), terms);
    append(li, snippet);

    var actions = el("div", "hit-actions");
    var copyButton = el("button", "button button-quiet", "Copy location");
    copyButton.type = "button";
    copyButton.setAttribute("aria-describedby", nameId);
    var status = el("span", "copy-status");
    status.setAttribute("aria-live", "polite");
    var locationText = locator ? path + " (" + locator + ")" : path;
    copyButton.addEventListener("click", function () {
      copyLocation(locationText, actions, status, index);
    });
    append(actions, copyButton, status);
    append(li, actions);
    return li;
  }

  async function copyLocation(text, actions, status, index) {
    var existing = actions.querySelector(".copy-fallback");
    if (existing) {
      actions.removeChild(existing);
    }
    status.textContent = "";
    var copied = false;
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        copied = true;
      } catch (err) {
        copied = false;
      }
    }
    if (copied) {
      status.textContent = "Copied";
      return;
    }
    // Fallback: show the location in a selected field the user can copy.
    var wrap = el("div", "copy-fallback");
    var fieldId = "copy-field-" + index;
    var label = el("label", null, "Couldn’t copy automatically. The location is selected: press Ctrl+C (or ⌘C) to copy it.");
    label.htmlFor = fieldId;
    var field = el("input");
    field.type = "text";
    field.id = fieldId;
    field.readOnly = true;
    field.value = text;
    append(wrap, label, field);
    append(actions, wrap);
    field.focus();
    field.select();
    status.textContent = "Select and copy below";
  }

  function hasFilters() {
    return activeFilterCount() > 0;
  }

  function activeFilterCount() {
    var count = 0;
    if (byId("filter-path").value.trim()) {
      count += 1;
    }
    if (byId("filter-kind").value) {
      count += 1;
    }
    if (byId("filter-since").value) {
      count += 1;
    }
    var k = byId("filter-k").value.trim();
    if (k && k !== String(DEFAULT_K)) {
      count += 1;
    }
    var pf = byId("filter-per-file").value.trim();
    if (pf && pf !== String(DEFAULT_PER_FILE)) {
      count += 1;
    }
    return count;
  }

  function renderFilterCount() {
    var n = activeFilterCount();
    byId("filters-count").textContent = n ? "(" + n + " active)" : "";
    show(byId("since-date-field"), byId("filter-since").value === "date");
  }

  function clearFilters() {
    byId("filter-path").value = "";
    byId("filter-kind").value = "";
    byId("filter-since").value = "";
    byId("filter-since-date").value = "";
    byId("filter-k").value = String(DEFAULT_K);
    byId("filter-per-file").value = String(DEFAULT_PER_FILE);
    ["filter-k", "filter-per-file", "filter-since-date"].forEach(function (id) {
      byId(id).removeAttribute("aria-invalid");
    });
    renderFilterCount();
  }

  // Index/config states shown in the results area when no search is on screen.
  function renderResultsState() {
    if (state.searchShown || state.searchError) {
      return;
    }
    var holder = byId("results-state");
    clear(holder);
    var st = state.status;
    if (!st) {
      if (state.connected && state.statusError) {
        append(holder, notice("error", "Couldn’t read Seekr’s status", [
          state.statusError.message || "The server reported an unexpected problem (" + (state.statusError.status || "no status") + ").",
        ], [button("Refresh", "button-quiet", refreshAll)]));
      } else if (state.connected || !state.everConnected) {
        append(holder, notice("quiet", null, ["Connecting to Seekr…"]));
      } else {
        append(holder, notice("error", "Seekr isn’t responding", ["Your search and filters are kept. Reconnect to continue."]));
      }
      return;
    }
    var updateAction = function () {
      return button("Update index", "button-secondary", function () {
        startJob("index");
      });
    };
    var canUpdate = !byId("update-index").disabled;
    var roots = rootCount();
    var idx = st.index;

    if (configInvalid()) {
      append(holder, notice("error", "The settings file can’t be read", [
        str(st.config.message),
        ["Seekr won’t change it automatically. Fix it by hand, then choose Refresh: ", el("span", "path-text", str(st.config.path))],
      ], [button("Refresh", "button-quiet", refreshAll)]));
    } else if (roots === 0 && (idx.state === "missing" || idx.state === "empty")) {
      append(holder, notice("info", "No folders yet", [
        "Seekr searches files in the folders you choose. Add a folder, then update the index to read it.",
      ], [button("Add folder", "button-secondary", function () {
        setView("folders", true);
      })]));
    } else if (idx.state === "missing" || idx.state === "empty") {
      var bodies = [idx.state === "missing" ? "Your folders are saved, but nothing has been indexed yet." : "The index exists but holds no documents yet."];
      if (st.daemon.present) {
        bodies.push(["A ", codeText("seekr daemon"), " process is running and maintains the index."]);
      } else {
        bodies.push("Update index reads the files in your folders so you can search them.");
      }
      append(holder, notice("info", idx.state === "missing" ? "Nothing indexed yet" : "The index is empty", bodies, canUpdate ? [updateAction()] : []));
    } else if (idx.state === "incompatible") {
      append(holder, notice("error", "The index needs rebuilding", [
        str(idx.message),
        "It was built with a different model or format. Seekr leaves it untouched; rebuild it from a terminal (seekr reindex).",
      ]));
    } else if (idx.state === "error") {
      append(holder, notice("error", "The index can’t be read", [
        str(idx.message),
        "Run seekr doctor in a terminal to diagnose it.",
      ]));
    } else if (idx.state === "ready") {
      var docs = isCount(idx.documents) ? plural(idx.documents, "document", "documents") : "your documents";
      append(holder, notice("quiet", null, ["Search " + docs + " in your folders by words, meaning, or both."]));
    }

    if (st.model.state === "unavailable" && !configInvalid() && idx.state === "ready") {
      append(holder, notice("caution", "Only exact-words search works right now", [
        "The local model is unavailable, so the other match options will fail. See Local model in Index status.",
      ]));
    }
    if (indexRunning() && idx.state === "ready") {
      append(holder, notice("caution", null, ["The index is updating. Exact-words search works meanwhile; results may change when it finishes."]));
    }
  }

  // ---------------------------------------------------------------- folders

  function renderFolders() {
    var list = byId("root-list");
    var empty = byId("roots-empty");
    var cfgLine = byId("config-path");
    var holder = byId("folders-state");
    clear(list);
    clear(holder);

    if (state.rootsError) {
      var err = state.rootsError;
      if (err.code === "config_invalid") {
        append(holder, notice("error", "Seekr can’t safely edit your settings file", [
          err.message,
          ["Edit it by hand to add folders: ", el("span", "path-text", configPath())],
        ], [button("Refresh", "button-quiet", refreshAll)]));
      } else {
        append(holder, notice("error", "Couldn’t load your folders", [err.message || "The server reported an unexpected problem."], [button("Refresh", "button-quiet", refreshAll)]));
      }
    }

    var roots = state.roots && Array.isArray(state.roots.roots) ? state.roots.roots : null;
    if (!roots) {
      show(list, false);
      show(empty, false);
    } else if (roots.length === 0) {
      show(list, false);
      show(empty, true);
    } else {
      roots.forEach(function (root) {
        var li = el("li", null, String(root));
        if (state.newRoot && root === state.newRoot) {
          li.className = "is-new";
        }
        append(list, li);
      });
      show(list, true);
      show(empty, false);
    }
    var p = configPath();
    clear(cfgLine);
    if (p) {
      append(cfgLine, "Saved in ", el("span", "path-text", p));
      show(cfgLine, true);
    } else {
      show(cfgLine, false);
    }
  }

  function setFolderError(message) {
    var input = byId("folder-path");
    var error = byId("folder-path-error");
    if (message) {
      input.setAttribute("aria-invalid", "true");
      setText(error, message);
    } else {
      input.removeAttribute("aria-invalid");
      setText(error, "");
    }
  }

  async function addFolder() {
    if (state.addPending) {
      return;
    }
    var input = byId("folder-path");
    var result = byId("add-result");
    var path = input.value.trim();
    clear(result);
    show(result, false);
    setFolderError("");

    if (!path) {
      setFolderError("Type the path of a folder.");
      input.focus();
      return;
    }
    var maxBytes = numberOr(session.limits && session.limits.root_path_bytes, 4096);
    if (new TextEncoder().encode(path).length > maxBytes) {
      setFolderError("That path is too long (over " + maxBytes + " bytes).");
      input.focus();
      return;
    }
    if (path.indexOf("\u0000") !== -1) {
      setFolderError("That path contains an invalid character.");
      input.focus();
      return;
    }

    state.addPending = true;
    renderControls();
    try {
      if (!state.roots) {
        await refreshRoots();
      }
      if (!state.roots || typeof state.roots.revision !== "string") {
        if (state.rootsError) {
          input.focus();
        }
        return;
      }
      var before = state.roots.roots.slice();
      var response = await api("POST", "/api/roots", { path: path, revision: state.roots.revision });
      var data = response.data;
      if (!data || !Array.isArray(data.roots)) {
        throw new ApiFailure("protocol", response.status, "protocol", "The server sent an unexpected reply.");
      }
      var added = null;
      data.roots.forEach(function (r) {
        if (added === null && before.indexOf(r) === -1) {
          added = r;
        }
      });
      if (added === null && data.roots.length) {
        added = data.roots[data.roots.length - 1];
      }
      state.roots = data;
      state.rootsError = null;
      state.newRoot = added;
      input.value = "";
      renderFolders();

      var parts = [];
      parts.push(notice("info", "Folder added", [
        ["Saved as ", el("span", "path-text", String(added || path))],
        "Choose Update index to read its files.",
      ]));
      if (data.daemon_restart_required) {
        parts.push(notice("caution", "Restart the background service", [
          str(data.warning) || "A seekr daemon process is running. Restart it so it sees the new folder.",
        ]));
      }
      parts.forEach(function (n) {
        append(result, n);
      });
      show(result, true);
      result.focus();
      announce("Folder added: " + (added || path) + (data.daemon_restart_required ? ". Restart the background service to use it." : "."));
      refreshStatus();
    } catch (err) {
      handleAddFolderError(err);
    } finally {
      state.addPending = false;
      renderControls();
    }
  }

  function handleAddFolderError(err) {
    var input = byId("folder-path");
    var result = byId("add-result");
    var fieldMessage = err.fields && typeof err.fields.path === "string" ? err.fields.path : "";
    if (err.kind === "network") {
      setFolderError("Seekr isn’t responding. Your path is kept; reconnect and try again.");
      input.focus();
      return;
    }
    if (err.code === "invalid_root" || err.code === "roots_limit" || err.code === "invalid_request") {
      setFolderError(fieldMessage || err.message || "That path can’t be added.");
      input.focus();
      return;
    }
    if (err.code === "duplicate_root") {
      setFolderError(err.message || "That folder is already in the list.");
      input.focus();
      return;
    }
    if (err.code === "stale_config") {
      setFolderError("Your settings file changed since this page loaded. The list has been refreshed; choose Add folder again.");
      refreshRoots();
      input.focus();
      return;
    }
    var title = "Couldn’t add the folder";
    var bodies = [err.message];
    if (err.code === "config_invalid") {
      title = "Seekr can’t safely edit your settings file";
      bodies = [err.message, ["Add the folder by editing this file by hand: ", el("span", "path-text", configPath())]];
    } else if (err.code === "busy") {
      bodies = ["Another task is running. Add the folder when it finishes."];
    } else if (err.code === "config_write_failed") {
      bodies = [err.message, "Nothing was changed. Check that the settings folder is writable."];
    } else if (!err.message) {
      bodies = ["The server reported an unexpected problem (" + (err.status || "no status") + ")."];
    }
    append(result, notice("error", title, bodies));
    show(result, true);
    result.focus();
    announce(title + ".");
  }

  // ---------------------------------------------------------------- views

  function setView(name, moveFocus) {
    state.view = name;
    ["search", "folders"].forEach(function (v) {
      var tab = byId("tab-" + v);
      var view = byId("view-" + v);
      var active = v === name;
      show(view, active);
      if (active) {
        tab.setAttribute("aria-current", "page");
      } else {
        tab.removeAttribute("aria-current");
      }
    });
    if (moveFocus) {
      var heading = byId(name === "search" ? "search-heading" : "folders-heading");
      heading.focus();
    }
    if (name === "folders" && !state.roots && state.connected) {
      refreshRoots();
    }
  }

  function applyLimits() {
    var limits = session.limits;
    if (!limits) {
      return;
    }
    if (isCount(limits.query_chars)) {
      byId("query").maxLength = limits.query_chars;
    }
    if (isCount(limits.k_max)) {
      byId("filter-k").max = String(limits.k_max);
    }
    if (isCount(limits.k_min)) {
      byId("filter-k").min = String(limits.k_min);
    }
    if (isCount(limits.per_file_max)) {
      byId("filter-per-file").max = String(limits.per_file_max);
    }
    if (isCount(limits.per_file_min)) {
      byId("filter-per-file").min = String(limits.per_file_min);
    }
  }

  function renderAll() {
    renderStatus();
    renderControls();
    renderJob();
    renderFolders();
    renderResultsState();
  }

  function refreshAll() {
    if (!state.connected) {
      connect(false);
      return;
    }
    refreshStatus();
    refreshRoots();
  }

  // ---------------------------------------------------------------- wiring

  function init() {
    byId("search-form").addEventListener("submit", function (event) {
      event.preventDefault();
      runSearch();
    });
    byId("add-folder-form").addEventListener("submit", function (event) {
      event.preventDefault();
      addFolder();
    });
    byId("folder-path").addEventListener("input", function () {
      if (byId("folder-path").getAttribute("aria-invalid") === "true") {
        setFolderError("");
      }
    });
    document.querySelectorAll(".view-tab").forEach(function (tab) {
      tab.addEventListener("click", function () {
        setView(tab.getAttribute("data-view"), true);
      });
    });
    document.querySelectorAll('input[name="mode"]').forEach(function (radio) {
      radio.addEventListener("change", function () {
        state.forcedFromMode = null;
      });
    });
    ["filter-path", "filter-kind", "filter-since", "filter-k", "filter-per-file", "filter-since-date"].forEach(function (id) {
      byId(id).addEventListener("input", renderFilterCount);
      byId(id).addEventListener("change", renderFilterCount);
    });
    byId("filters-clear").addEventListener("click", function () {
      clearFilters();
      byId("filter-path").focus();
    });
    byId("update-index").addEventListener("click", function () {
      startJob("index");
    });
    byId("check-model").addEventListener("click", function () {
      startJob("model_check");
    });
    byId("status-refresh").addEventListener("click", refreshAll);
    byId("reconnect").addEventListener("click", function () {
      connect(true);
    });
    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "visible") {
        refreshAll();
      }
    });
    window.addEventListener("online", function () {
      if (!state.connected) {
        connect(true);
      }
    });

    renderFilterCount();
    renderAll();
    connect(false);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

/*
  Seekr local workspace.

  Security rules followed throughout this file:
  - The session token lives only in the `session` closure below. It is never
    written to a URL, the DOM, storage, or browser logs.
  - Every server-provided string (paths, locators, snippets, messages) is
    rendered with textContent / text nodes. Highlighting builds <mark>
    elements from text nodes; no HTML strings are parsed.
  - Blob object URLs created for page previews are always revoked when
    closing, switching previews, or unloading.
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
    course: null,
    courseError: null,
    courseSeq: 0,
    job: null,
    jobTimer: null,
    lastJob: null,
    searchSeq: 0,
    searchInFlight: false,
    searchQueued: false,
    searchShown: false,
    searchError: null,
    lastHits: [],
    lastSpread: [],
    lastTookMs: null,
    lastQuery: "",
    lastTerms: [],
    visibleHits: [],
    selectedSpreadKey: null,
    activeHitIndex: -1,
    previewSeq: 0,
    previewAbort: null,
    previewOpen: false,
    previewChunkId: null,
    previewHit: null,
    previewData: null,
    previewObjectUrl: null,
    openPending: false,
    forcedFromMode: null,
    view: "search",
    addPending: false,
    removePending: false,
    downloadStarting: false,
    lastDownload: null,
    firstRunDismissed: false,
    lastModelState: null,
    statusSeq: 0,
    statusError: null,
    rootsSeq: 0,
    newRoot: null,
    rootsNotice: null,
    pollInFlight: false,
    finishedJobs: Object.create(null),
  };

  var KIND_LABELS = {
    pdf: "PDF",
    markdown: "Markdown",
    text: "Plain text",
    code: "Source code",
    transcript: "Transcript",
    word: "Word",
    slides: "Slides",
    notebook: "Notebook",
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

  function displayModule(name) {
    if (typeof name !== "string") {
      return "";
    }
    return name.replace(/[-_\s]+/g, " ").trim();
  }

  function describeLocator(locator) {
    var text = str(locator).trim();
    if (!text) {
      return "";
    }
    if (text.charAt(0) === "\u00a7") {
      var title = text.slice(1).trim();
      return title ? "section " + title : text;
    }
    var patterns = [
      [/^p\.(\d+(?:-\d+)?)$/, "page "],
      [/^L(\d+-\d+)$/, "lines "],
      [/^L(\d+)$/, "line "],
      [/^slide\s+(\d+)$/, "slide "],
      [/^cell\s+(\d+)$/, "cell "],
      [/^\u00b6(\d+)$/, "paragraph "],
    ];
    for (var i = 0; i < patterns.length; i += 1) {
      var match = patterns[i][0].exec(text);
      if (match) {
        return patterns[i][1] + match[1];
      }
    }
    if (/^(?:\d+:)?\d{1,2}:\d{2}$/.test(text)) {
      return "at " + text;
    }
    return text;
  }

  function formatPosixLocation(path, locator) {
    var base = str(path);
    var desc = describeLocator(locator);
    return desc ? base + " (" + desc + ")" : base;
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

  async function apiRequest(method, path, body, options) {
    var opts = options || {};
    try {
      await ensureSession(false);
    } catch (err) {
      if (err.kind === "network") {
        markDisconnected();
      }
      throw err;
    }
    var headers = {
      "X-Seekr-Token": session.token,
      Accept: opts.accept || "application/json",
    };
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
    if (response.ok) {
      if (opts.responseType === "blob") {
        try {
          return { status: response.status, data: await response.blob() };
        } catch (err) {
          if (isAbort(err)) {
            throw new ApiFailure("aborted");
          }
          throw new ApiFailure("protocol", response.status, "protocol", "Could not read preview image.");
        }
      }
      return { status: response.status, data: await readJson(response) };
    }
    var data = await readJson(response);
    var failure = failureFrom(response, data);
    if (failure.status === 403 && failure.code === "forbidden" && !opts.retried) {
      try {
        await ensureSession(true);
      } catch (err) {
        if (err.kind === "network") {
          markDisconnected();
        }
        throw err;
      }
      return apiRequest(method, path, body, {
        signal: opts.signal,
        retried: true,
        accept: opts.accept,
        responseType: opts.responseType,
      });
    }
    throw failure;
  }

  function api(method, path, body, options) {
    return apiRequest(method, path, body, options);
  }

  function apiBlob(path, options) {
    var opts = options || {};
    return apiRequest("GET", path, undefined, {
      signal: opts.signal,
      accept: "image/png",
      responseType: "blob",
    });
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
    await Promise.all([refreshStatus(), refreshRoots(), refreshCourse()]);
    if (state.job && !state.jobTimer) {
      pollJob();
    }
  }

  // ---------------------------------------------------------------- status & course

  function engineBusy() {
    if (state.job) {
      return true;
    }
    var lane = state.status && state.status.activity ? state.status.activity.lane : "idle";
    return lane === "index" || lane === "model_check" || lane === "download";
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
      var nextModelState = data.model.state;
      if (nextModelState === "unavailable" && state.lastModelState !== "unavailable") {
        state.firstRunDismissed = false;
      }
      state.lastModelState = nextModelState;
      state.status = data;
      state.statusError = null;
      var activity = data.activity || {};
      if (!state.job && typeof activity.job_id === "string" && activity.job_id &&
          !state.finishedJobs[activity.job_id] &&
          (activity.lane === "index" || activity.lane === "model_check" || activity.lane === "download")) {
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
    renderFirstRunPanel();
    renderFolders();
    renderControls();
    renderResultsState();
  }

  async function refreshCourse() {
    var seq = ++state.courseSeq;
    try {
      var result = await api("GET", "/api/course");
      if (seq !== state.courseSeq) {
        return;
      }
      var data = result.data;
      if (data && Array.isArray(data.modules)) {
        state.course = data;
        state.courseError = null;
      }
    } catch (err) {
      if ((err.kind === "http" || err.kind === "protocol") && seq === state.courseSeq) {
        state.course = { modules: [] };
        state.courseError = err;
      }
    }
    renderCourseFilters();
  }

  function courseModules() {
    return state.course && Array.isArray(state.course.modules) ? state.course.modules : [];
  }

  function validCourseWeeks() {
    var selectedMod = byId("module-filter").value;
    var seen = Object.create(null);
    var weeks = [];
    courseModules().forEach(function (mod) {
      if (!mod || typeof mod.name !== "string") {
        return;
      }
      if (selectedMod && mod.name !== selectedMod) {
        return;
      }
      (Array.isArray(mod.weeks) ? mod.weeks : []).forEach(function (w) {
        if (typeof w !== "number" || !isFinite(w)) {
          return;
        }
        var n = Math.floor(w);
        if (n < 1 || n > 99 || seen[n]) {
          return;
        }
        seen[n] = true;
        weeks.push(n);
      });
    });
    weeks.sort(function (a, b) {
      return a - b;
    });
    return weeks;
  }

  function populateWeekSelect(selectEl, weeks) {
    var prev = selectEl.value;
    clear(selectEl);
    var anyOpt = el("option", null, "Any");
    anyOpt.value = "";
    append(selectEl, anyOpt);
    var foundPrev = false;
    weeks.forEach(function (w) {
      var val = String(w);
      var opt = el("option", null, "Week " + w);
      opt.value = val;
      if (val === prev) {
        foundPrev = true;
      }
      append(selectEl, opt);
    });
    selectEl.value = foundPrev ? prev : "";
  }

  function renderWeekOptions() {
    var weeks = validCourseWeeks();
    populateWeekSelect(byId("week-from"), weeks);
    populateWeekSelect(byId("week-to"), weeks);
    renderFilterCount();
  }

  function renderCourseFilters() {
    var moduleSelect = byId("module-filter");
    var prevMod = moduleSelect.value;
    clear(moduleSelect);
    var allOpt = el("option", null, "All modules");
    allOpt.value = "";
    append(moduleSelect, allOpt);

    var foundPrev = false;
    courseModules().forEach(function (mod) {
      if (!mod || typeof mod.name !== "string" || !mod.name) {
        return;
      }
      var label = str(mod.display) || mod.name;
      var opt = el("option", null, label);
      opt.value = mod.name;
      if (mod.name === prevMod) {
        foundPrev = true;
      }
      append(moduleSelect, opt);
    });
    moduleSelect.value = foundPrev ? prevMod : "";
    renderWeekOptions();
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
        "Download the search model from the setup panel or provision it locally, then choose Check model. Exact-words search still works.";
    }
    setText(modelDetail, modelText);

    var service = st.service && typeof st.service === "object" ? st.service : null;
    clear(daemonDetail);
    if (service && service.running) {
      setStatusValue(daemonState, "Running", "ok");
      append(daemonDetail, "This page is served by the ", codeText("seekr service"),
        (isCount(service.pid) ? " (process " + service.pid + ")" : ""),
        ". It keeps the search model loaded, watches your folders and keeps the index up to date. Stop it with ",
        codeText("seekr service stop"), ".");
    } else if (st.daemon.present) {
      setStatusValue(daemonState, "Older daemon running", "warn");
      append(daemonDetail, "An older ", codeText("seekr daemon"),
        " process is running. Searches go through it, and it looks after the index itself. Stop it and use ",
        codeText("seekr service start"), " instead.");
    } else {
      setStatusValue(daemonState, "Not running", "off");
      append(daemonDetail, "This page runs in the terminal (", codeText("seekr web --foreground"),
        "), so nothing watches your folders. Run ", codeText("seekr web"),
        " instead to start the background service.");
    }
    show(daemonDetail, true);
  }

  // ---------------------------------------------------------------- controls & first-run

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
      reason = ["Turned off while an older ", codeText("seekr daemon"), " process maintains the index. Stop it (Ctrl+C in its terminal), then choose Refresh. ", codeText("seekr service start"), " replaces it."];
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
      reason = ["Updating needs the local model. Download or check the model once it’s available."];
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

    var addDisabled = state.addPending || !state.connected;
    byId("add-folder-submit").disabled = addDisabled;
    byId("first-run-add-folder").disabled = addDisabled;

    var downloadBtn = byId("download-model-button");
    downloadBtn.disabled = !state.connected || !st || configInvalid() || engineBusy() || state.downloadStarting;

    var firstRunUpdate = byId("first-run-update-index");
    var hasRoots = isCount(rootCount()) && rootCount() > 0;
    show(firstRunUpdate, hasRoots && !updateDisabled);
    firstRunUpdate.disabled = updateDisabled;

    applyModeAvailability();
  }

  function renderFirstRunPanel() {
    var panel = byId("first-run-panel");
    var st = state.status;
    if (!st || configInvalid()) {
      show(panel, false);
      return;
    }
    var roots = rootCount();
    var noRoots = roots === 0;
    var modelUnavailable = Boolean(st.model && st.model.state === "unavailable");
    var downloadActive = Boolean(state.job && state.job.operation === "download");
    var downloadFailed = Boolean(state.lastDownload && state.lastDownload.failed);
    var shouldShow = downloadActive || downloadFailed || (!state.firstRunDismissed && (noRoots || modelUnavailable));

    show(panel, shouldShow);
    if (!shouldShow) {
      return;
    }

    var lede = byId("first-run-lede");
    if (noRoots && modelUnavailable) {
      lede.textContent = "Add a folder to search and download the local search model.";
    } else if (noRoots) {
      lede.textContent = "Add your notes or coursework folders below, and download the search model if you haven’t already.";
    } else if (modelUnavailable) {
      lede.textContent = "The local search model isn’t available yet. Download it below, or use Exact words search offline.";
    } else {
      lede.textContent = "Downloading the local search model and language grammars.";
    }

    var folderForm = byId("first-run-folder-form");
    show(folderForm, noRoots || Boolean(byId("first-run-folder-status").textContent) || Boolean(byId("first-run-folder-error").textContent));

    var statusWrap = byId("first-run-download-status");
    var summary = byId("first-run-download-summary");
    var linesEl = byId("first-run-download-lines");
    var lines = [];
    var summaryText = "";

    if (downloadActive) {
      var elapsed = state.job.data ? formatElapsed(state.job.data.elapsed_ms) : "";
      summaryText = "Downloading search model" + (elapsed ? " (" + elapsed + ")" : "") + "…";
      if (state.job.data && state.job.data.report && Array.isArray(state.job.data.report.lines)) {
        lines = state.job.data.report.lines;
      }
    } else if (state.lastDownload) {
      summaryText = state.lastDownload.summary || "";
      lines = Array.isArray(state.lastDownload.lines) ? state.lastDownload.lines : [];
    }

    summary.textContent = summaryText;
    if (lines.length) {
      linesEl.textContent = lines.join("\n");
      show(linesEl, true);
    } else {
      linesEl.textContent = "";
      show(linesEl, false);
    }
    show(statusWrap, Boolean(summaryText || lines.length));
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
      return;
    }
    pollJob();
  }

  function continueAfterReplacedJob() {
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
      if (data.operation === "index" || data.operation === "model_check" || data.operation === "download") {
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
        return;
      }
      state.job = null;
      state.finishedJobs[job.id] = true;
      if (err.code === "job_not_found") {
        state.lastJob = { operation: job.operation, lost: true };
      } else {
        state.lastJob = { operation: job.operation, failure: err };
      }
      if (job.operation === "download") {
        state.lastDownload = {
          failed: true,
          summary: err.message || "Download status could not be checked.",
          lines: [],
        };
      }
      renderAll();
      refreshStatus();
    }
  }

  function finishJob(job, data) {
    state.finishedJobs[job.id] = true;
    state.job = null;
    state.lastJob = { operation: job.operation, data: data };
    if (job.operation === "download") {
      var rep = data.report && typeof data.report === "object" ? data.report : {};
      var lines = Array.isArray(rep.lines) ? rep.lines : [];
      var succeeded = data.state === "succeeded";
      state.lastDownload = {
        failed: !succeeded,
        summary: succeeded
          ? "Search model downloaded. Choose Update index or Check model."
          : (str(data.error) || "Model download failed."),
        lines: lines,
      };
    }
    announce(jobSummary(job.operation, data));
    renderAll();
    refreshStatus();
    if (job.operation === "index" || job.operation === "download") {
      refreshRoots();
      refreshCourse();
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
    if (operation === "download") {
      if (data.state === "succeeded") {
        return "Model download finished" + (took ? " in " + took : "") + ".";
      }
      return "Model download failed" + (data.error ? ": " + data.error : ".");
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
      var isDownload = job.operation === "download";
      var titleText = isIndex ? "Updating index…" : (isDownload ? "Downloading search model…" : "Checking model…");
      append(body, el("p", "job-title", titleText));
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
      append(body, jobStartError(last.startError, op));
      return;
    }
    if (last.lost) {
      var lostTitle = op === "model_check" ? "Model check" : (op === "download" ? "Model download" : "Index update");
      append(body,
        el("p", "job-title", lostTitle),
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
          el("p", "status-detail", "Download the search model from the setup panel or provision it locally, then check again."));
      }
      return;
    }

    if (op === "download") {
      var dlReport = data.report && typeof data.report === "object" ? data.report : {};
      var dlLines = Array.isArray(dlReport.lines) ? dlReport.lines : [];
      if (data.state === "succeeded") {
        append(body,
          el("p", "job-title", "Search model downloaded"),
          took ? el("p", "job-line", "Finished in " + took + ".") : null);
      } else {
        append(body,
          el("p", "job-title is-bad", "Model download failed"),
          el("p", "job-line", str(data.error) || "Could not download the search model."));
      }
      if (dlLines.length) {
        append(body, el("pre", "download-lines", dlLines.join("\n")));
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

  function jobStartError(err, operation) {
    var bodies = [err.message];
    var actions = [];
    var title = operation === "index"
      ? "Couldn’t start the index update"
      : (operation === "download" ? "Couldn’t start the model download" : "Couldn’t start the model check");
    if (err.code === "no_roots") {
      bodies = ["There are no folders to read yet."];
      actions.push(button("Add folder", "button-secondary", function () {
        setView("folders", true);
      }));
    } else if (err.code === "daemon_active") {
      bodies = [err.message, ["An older ", codeText("seekr daemon"), " process maintains the index. Stop it to update from this page; ", codeText("seekr service start"), " replaces it."]];
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

  async function startDownload() {
    if (state.downloadStarting || engineBusy()) {
      return;
    }
    state.downloadStarting = true;
    state.firstRunDismissed = false;
    state.lastDownload = { failed: false, summary: "Starting model download…", lines: [] };
    renderFirstRunPanel();
    renderControls();
    try {
      var result = await api("POST", "/api/setup/download", { rerank: false });
      var data = result.data;
      if (!data || typeof data.job_id !== "string" || !data.job_id) {
        throw new ApiFailure("protocol", result.status, "protocol", "The server sent an unexpected reply.");
      }
      announce("Search model download started.");
      startPolling(data.job_id, "download");
    } catch (err) {
      if (err.kind === "network") {
        return;
      }
      state.lastDownload = {
        failed: true,
        summary: err.message || "Couldn’t start model download.",
        lines: [],
      };
      state.lastJob = { operation: "download", startError: err };
      renderFirstRunPanel();
      renderJob();
      announce("Model download not started. " + (err.message || ""));
      refreshStatus();
    } finally {
      state.downloadStarting = false;
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

  function appendSnippetHighlights(parent, text, rawHighlights, fallbackTerms) {
    if (!text) {
      return;
    }
    if (!Array.isArray(rawHighlights)) {
      appendHighlighted(parent, text, fallbackTerms || []);
      return;
    }
    var len = text.length;
    var spans = [];
    rawHighlights.forEach(function (pair) {
      if (!Array.isArray(pair) || pair.length < 2) {
        return;
      }
      var start = Number(pair[0]);
      var end = Number(pair[1]);
      if (!isFinite(start) || !isFinite(end)) {
        return;
      }
      start = Math.max(0, Math.min(len, Math.floor(start)));
      end = Math.max(0, Math.min(len, Math.floor(end)));
      if (start < end) {
        spans.push([start, end]);
      }
    });
    spans.sort(function (a, b) {
      return a[0] - b[0] || a[1] - b[1];
    });
    var cursor = 0;
    spans.forEach(function (span) {
      var start = Math.max(cursor, span[0]);
      var end = span[1];
      if (start >= end) {
        return;
      }
      if (start > cursor) {
        append(parent, text.slice(cursor, start));
      }
      append(parent, el("mark", null, text.slice(start, end)));
      cursor = end;
    });
    if (cursor < len) {
      append(parent, text.slice(cursor));
    }
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
      return { error: label + " must be a whole number from " + min + " to " + max + ".", input: input };
    }
    var n = parseInt(raw, 10);
    if (n < min || n > max) {
      input.setAttribute("aria-invalid", "true");
      return { error: label + " must be from " + min + " to " + max + ".", input: input };
    }
    input.removeAttribute("aria-invalid");
    return { value: n };
  }

  function buildWeekFilter() {
    var rawFrom = byId("week-from").value.trim();
    var rawTo = byId("week-to").value.trim();
    if (!rawFrom && !rawTo) {
      return "";
    }
    if (rawFrom && rawTo) {
      var a = parseInt(rawFrom, 10);
      var b = parseInt(rawTo, 10);
      if (isFinite(a) && isFinite(b)) {
        var lo = Math.max(1, Math.min(99, Math.min(a, b)));
        var hi = Math.max(1, Math.min(99, Math.max(a, b)));
        return lo === hi ? String(lo) : lo + "-" + hi;
      }
    }
    if (rawFrom) {
      var start = parseInt(rawFrom, 10);
      if (isFinite(start)) {
        var clampedStart = Math.max(1, Math.min(99, start));
        return clampedStart + "-99";
      }
    }
    if (rawTo) {
      var end = parseInt(rawTo, 10);
      if (isFinite(end)) {
        var clampedEnd = Math.max(1, Math.min(99, end));
        return "1-" + clampedEnd;
      }
    }
    return "";
  }

  function buildSearchRequest() {
    var query = byId("query").value;
    var body = { query: query, mode: selectedMode(), spread: true };
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
    var mod = byId("module-filter").value;
    if (mod) {
      body.module = mod;
    }
    var week = buildWeekFilter();
    if (week) {
      body.week = week;
    }
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
      submit.setAttribute("aria-disabled", "true");
    } else {
      submit.removeAttribute("aria-disabled");
    }
  }

  async function runSearch() {
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
      state.lastHits = [];
      state.lastSpread = [];
      state.visibleHits = [];
      state.selectedSpreadKey = null;
      state.activeHitIndex = -1;
      closePreview();
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
    clear(byId("spread-strip"));
    show(byId("spread-strip"), false);
    setText(byId("results-meta"), "");
  }

  function showSearchError(node) {
    state.searchShown = false;
    state.searchError = node;
    state.lastHits = [];
    state.lastSpread = [];
    state.visibleHits = [];
    state.selectedSpreadKey = null;
    state.activeHitIndex = -1;
    closePreview();
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
      title = "The older seekr daemon didn’t answer";
      bodies = [err.message, ["Stop the older ", codeText("seekr daemon"), " process so this page searches the index directly; ", codeText("seekr service start"), " replaces it."]];
    } else if (!err.message) {
      bodies = ["The server reported an unexpected problem (" + (err.status || "no status") + "). Try again."];
    }
    showSearchError(notice("error", title, bodies, actions));
    announce(title + ".");
    if (err.kind === "http" && code !== "invalid_request") {
      refreshStatus();
    }
  }

  function spreadKey(group) {
    var mod = typeof group.module === "string" ? group.module : "__none__";
    var week = typeof group.week === "number" && isFinite(group.week) ? String(group.week) : "__none__";
    return mod + "::" + week;
  }

  function spreadChipLabel(group) {
    if (typeof group.week !== "number" || !isFinite(group.week)) {
      return "other";
    }
    return group.first ? "Week " + group.week + " (first)" : "Week " + group.week;
  }

  function findSelectedSpreadGroup() {
    if (!state.selectedSpreadKey || !Array.isArray(state.lastSpread)) {
      return null;
    }
    for (var i = 0; i < state.lastSpread.length; i += 1) {
      if (spreadKey(state.lastSpread[i]) === state.selectedSpreadKey) {
        return state.lastSpread[i];
      }
    }
    return null;
  }

  function hitMatchesSpreadGroup(hit, group) {
    if (!group) {
      return true;
    }
    if (Array.isArray(group.chunk_ids) && group.chunk_ids.indexOf(hit.chunk_id) !== -1) {
      return true;
    }
    var hitMod = typeof hit.module === "string" ? hit.module : null;
    var groupMod = typeof group.module === "string" ? group.module : null;
    var hitWeek = typeof hit.week === "number" && isFinite(hit.week) ? hit.week : null;
    var groupWeek = typeof group.week === "number" && isFinite(group.week) ? group.week : null;
    return hitMod === groupMod && hitWeek === groupWeek;
  }

  function renderSpreadStrip() {
    var strip = byId("spread-strip");
    clear(strip);
    var groups = Array.isArray(state.lastSpread) ? state.lastSpread : [];
    if (!state.searchShown || !groups.length) {
      show(strip, false);
      return;
    }
    show(strip, true);

    var header = el("div", "spread-head");
    append(header, el("span", "spread-title", "Appears in"));
    var allBtn = button("All results", "button-quiet spread-clear" + (state.selectedSpreadKey === null ? " is-active" : ""), function () {
      if (state.selectedSpreadKey !== null) {
        state.selectedSpreadKey = null;
        renderSpreadStrip();
        renderVisibleHits();
      }
    });
    allBtn.setAttribute("aria-pressed", state.selectedSpreadKey === null ? "true" : "false");
    append(header, allBtn);
    append(strip, header);

    var byModule = [];
    var modIndex = Object.create(null);
    groups.forEach(function (group) {
      if (!group || typeof group !== "object") {
        return;
      }
      var modKey = typeof group.module === "string" ? group.module : "__none__";
      if (modIndex[modKey] === undefined) {
        modIndex[modKey] = byModule.length;
        byModule.push({ module: group.module || null, items: [] });
      }
      byModule[modIndex[modKey]].items.push(group);
    });

    var rows = el("div", "spread-rows");
    byModule.forEach(function (entry) {
      var row = el("div", "spread-row");
      var modLabel = entry.module ? (displayModule(entry.module) || "other") : "other";
      append(row, el("span", "spread-module", modLabel));
      var chips = el("div", "spread-chips");
      entry.items.forEach(function (group) {
        var key = spreadKey(group);
        var isActive = state.selectedSpreadKey === key;
        var cls = "spread-chip" + (group.first ? " is-first" : "") + (isActive ? " is-active" : "");
        var chip = el("button", cls, spreadChipLabel(group));
        chip.type = "button";
        chip.setAttribute("aria-pressed", isActive ? "true" : "false");
        chip.addEventListener("click", function () {
          state.selectedSpreadKey = state.selectedSpreadKey === key ? null : key;
          renderSpreadStrip();
          renderVisibleHits();
        });
        append(chips, chip);
      });
      append(row, chips);
      append(rows, row);
    });
    append(strip, rows);
  }

  function renderHits(body, data) {
    state.searchShown = true;
    state.searchError = null;
    state.lastHits = (data.hits || []).filter(function (hit) {
      return Boolean(hit && typeof hit === "object" && typeof hit.path === "string");
    });
    state.lastSpread = Array.isArray(data.spread) ? data.spread : [];
    state.lastTookMs = data.took_ms;
    state.lastQuery = body.query.trim();
    state.lastTerms = queryTerms(body.query);
    state.selectedSpreadKey = null;
    state.activeHitIndex = -1;
    closePreview();

    renderSpreadStrip();
    renderVisibleHits();
  }

  function renderVisibleHits() {
    var holder = byId("results-state");
    clear(holder);
    var list = byId("hit-list");
    clear(list);
    var meta = byId("results-meta");

    var totalHits = state.lastHits.length;
    if (totalHits === 0) {
      show(list, false);
      setText(meta, "");
      state.visibleHits = [];
      state.activeHitIndex = -1;
      append(holder, notice("quiet", "No passages match", [
        ["Nothing in the index matched ", el("q", null, state.lastQuery), "."],
        hasFilters() ? "Try clearing the filters, using fewer words, or a different match option." : "Try fewer or different words, or a different match option.",
      ], hasFilters() ? [button("Clear filters and search again", "button-quiet", function () {
        clearFilters();
        runSearch();
      })] : []));
      announce("No passages match.");
      return;
    }

    var activeGroup = findSelectedSpreadGroup();
    var filtered = activeGroup
      ? state.lastHits.filter(function (hit) { return hitMatchesSpreadGroup(hit, activeGroup); })
      : state.lastHits.slice();
    state.visibleHits = filtered;

    if (state.activeHitIndex >= filtered.length) {
      state.activeHitIndex = filtered.length - 1;
    }

    if (filtered.length === 0) {
      show(list, false);
      setText(meta, "Showing 0 of " + plural(totalHits, "passage", "passages") + ".");
      append(holder, notice("quiet", "No matching passages in this group", [
        "This group was found in the broader course spread, outside the top " + formatCount(totalHits) + " results shown.",
      ], [button("All results", "button-quiet", function () {
        state.selectedSpreadKey = null;
        renderSpreadStrip();
        renderVisibleHits();
      })]));
      return;
    }

    filtered.forEach(function (hit, i) {
      append(list, renderHit(hit, i, state.lastTerms));
    });
    show(list, true);

    var took = formatTook(state.lastTookMs);
    var summary;
    if (activeGroup) {
      var modName = activeGroup.module ? (displayModule(activeGroup.module) || "other") : "other";
      summary = "Showing " + formatCount(filtered.length) + " of " + plural(totalHits, "passage", "passages") +
        " in " + modName + " · " + spreadChipLabel(activeGroup) + ".";
    } else {
      summary = plural(filtered.length, "passage", "passages") + (took ? ", found in " + took : "") + ".";
    }
    setText(meta, summary);
    announce(summary);
  }

  function hitHeaderText(hit) {
    var parts = [];
    var mod = displayModule(hit.module);
    if (mod) {
      parts.push(mod);
    }
    if (typeof hit.week === "number" && isFinite(hit.week)) {
      parts.push("Week " + hit.week);
    }
    var file = baseName(hit.path);
    if (file) {
      parts.push(file);
    }
    var loc = describeLocator(hit.locator);
    if (loc) {
      parts.push(loc);
    }
    return parts.join(" · ") || hit.path;
  }

  function hitLocationText(hit, usePosix) {
    if (usePosix) {
      return str(hit.posix_location) || formatPosixLocation(hit.path, hit.locator);
    }
    return str(hit.location) || str(hit.posix_location) || formatPosixLocation(hit.path, hit.locator);
  }

  function renderHit(hit, index, terms) {
    var path = hit.path;
    var locator = str(hit.locator);
    var locDesc = describeLocator(locator);
    var li = el("li", "hit");
    li.tabIndex = -1;
    li.setAttribute("data-hit-index", String(index));
    if (typeof hit.chunk_id === "number") {
      li.setAttribute("data-chunk-id", String(hit.chunk_id));
    }
    if (index === state.activeHitIndex) {
      li.classList.add("is-active");
    }
    if (state.previewOpen && state.previewChunkId === hit.chunk_id) {
      li.classList.add("is-previewing");
    }

    var nameId = "hit-name-" + index;
    var head = el("div", "hit-head");
    var name = el("h3", "hit-name", hitHeaderText(hit));
    name.id = nameId;
    append(head, name);

    var kind = str(hit.kind);
    if (kind) {
      var kindLabel = Object.prototype.hasOwnProperty.call(KIND_LABELS, kind) ? KIND_LABELS[kind] : kind;
      append(head, el("span", "hit-kind", kindLabel));
    }
    append(li, head);

    append(li, el("p", "hit-path", path));

    if (locDesc) {
      var loc = el("p", "hit-locator");
      append(loc, el("span", "visually-hidden", "Location: "), locDesc);
      append(li, loc);
    }

    var snippet = el("p", "hit-snippet");
    appendSnippetHighlights(snippet, str(hit.snippet), hit.highlights, terms);
    append(li, snippet);

    var actions = el("div", "hit-actions");
    var previewBtn = el("button", "button button-quiet hit-preview", "Preview");
    previewBtn.type = "button";
    previewBtn.setAttribute("aria-describedby", nameId);
    previewBtn.addEventListener("click", function (event) {
      event.stopPropagation();
      selectAndPreviewHit(index);
    });

    var copyButton = el("button", "button button-quiet hit-copy", "Copy location");
    copyButton.type = "button";
    copyButton.setAttribute("aria-describedby", nameId);
    var status = el("span", "copy-status");
    status.setAttribute("aria-live", "polite");
    copyButton.addEventListener("click", function (event) {
      event.stopPropagation();
      var text = hitLocationText(hit, byId("posix-toggle").checked);
      copyLocation(text, actions, status, "copy-field-" + index);
    });
    append(actions, previewBtn, copyButton, status);
    append(li, actions);

    li.addEventListener("click", function (event) {
      if (event.target && event.target.closest("button, a, input, label")) {
        return;
      }
      selectAndPreviewHit(index);
    });

    return li;
  }

  function updateActiveHitClasses() {
    var nodes = byId("hit-list").querySelectorAll(".hit");
    nodes.forEach(function (node, i) {
      node.classList.toggle("is-active", i === state.activeHitIndex);
      var cid = Number(node.getAttribute("data-chunk-id"));
      node.classList.toggle("is-previewing", state.previewOpen && cid === state.previewChunkId);
    });
  }

  function setActiveHit(index, shouldFocus) {
    if (!state.visibleHits.length) {
      state.activeHitIndex = -1;
      return;
    }
    var clamped = Math.max(0, Math.min(state.visibleHits.length - 1, index));
    state.activeHitIndex = clamped;
    updateActiveHitClasses();
    var target = byId("hit-list").querySelector('.hit[data-hit-index="' + clamped + '"]');
    if (target) {
      target.scrollIntoView({ block: "nearest" });
      if (shouldFocus) {
        target.focus({ preventScroll: true });
      }
    }
  }

  function selectAndPreviewHit(index) {
    if (index < 0 || index >= state.visibleHits.length) {
      return;
    }
    setActiveHit(index, false);
    openPreview(state.visibleHits[index]);
  }

  // ---------------------------------------------------------------- preview & open

  function revokePreviewObjectUrl() {
    var img = byId("preview-image");
    if (img) {
      img.removeAttribute("src");
      show(img, false);
    }
    if (state.previewObjectUrl) {
      URL.revokeObjectURL(state.previewObjectUrl);
      state.previewObjectUrl = null;
    }
  }

  function activePreviewLocation() {
    var posixChecked = byId("posix-toggle").checked;
    var preview = state.previewData;
    var hit = state.previewHit;
    if (posixChecked) {
      if (hit && str(hit.posix_location)) {
        return hit.posix_location;
      }
      if (preview) {
        return formatPosixLocation(preview.path, preview.locator);
      }
      if (hit) {
        return formatPosixLocation(hit.path, hit.locator);
      }
      return "";
    }
    if (preview && str(preview.location)) {
      return preview.location;
    }
    if (hit && str(hit.location)) {
      return hit.location;
    }
    if (preview) {
      return formatPosixLocation(preview.path, preview.locator);
    }
    return hit ? formatPosixLocation(hit.path, hit.locator) : "";
  }

  function updatePreviewLocationDisplay() {
    byId("preview-location").textContent = activePreviewLocation();
    var fallback = byId("preview-copy-fallback");
    clear(fallback);
    show(fallback, false);
  }

  function closePreview(restoreFocus) {
    if (state.previewAbort) {
      state.previewAbort.abort();
      state.previewAbort = null;
    }
    state.previewSeq += 1;
    state.previewOpen = false;
    state.previewChunkId = null;
    state.previewHit = null;
    state.previewData = null;
    revokePreviewObjectUrl();

    byId("preview-title").textContent = "Preview";
    byId("preview-location").textContent = "";
    byId("preview-text").textContent = "";
    setText(byId("preview-feedback"), "");
    clear(byId("preview-copy-fallback"));
    show(byId("preview-copy-fallback"), false);
    show(byId("preview-panel"), false);
    byId("results-and-preview").classList.remove("has-preview");
    updateActiveHitClasses();

    if (restoreFocus && state.activeHitIndex >= 0) {
      var activeNode = byId("hit-list").querySelector('.hit[data-hit-index="' + state.activeHitIndex + '"]');
      if (activeNode) {
        activeNode.focus({ preventScroll: true });
      }
    }
  }

  async function openPreview(hit) {
    if (!hit || typeof hit.chunk_id !== "number" || !isFinite(hit.chunk_id) || hit.chunk_id <= 0) {
      return;
    }
    var chunkId = Math.floor(hit.chunk_id);
    if (state.previewAbort) {
      state.previewAbort.abort();
    }
    var controller = typeof AbortController === "function" ? new AbortController() : null;
    state.previewAbort = controller;
    var signal = controller ? controller.signal : undefined;
    var seq = ++state.previewSeq;

    revokePreviewObjectUrl();
    state.previewOpen = true;
    state.previewChunkId = chunkId;
    state.previewHit = hit;
    state.previewData = null;

    byId("preview-title").textContent = hitHeaderText(hit);
    updatePreviewLocationDisplay();
    byId("preview-text").textContent = "Loading preview…";
    setText(byId("preview-feedback"), "");
    byId("open-button").disabled = false;
    byId("copy-location").disabled = false;
    show(byId("preview-panel"), true);
    byId("results-and-preview").classList.add("has-preview");
    updateActiveHitClasses();

    try {
      var res = await api("GET", "/api/preview?chunk=" + encodeURIComponent(String(chunkId)), undefined, { signal: signal });
      if (seq !== state.previewSeq || !state.previewOpen) {
        return;
      }
      var preview = res.data;
      if (!preview || typeof preview !== "object" || typeof preview.text !== "string") {
        throw new ApiFailure("protocol", res.status, "protocol", "The server sent an unexpected preview reply.");
      }
      state.previewData = preview;
      byId("preview-title").textContent = hitHeaderText(hit);
      updatePreviewLocationDisplay();
      byId("preview-text").textContent = preview.text;

      if (preview.page_image) {
        try {
          var imgRes = await apiBlob("/api/preview/page?chunk=" + encodeURIComponent(String(chunkId)), { signal: signal });
          if (seq !== state.previewSeq || !state.previewOpen) {
            return;
          }
          revokePreviewObjectUrl();
          var objectUrl = URL.createObjectURL(imgRes.data);
          state.previewObjectUrl = objectUrl;
          var imgEl = byId("preview-image");
          imgEl.alt = "Page preview for " + baseName(str(preview.path));
          imgEl.src = objectUrl;
          show(imgEl, true);
        } catch (imgErr) {
          if (isAbort(imgErr) || seq !== state.previewSeq || !state.previewOpen) {
            return;
          }
          revokePreviewObjectUrl();
          setText(byId("preview-feedback"), imgErr.message || "Page image could not be loaded.");
        }
      } else {
        revokePreviewObjectUrl();
      }
    } catch (err) {
      if (isAbort(err) || seq !== state.previewSeq || !state.previewOpen) {
        return;
      }
      revokePreviewObjectUrl();
      byId("preview-text").textContent = "";
      setText(byId("preview-feedback"), err.message || "Could not load passage preview.");
    }
  }

  async function openChunk(chunkId) {
    if (typeof chunkId !== "number" || !isFinite(chunkId) || chunkId <= 0) {
      return;
    }
    if (state.openPending) {
      return;
    }
    var targetId = Math.floor(chunkId);
    var openBtn = byId("open-button");
    var feedback = byId("preview-feedback");
    state.openPending = true;
    openBtn.disabled = true;
    if (state.previewOpen) {
      setText(feedback, "Opening…");
    }
    try {
      await api("POST", "/api/open", { chunk: targetId });
      if (state.previewOpen) {
        setText(feedback, "Opened in local app.");
      }
      announce("Opened file.");
    } catch (err) {
      if (err.kind === "network") {
        return;
      }
      var msg = err.message || "Couldn’t open file.";
      if (state.previewOpen) {
        setText(feedback, msg);
      } else {
        var holder = byId("results-state");
        clear(holder);
        append(holder, notice("error", "Couldn’t open file", [msg]));
      }
      announce(msg);
    } finally {
      state.openPending = false;
      openBtn.disabled = false;
    }
  }

  async function copyLocation(text, fallbackHost, statusEl, fieldId) {
    if (fallbackHost.classList && fallbackHost.classList.contains("preview-copy-fallback")) {
      clear(fallbackHost);
      show(fallbackHost, false);
    } else {
      var existing = fallbackHost.querySelector(".copy-fallback");
      if (existing) {
        fallbackHost.removeChild(existing);
      }
    }
    setText(statusEl, "");
    var copied = false;
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function" && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        copied = true;
      } catch (err) {
        copied = false;
      }
    }
    if (copied) {
      setText(statusEl, "Copied");
      return;
    }
    var wrap = el("div", "copy-fallback");
    var label = el("label", null, "Couldn’t copy automatically. The location is selected: press Ctrl+C (or ⌘C) to copy it.");
    label.htmlFor = fieldId;
    var field = el("input");
    field.type = "text";
    field.id = fieldId;
    field.readOnly = true;
    field.value = text;
    append(wrap, label, field);
    append(fallbackHost, wrap);
    if (fallbackHost.classList && fallbackHost.classList.contains("preview-copy-fallback")) {
      show(fallbackHost, true);
    }
    field.focus();
    field.select();
    setText(statusEl, "Select and copy below");
  }

  // ---------------------------------------------------------------- filters

  function hasFilters() {
    return activeFilterCount() > 0;
  }

  function activeFilterCount() {
    var count = 0;
    if (byId("module-filter").value) {
      count += 1;
    }
    if (byId("week-from").value || byId("week-to").value) {
      count += 1;
    }
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
    byId("module-filter").value = "";
    renderWeekOptions();
    byId("week-from").value = "";
    byId("week-to").value = "";
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
        bodies.push(["An older ", codeText("seekr daemon"), " process is running and maintains the index. Stop it and use ", codeText("seekr service start"), " instead."]);
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
        "The local model is unavailable, so the other match options will fail. Download it in the setup panel or check Local model in Index status.",
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
          ["Edit it by hand to add or remove folders: ", el("span", "path-text", configPath())],
        ], [button("Refresh", "button-quiet", refreshAll)]));
      } else {
        append(holder, notice("error", "Couldn’t load your folders", [err.message || "The server reported an unexpected problem."], [button("Refresh", "button-quiet", refreshAll)]));
      }
    }

    if (state.rootsNotice) {
      append(holder, state.rootsNotice);
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
        var rootStr = String(root);
        var li = el("li", "root-item");
        if (state.newRoot && root === state.newRoot) {
          li.className += " is-new";
        }
        var pathSpan = el("span", "root-path", rootStr);
        var removeBtn = el("button", "button button-quiet remove-root", "Remove");
        removeBtn.type = "button";
        removeBtn.setAttribute("aria-label", "Remove folder " + rootStr);
        removeBtn.disabled = state.removePending || state.addPending || !state.connected;
        removeBtn.addEventListener("click", function () {
          removeFolder(rootStr);
        });
        append(li, pathSpan, removeBtn);
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

  function setFieldError(inputId, errorId, message) {
    var input = byId(inputId);
    var error = byId(errorId);
    if (message) {
      input.setAttribute("aria-invalid", "true");
      setText(error, message);
    } else {
      input.removeAttribute("aria-invalid");
      setText(error, "");
    }
  }

  function setFolderError(message) {
    setFieldError("folder-path", "folder-path-error", message);
  }

  async function submitAddRoot(options) {
    if (state.addPending || state.removePending) {
      return;
    }
    var input = byId(options.inputId);
    var errorId = options.errorId;
    var path = input.value.trim();
    if (options.onClear) {
      options.onClear();
    }
    setFieldError(options.inputId, errorId, "");

    if (!path) {
      setFieldError(options.inputId, errorId, "Type the path of a folder.");
      input.focus();
      return;
    }
    var maxBytes = numberOr(session.limits && session.limits.root_path_bytes, 4096);
    if (new TextEncoder().encode(path).length > maxBytes) {
      setFieldError(options.inputId, errorId, "That path is too long (over " + maxBytes + " bytes).");
      input.focus();
      return;
    }
    if (path.indexOf("\u0000") !== -1) {
      setFieldError(options.inputId, errorId, "That path contains an invalid character.");
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
      state.rootsNotice = null;
      state.newRoot = added;
      input.value = "";
      renderFolders();
      renderFirstRunPanel();
      renderResultsState();

      if (options.onSuccess) {
        options.onSuccess(added || path, data);
      }
      announce("Folder added: " + (added || path) + ".");
      refreshStatus();
    } catch (err) {
      handleAddFolderError(err, options);
    } finally {
      state.addPending = false;
      renderControls();
    }
  }

  function addFolder() {
    var result = byId("add-result");
    return submitAddRoot({
      inputId: "folder-path",
      errorId: "folder-path-error",
      onClear: function () {
        clear(result);
        show(result, false);
      },
      onSuccess: function (addedPath) {
        append(result, notice("info", "Folder added", [
          ["Saved as ", el("span", "path-text", String(addedPath))],
          "Choose Update index to read its files.",
        ]));
        show(result, true);
        result.focus();
      },
      onOtherError: function (node) {
        append(result, node);
        show(result, true);
        result.focus();
      },
    });
  }

  function addFolderFromFirstRun() {
    var statusEl = byId("first-run-folder-status");
    return submitAddRoot({
      inputId: "first-run-folder-path",
      errorId: "first-run-folder-error",
      onClear: function () {
        setText(statusEl, "");
      },
      onSuccess: function (addedPath) {
        setText(statusEl, "Added " + addedPath + ". Choose Update index when ready.");
      },
      onOtherError: function (node, err) {
        setFieldError("first-run-folder-path", "first-run-folder-error", err.message || "Couldn’t add the folder.");
      },
    });
  }

  function handleAddFolderError(err, options) {
    var input = byId(options.inputId);
    var errorId = options.errorId;
    var fieldMessage = err.fields && typeof err.fields.path === "string" ? err.fields.path : "";
    if (err.kind === "network") {
      setFieldError(options.inputId, errorId, "Seekr isn’t responding. Your path is kept; reconnect and try again.");
      input.focus();
      return;
    }
    if (err.code === "invalid_root" || err.code === "roots_limit" || err.code === "invalid_request") {
      setFieldError(options.inputId, errorId, fieldMessage || err.message || "That path can’t be added.");
      input.focus();
      return;
    }
    if (err.code === "duplicate_root") {
      setFieldError(options.inputId, errorId, err.message || "That folder is already in the list.");
      input.focus();
      return;
    }
    if (err.code === "stale_config") {
      setFieldError(options.inputId, errorId, "Your settings file changed since this page loaded. The list has been refreshed; choose Add folder again.");
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
    var errNotice = notice("error", title, bodies);
    if (options.onOtherError) {
      options.onOtherError(errNotice, err);
    }
    announce(title + ".");
  }

  async function removeFolder(rootPath) {
    if (state.removePending || state.addPending) {
      return;
    }
    state.removePending = true;
    state.rootsNotice = null;
    renderFolders();
    renderControls();
    try {
      if (!state.roots || typeof state.roots.revision !== "string") {
        await refreshRoots();
      }
      if (!state.roots || typeof state.roots.revision !== "string") {
        return;
      }
      var response = await api("POST", "/api/roots/remove", {
        path: rootPath,
        revision: state.roots.revision,
      });
      var data = response.data;
      if (!data || !Array.isArray(data.roots)) {
        throw new ApiFailure("protocol", response.status, "protocol", "The server sent an unexpected reply.");
      }
      state.roots = data;
      state.rootsError = null;
      if (state.newRoot === rootPath) {
        state.newRoot = null;
      }
      state.rootsNotice = notice("info", "Folder removed", [
        ["Removed ", el("span", "path-text", rootPath), " from settings and purged its indexed entries."],
      ]);
      announce("Folder removed: " + rootPath + ".");
      renderFolders();
      renderFirstRunPanel();
      renderResultsState();
      refreshStatus();
      refreshCourse();
    } catch (err) {
      if (err.kind === "network") {
        state.rootsNotice = notice("error", "Seekr isn’t responding", [
          "Reconnect, then choose Remove again.",
        ]);
      } else if (err.code === "daemon_active") {
        state.rootsNotice = notice("error", "Stop the older seekr daemon first", [
          err.message || "An older seekr daemon process is running and maintains the index.",
          ["Stop the older ", codeText("seekr daemon"), " process, then choose Refresh and try Remove again. ", codeText("seekr service start"), " replaces it."],
        ]);
      } else if (err.code === "stale_config") {
        state.rootsNotice = notice("error", "Settings file changed", [
          "Your settings file changed since this page loaded. The folder list has been refreshed; choose Remove again.",
        ]);
        refreshRoots();
      } else if (err.code === "busy") {
        state.rootsNotice = notice("error", "Seekr is busy", [
          err.message || "Another task is running. Try removing the folder when it finishes.",
        ]);
      } else if (err.code === "config_invalid") {
        state.rootsNotice = notice("error", "Seekr can’t safely edit your settings file", [
          err.message,
          ["Settings file: ", el("span", "path-text", configPath())],
        ]);
      } else {
        state.rootsNotice = notice("error", "Couldn’t remove the folder", [
          err.message || "The server reported an unexpected problem.",
        ]);
      }
      announce("Couldn’t remove the folder.");
      renderFolders();
    } finally {
      state.removePending = false;
      renderFolders();
      renderControls();
    }
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
    renderFirstRunPanel();
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
    refreshCourse();
  }

  // ---------------------------------------------------------------- keyboard

  function isEditableTarget(target) {
    if (!target || typeof target.tagName !== "string") {
      return false;
    }
    var tag = target.tagName.toUpperCase();
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || Boolean(target.isContentEditable);
  }

  function isArrowBlockedTarget(target) {
    if (!target || typeof target.tagName !== "string") {
      return false;
    }
    var tag = target.tagName.toUpperCase();
    if (tag === "SELECT" || tag === "TEXTAREA" || target.isContentEditable) {
      return true;
    }
    if (tag === "INPUT") {
      return target.id !== "query";
    }
    return false;
  }

  function handleKeyDown(event) {
    var target = event.target;
    if (event.key === "Escape" && state.previewOpen) {
      event.preventDefault();
      closePreview(true);
      return;
    }

    if (event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }

    if (event.key === "/" && !isEditableTarget(target)) {
      event.preventDefault();
      if (state.view !== "search") {
        setView("search", false);
      }
      byId("query").focus();
      return;
    }

    if (state.view !== "search") {
      return;
    }

    if ((event.key === "ArrowDown" || event.key === "ArrowUp") && state.visibleHits.length && !isArrowBlockedTarget(target)) {
      event.preventDefault();
      var delta = event.key === "ArrowDown" ? 1 : -1;
      var nextIndex;
      if (state.activeHitIndex < 0) {
        nextIndex = event.key === "ArrowDown" ? 0 : state.visibleHits.length - 1;
      } else {
        nextIndex = state.activeHitIndex + delta;
      }
      setActiveHit(nextIndex, true);
      return;
    }

    if (event.key === "Enter" && state.activeHitIndex >= 0 && state.activeHitIndex < state.visibleHits.length) {
      var onHitCard = Boolean(target && target.classList && target.classList.contains("hit"));
      var interactive = Boolean(target && target.closest && target.closest("input, textarea, select, button, a, summary, [contenteditable]"));
      if (onHitCard || !interactive) {
        event.preventDefault();
        selectAndPreviewHit(state.activeHitIndex);
        return;
      }
    }

    if ((event.key === "o" || event.key === "O") && !isEditableTarget(target)) {
      var chunkId = state.previewChunkId;
      if (!chunkId && state.activeHitIndex >= 0 && state.activeHitIndex < state.visibleHits.length) {
        chunkId = state.visibleHits[state.activeHitIndex].chunk_id;
      }
      if (typeof chunkId === "number" && isFinite(chunkId) && chunkId > 0) {
        event.preventDefault();
        openChunk(chunkId);
      }
    }
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
    byId("first-run-folder-form").addEventListener("submit", function (event) {
      event.preventDefault();
      addFolderFromFirstRun();
    });
    byId("folder-path").addEventListener("input", function () {
      if (byId("folder-path").getAttribute("aria-invalid") === "true") {
        setFolderError("");
      }
    });
    byId("first-run-folder-path").addEventListener("input", function () {
      if (byId("first-run-folder-path").getAttribute("aria-invalid") === "true") {
        setFieldError("first-run-folder-path", "first-run-folder-error", "");
      }
    });
    byId("download-model-button").addEventListener("click", startDownload);
    byId("first-run-update-index").addEventListener("click", function () {
      startJob("index");
    });
    byId("first-run-dismiss").addEventListener("click", function () {
      state.firstRunDismissed = true;
      state.lastDownload = null;
      renderFirstRunPanel();
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
    var rerunCourseSearch = function () {
      if (state.searchShown && byId("query").value.trim()) {
        runSearch();
      }
    };
    byId("module-filter").addEventListener("change", function () {
      renderWeekOptions();
      rerunCourseSearch();
    });
    ["week-from", "week-to"].forEach(function (id) {
      byId(id).addEventListener("change", function () {
        renderFilterCount();
        rerunCourseSearch();
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
    byId("preview-close").addEventListener("click", function () {
      closePreview(true);
    });
    byId("open-button").addEventListener("click", function () {
      if (state.previewChunkId) {
        openChunk(state.previewChunkId);
      }
    });
    byId("copy-location").addEventListener("click", function () {
      var loc = activePreviewLocation();
      if (loc) {
        copyLocation(loc, byId("preview-copy-fallback"), byId("preview-feedback"), "preview-copy-field");
      }
    });
    byId("posix-toggle").addEventListener("change", function () {
      if (state.previewOpen) {
        updatePreviewLocationDisplay();
      }
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
    document.addEventListener("keydown", handleKeyDown);
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
    window.addEventListener("pagehide", revokePreviewObjectUrl);
    window.addEventListener("beforeunload", revokePreviewObjectUrl);

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

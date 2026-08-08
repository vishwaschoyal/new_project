/* AI Coding Workspace — browser client.
 *
 * Three responsibilities: talk to the JSON API, consume the SSE agent stream,
 * and render model output safely.
 *
 * On that last point: model output is untrusted input. It is Markdown from a
 * language model that has just been reading an arbitrary repository, so it can
 * contain anything a file in that repository contains. It is therefore parsed
 * with Marked and *always* passed through DOMPurify before it touches the DOM.
 * If either library failed to load, rendering falls back to escaped plain text
 * rather than quietly inserting raw HTML.
 */
(() => {
  "use strict";

  // ── state ────────────────────────────────────────────────────────────
  const state = {
    threadId: null,
    mode: "ask",
    repo: null,
    running: false,
    controller: null,
    usage: { input: 0, cached: 0, output: 0, cost: 0 },
    currentPath: "",
  };

  const THREAD_KEY = "acw.threads";
  const ACTIVE_KEY = "acw.activeThread";

  const $ = (id) => document.getElementById(id);
  const el = {
    messages: $("messages"), empty: $("empty-state"), prompt: $("prompt"),
    send: $("send"), stop: $("stop"), composer: $("composer"),
    githubInput: $("github-input"), browseProfile: $("browse-profile"),
    loadRepo: $("load-repo"), repoList: $("repo-list"), repoPicker: $("repo-picker"),
    repoStatus: $("repo-status"), repoLink: $("repo-link"), repoMeta: $("repo-meta"),
    unloadRepo: $("unload-repo"), branchSelect: $("branch-select"),
    filesPanel: $("files-panel"), fileTree: $("file-tree"), breadcrumb: $("breadcrumb"),
    threadList: $("thread-list"), newThread: $("new-thread"),
    modal: $("modal"), modalTitle: $("modal-title"), modalBody: $("modal-body"),
    modalFoot: $("modal-foot"), modalClose: $("modal-close"),
    toast: $("toast"), modelBadge: $("model-badge"),
    taskBar: $("task-bar"), taskBranch: $("task-branch"), taskFiles: $("task-files"),
    reviewDiff: $("review-diff"), discardTask: $("discard-task"),
    uInput: $("u-input"), uCached: $("u-cached"), uOutput: $("u-output"),
    uCost: $("u-cost"), cacheFill: $("cache-fill"), cacheLabel: $("cache-label"),
  };

  // ── safe rendering ───────────────────────────────────────────────────
  const escapeHtml = (text) =>
    String(text).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const librariesReady = () =>
    typeof window.marked !== "undefined" && typeof window.DOMPurify !== "undefined";

  function renderMarkdown(text) {
    if (!librariesReady()) {
      return `<pre class="plain">${escapeHtml(text)}</pre>`;
    }
    const html = window.marked.parse(String(text ?? ""), { breaks: true, gfm: true });
    return window.DOMPurify.sanitize(html, {
      ALLOWED_TAGS: ["p", "br", "strong", "em", "code", "pre", "ul", "ol", "li", "a",
                     "h1", "h2", "h3", "h4", "blockquote", "table", "thead", "tbody",
                     "tr", "th", "td", "hr", "del", "span"],
      ALLOWED_ATTR: ["href", "title", "class"],
      // Model text can contain a javascript: or data: URI lifted from a repo.
      ALLOWED_URI_REGEXP: /^(?:https?:|mailto:|#)/i,
    });
  }

  function highlight(scope) {
    if (typeof window.hljs === "undefined") return;
    scope.querySelectorAll("pre code").forEach((block) => {
      try { window.hljs.highlightElement(block); } catch { /* non-fatal */ }
    });
  }

  /* Turn `path/to/file.py:120-140` inside rendered output into a link that
   * opens the file viewer at those lines. Runs on the sanitized DOM, over text
   * nodes only, so it cannot reintroduce markup. */
  const CITATION_RE = /\b([A-Za-z0-9_./\-]+\.[A-Za-z0-9_]+):(\d+)(?:-(\d+))?\b/g;

  function linkifyCitations(scope) {
    const walker = document.createTreeWalker(scope, NodeFilter.SHOW_TEXT);
    const targets = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (node.parentElement.closest("a, pre")) continue;
      if (CITATION_RE.test(node.nodeValue)) targets.push(node);
      CITATION_RE.lastIndex = 0;
    }
    targets.forEach((node) => {
      const fragment = document.createDocumentFragment();
      let lastIndex = 0;
      node.nodeValue.replace(CITATION_RE, (match, path, start, end, offset) => {
        fragment.append(node.nodeValue.slice(lastIndex, offset));
        const link = document.createElement("a");
        link.className = "citation";
        link.textContent = match;
        link.href = "#";
        link.addEventListener("click", (event) => {
          event.preventDefault();
          openFile(path, Number(start), Number(end || start));
        });
        fragment.append(link);
        lastIndex = offset + match.length;
        return match;
      });
      fragment.append(node.nodeValue.slice(lastIndex));
      node.parentNode.replaceChild(fragment, node);
    });
  }

  // ── http ─────────────────────────────────────────────────────────────
  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  }

  function toast(message, kind = "info") {
    el.toast.textContent = message;
    el.toast.className = `toast toast-${kind}`;
    el.toast.hidden = false;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => { el.toast.hidden = true; }, 4200);
  }

  // ── threads ──────────────────────────────────────────────────────────
  const newThreadId = () =>
    "t" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

  function readThreads() {
    try { return JSON.parse(localStorage.getItem(THREAD_KEY) || "[]"); }
    catch { return []; }
  }

  function rememberThread(id, title) {
    const threads = readThreads().filter((t) => t.id !== id);
    threads.unshift({ id, title: title.slice(0, 60), at: Date.now() });
    localStorage.setItem(THREAD_KEY, JSON.stringify(threads.slice(0, 12)));
    localStorage.setItem(ACTIVE_KEY, id);
    renderThreads();
  }

  function renderThreads() {
    const threads = readThreads();
    el.threadList.innerHTML = "";
    threads.forEach((thread) => {
      const item = document.createElement("li");
      item.className = thread.id === state.threadId ? "active" : "";
      item.textContent = thread.title || thread.id;
      item.title = thread.title || thread.id;
      item.addEventListener("click", () => openThread(thread.id));
      el.threadList.append(item);
    });
  }

  async function openThread(id) {
    state.threadId = id;
    localStorage.setItem(ACTIVE_KEY, id);
    el.messages.innerHTML = "";
    resetUsage();

    try {
      const { messages } = await api(`/api/history?thread_id=${encodeURIComponent(id)}`);
      messages.forEach((message) => {
        const node = addMessage(message.role, message.content);
        if (message.metadata?.usage) applyUsage(message.metadata.usage, false);
        if (message.role === "assistant") finishMessage(node);
      });
      if (!messages.length) showEmptyState();
    } catch { showEmptyState(); }

    await refreshRepo();
    await refreshTask();
    renderThreads();
  }

  function showEmptyState() {
    if (!el.messages.querySelector(".msg")) {
      el.messages.innerHTML = "";
      el.messages.append(el.empty);
      el.empty.hidden = false;
    }
  }

  // ── repository ───────────────────────────────────────────────────────
  async function browseProfile() {
    const url = el.githubInput.value.trim();
    if (!url) return toast("Enter a GitHub profile or repository URL.", "warn");

    el.browseProfile.disabled = true;
    try {
      const { profile, repositories } = await api("/api/repo/profile", {
        method: "POST", body: JSON.stringify({ url }),
      });
      el.repoList.innerHTML = "";
      el.repoList.hidden = false;
      repositories.forEach((repo) => {
        const item = document.createElement("li");
        item.innerHTML = `<strong>${escapeHtml(repo.name)}</strong>
          <span class="repo-lang">${escapeHtml(repo.language || "")}</span>
          <span class="repo-stars">★ ${repo.stars}</span>`;
        item.addEventListener("click", () => loadRepository(repo.html_url));
        el.repoList.append(item);
      });
      toast(`${repositories.length} repositories for ${profile.login}`);
    } catch (error) {
      toast(error.message, "error");
    } finally {
      el.browseProfile.disabled = false;
    }
  }

  async function loadRepository(url) {
    const target = url || el.githubInput.value.trim();
    if (!target) return toast("Enter a repository URL.", "warn");
    if (!state.threadId) state.threadId = newThreadId();

    el.loadRepo.disabled = true;
    el.loadRepo.textContent = "Cloning…";
    try {
      const { workspace } = await api("/api/repo/load", {
        method: "POST",
        body: JSON.stringify({ thread_id: state.threadId, url: target }),
      });
      applyWorkspace(workspace);
      await Promise.all([loadBranches(), loadTree("")]);
      toast(`Loaded ${workspace.repo}`);
    } catch (error) {
      toast(error.message, "error");
    } finally {
      el.loadRepo.disabled = false;
      el.loadRepo.textContent = "Load repo";
    }
  }

  function applyWorkspace(workspace) {
    state.repo = workspace;
    el.repoPicker.hidden = true;
    el.repoList.hidden = true;
    el.repoStatus.hidden = false;
    el.filesPanel.hidden = false;
    el.repoLink.textContent = workspace.repo;
    el.repoLink.href = workspace.html_url;
    el.repoMeta.textContent = `${workspace.head_sha} · default ${workspace.default_branch}`;
    el.send.disabled = false;
  }

  async function refreshRepo() {
    if (!state.threadId) return;
    try {
      const status = await api(`/api/repo/status?thread_id=${encodeURIComponent(state.threadId)}`);
      if (status.loaded) {
        applyWorkspace(status);
        await Promise.all([loadBranches(), loadTree("")]);
      } else {
        clearRepo();
      }
    } catch { clearRepo(); }
  }

  function clearRepo() {
    state.repo = null;
    el.repoPicker.hidden = false;
    el.repoStatus.hidden = true;
    el.filesPanel.hidden = true;
    el.send.disabled = true;
  }

  async function loadBranches() {
    try {
      const branches = await api(`/api/repo/branches?thread_id=${encodeURIComponent(state.threadId)}`);
      el.branchSelect.innerHTML = "";
      branches.all.forEach((branch) => {
        const option = document.createElement("option");
        option.value = branch;
        option.textContent = branch;
        option.selected = branch === branches.current;
        el.branchSelect.append(option);
      });
    } catch { /* a repo with one branch is fine */ }
  }

  async function switchBranch(branch) {
    try {
      const { workspace } = await api("/api/repo/branch", {
        method: "POST",
        body: JSON.stringify({ thread_id: state.threadId, branch }),
      });
      applyWorkspace(workspace);
      await loadTree("");
      toast(`Switched to ${branch}`);
    } catch (error) {
      toast(error.message, "error");
      await loadBranches();
    }
  }

  async function loadTree(path) {
    try {
      const tree = await api(
        `/api/repo/tree?thread_id=${encodeURIComponent(state.threadId)}&path=${encodeURIComponent(path)}`);
      state.currentPath = tree.path;
      renderBreadcrumb(tree.path);

      el.fileTree.innerHTML = "";
      if (tree.path) {
        const up = document.createElement("li");
        up.className = "dir";
        up.textContent = "‥";
        up.addEventListener("click", () =>
          loadTree(tree.path.split("/").slice(0, -1).join("/")));
        el.fileTree.append(up);
      }
      tree.entries.forEach((entry) => {
        const item = document.createElement("li");
        item.className = entry.type === "dir" ? "dir" : "file";
        item.textContent = entry.name;
        item.title = entry.path;
        item.addEventListener("click", () =>
          entry.type === "dir" ? loadTree(entry.path) : openFile(entry.path));
        el.fileTree.append(item);
      });
    } catch (error) {
      toast(error.message, "error");
    }
  }

  function renderBreadcrumb(path) {
    el.breadcrumb.innerHTML = "";
    const root = document.createElement("span");
    root.textContent = state.repo ? state.repo.repo.split("/")[1] : "root";
    root.addEventListener("click", () => loadTree(""));
    el.breadcrumb.append(root);
    let accumulated = "";
    (path ? path.split("/") : []).forEach((part) => {
      accumulated = accumulated ? `${accumulated}/${part}` : part;
      const crumb = document.createElement("span");
      crumb.textContent = part;
      const target = accumulated;
      crumb.addEventListener("click", () => loadTree(target));
      el.breadcrumb.append(crumb);
    });
  }

  async function openFile(path, startLine, endLine) {
    try {
      const file = await api(
        `/api/repo/file?thread_id=${encodeURIComponent(state.threadId)}&path=${encodeURIComponent(path)}`);

      const lines = file.content.split("\n");
      const numbered = lines.map((line, index) => {
        const number = index + 1;
        const hit = startLine && number >= startLine && number <= (endLine || startLine);
        return `<tr class="${hit ? "hit" : ""}" id="L${number}">
                  <td class="ln">${number}</td><td class="lc">${escapeHtml(line) || " "}</td>
                </tr>`;
      }).join("");

      openModal(path, `<table class="code">${numbered}</table>`);
      if (startLine) {
        requestAnimationFrame(() => {
          const row = el.modalBody.querySelector(`#L${startLine}`);
          if (row) row.scrollIntoView({ block: "center" });
        });
      }
    } catch (error) {
      toast(error.message, "error");
    }
  }

  // ── modal ────────────────────────────────────────────────────────────
  function openModal(title, html, footerNodes) {
    el.modalTitle.textContent = title;
    el.modalBody.innerHTML = html;
    el.modalFoot.innerHTML = "";
    if (footerNodes?.length) {
      footerNodes.forEach((node) => el.modalFoot.append(node));
      el.modalFoot.hidden = false;
    } else {
      el.modalFoot.hidden = true;
    }
    el.modal.hidden = false;
  }

  const closeModal = () => { el.modal.hidden = true; };

  // ── messages ─────────────────────────────────────────────────────────
  function addMessage(role, text = "") {
    el.empty.remove?.();
    const wrapper = document.createElement("div");
    wrapper.className = `msg msg-${role}`;

    const body = document.createElement("div");
    body.className = "msg-body";
    if (role === "user") {
      body.textContent = text;
    } else {
      body.dataset.raw = text;
      body.innerHTML = renderMarkdown(text);
    }
    wrapper.append(body);
    el.messages.append(wrapper);
    scrollToBottom();
    return wrapper;
  }

  function appendDelta(node, delta) {
    const body = node.querySelector(".msg-body");
    body.dataset.raw = (body.dataset.raw || "") + delta;
    // Re-render each delta: Markdown is not incrementally parseable, and a
    // half-written code fence rendered as text then reflowed looks broken.
    body.innerHTML = renderMarkdown(body.dataset.raw);
    scrollToBottom();
  }

  function finishMessage(node) {
    const body = node.querySelector(".msg-body");
    body.innerHTML = renderMarkdown(body.dataset.raw || body.textContent);
    highlight(body);
    linkifyCitations(body);
  }

  function scrollToBottom() {
    el.messages.scrollTop = el.messages.scrollHeight;
  }

  // ── tool trail ───────────────────────────────────────────────────────
  function createTrail(node) {
    const trail = document.createElement("details");
    trail.className = "trail";
    trail.open = true;
    trail.innerHTML = `<summary><span class="trail-count">0</span> steps</summary><ol></ol>`;
    node.prepend(trail);
    return trail;
  }

  function trailAdd(trail, label, kind = "") {
    const list = trail.querySelector("ol");
    const item = document.createElement("li");
    item.className = kind;
    item.innerHTML = label;
    list.append(item);
    trail.querySelector(".trail-count").textContent = list.children.length;
    scrollToBottom();
    return item;
  }

  const TOOL_ICON = {
    grep: "⌕", glob: "▤", read: "▤", bash: "$", edit: "✎",
    run_check: "▶", record_finding: "✓", add_question: "?", delegate: "⑃",
  };

  // ── usage ────────────────────────────────────────────────────────────
  function resetUsage() {
    state.usage = { input: 0, cached: 0, output: 0, cost: 0 };
    renderUsage();
  }

  function applyUsage(usage, replace = true) {
    if (replace) {
      state.usage = {
        input: usage.input_tokens || 0,
        cached: usage.cached_input_tokens || 0,
        output: usage.output_tokens || 0,
        cost: usage.cost_usd || 0,
      };
    } else {
      state.usage.input += usage.input_tokens || 0;
      state.usage.cached += usage.cached_input_tokens || 0;
      state.usage.output += usage.output_tokens || 0;
      state.usage.cost += usage.cost_usd || 0;
    }
    renderUsage();
  }

  function renderUsage() {
    const { input, cached, output, cost } = state.usage;
    el.uInput.textContent = input.toLocaleString();
    el.uCached.textContent = cached.toLocaleString();
    el.uOutput.textContent = output.toLocaleString();
    el.uCost.textContent = cost < 0.01 && cost > 0 ? `$${cost.toFixed(4)}` : `$${cost.toFixed(2)}`;
    const rate = input ? Math.round((cached / input) * 100) : 0;
    el.cacheFill.style.width = `${rate}%`;
    el.cacheLabel.textContent = input ? `${rate}% of input served from cache` : "no requests yet";
  }

  // ── the run ──────────────────────────────────────────────────────────
  async function send(question) {
    if (state.running) return;
    if (!state.repo) return toast("Load a repository first.", "warn");

    state.running = true;
    el.send.disabled = true;
    el.stop.hidden = false;
    el.prompt.value = "";
    el.prompt.style.height = "auto";

    addMessage("user", question);
    rememberThread(state.threadId, question);

    const node = addMessage("assistant", "");
    const trail = createTrail(node);
    const pending = new Map();
    let sawContent = false;

    state.controller = new AbortController();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: state.threadId, message: question, mode: state.mode }),
        signal: state.controller.signal,
      });

      if (!response.ok || !response.body) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || `Request failed (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line; a frame may split across
        // chunks, so keep the tail until its terminator arrives.
        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";

        for (const frame of frames) {
          if (!frame.trim() || frame.startsWith(":")) continue;
          let type = "message";
          let raw = "";
          frame.split("\n").forEach((line) => {
            if (line.startsWith("event:")) type = line.slice(6).trim();
            else if (line.startsWith("data:")) raw += line.slice(5).trim();
          });
          if (!raw) continue;
          let data;
          try { data = JSON.parse(raw); } catch { continue; }
          handleEvent(type, data, { node, trail, pending, onContent: () => { sawContent = true; } });
        }
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        trailAdd(trail, `<span class="bad">✕ ${escapeHtml(error.message)}</span>`, "bad");
        toast(error.message, "error");
      }
    } finally {
      state.running = false;
      state.controller = null;
      el.send.disabled = false;
      el.stop.hidden = true;
      finishMessage(node);
      if (!sawContent && !node.querySelector(".msg-body").dataset.raw) {
        node.querySelector(".msg-body").textContent = "(no answer was produced)";
      }
      refreshTask();
    }
  }

  function handleEvent(type, data, ctx) {
    const { node, trail, pending } = ctx;

    switch (type) {
      case "tool_start": {
        const icon = TOOL_ICON[data.tool] || "•";
        const args = escapeHtml(summariseArgs(data.tool, data.arguments));
        const item = trailAdd(trail, `<span class="spin">${icon}</span> <b>${escapeHtml(data.tool)}</b> ${args}`, "running");
        pending.set(`${data.tool}:${data.step}`, item);
        break;
      }
      case "tool_end": {
        const item = pending.get(`${data.tool}:${data.step}`);
        const icon = TOOL_ICON[data.name] || "•";
        const mark = data.ok ? "✓" : "✕";
        const html = `<span class="${data.ok ? "good" : "bad"}">${mark}</span>
          <b>${escapeHtml(data.name)}</b> ${escapeHtml(data.summary)}
          <span class="ms">${Math.round(data.duration_ms)}ms</span>`;
        if (item) { item.innerHTML = html; item.className = data.ok ? "" : "bad"; }
        else trailAdd(trail, html, data.ok ? "" : "bad");
        break;
      }
      case "finding":
        trailAdd(trail, `<span class="good">✓</span> ${escapeHtml(data.claim)}
                 <code>${escapeHtml(data.reference)}</code>`, "finding");
        break;
      case "subagents_start":
        trailAdd(trail, `<span class="fan">⑃</span> delegating to ${data.count} workers:
                 ${escapeHtml((data.names || []).join(", "))}`, "fan");
        break;
      case "subagent_tool_end":
        trailAdd(trail, `<span class="worker">${escapeHtml(data.worker)}</span>
                 ${escapeHtml(data.name)} — ${escapeHtml(data.summary || "")}`, "sub");
        break;
      case "subagent_finding":
        trailAdd(trail, `<span class="worker">${escapeHtml(data.worker)}</span>
                 ${escapeHtml(data.claim)} <code>${escapeHtml(data.reference)}</code>`, "sub finding");
        break;
      case "subagents_end":
        trailAdd(trail, `<span class="fan">⑃</span> merged ${data.evidence_merged} findings
                 from ${data.count} workers`, "fan");
        break;
      case "compaction":
        trailAdd(trail, `<span class="dim">⌫ compacted ${data.observations} observations</span>`, "dim");
        break;
      case "challenge":
        trailAdd(trail, `<span class="warn">↻ not finished — parts of the question remain open</span>`, "warn");
        break;
      case "content":
        ctx.onContent();
        appendDelta(node, data.delta);
        break;
      case "usage":
        applyUsage(data);
        break;
      case "done":
        trail.open = false;
        if (data.unsupported_citations?.length) {
          trailAdd(trail, `<span class="warn">⚠ ${data.unsupported_citations.length}
                   unverified citation(s) removed</span>`, "warn");
        }
        renderDoneFooter(node, data);
        break;
      case "error":
        trailAdd(trail, `<span class="bad">✕ ${escapeHtml(data.message)}</span>`, "bad");
        break;
    }
  }

  function summariseArgs(tool, args = {}) {
    if (tool === "grep") return `/${args.pattern || ""}/${args.glob ? ` in ${args.glob}` : ""}`;
    if (tool === "glob") return args.pattern || "";
    if (tool === "read") return `${args.path || ""}${args.offset ? `:${args.offset}` : ""}`;
    if (tool === "bash" || tool === "run_check") return args.command || "";
    if (tool === "edit") return args.path || "";
    if (tool === "record_finding") return args.claim || "";
    if (tool === "delegate") return `${(args.tasks || []).length} tasks`;
    return "";
  }

  function renderDoneFooter(node, data) {
    const footer = document.createElement("div");
    footer.className = "msg-foot";
    const usage = data.usage || {};
    const cacheRate = usage.input_tokens
      ? Math.round((usage.cached_input_tokens / usage.input_tokens) * 100) : 0;

    footer.innerHTML = `
      <span title="Why the run stopped">${escapeHtml(data.termination_reason)}</span>
      <span>${data.steps_used} steps</span>
      <span>${(usage.input_tokens || 0).toLocaleString()} in · ${cacheRate}% cached</span>
      <span>${(usage.output_tokens || 0).toLocaleString()} out</span>
      <span class="cost">$${(usage.cost_usd || 0).toFixed(4)}</span>
      <span>${data.wall_seconds}s</span>`;
    node.append(footer);
  }

  // ── tasks ────────────────────────────────────────────────────────────
  async function refreshTask() {
    if (!state.threadId) return;
    try {
      const { active, task } = await api(
        `/api/task/status?thread_id=${encodeURIComponent(state.threadId)}`);
      el.taskBar.hidden = !active;
      if (active) {
        el.taskBranch.textContent = task.branch;
        el.taskFiles.textContent =
          `${task.edited_files.length} file(s) · ${task.checks.length} check(s)`;
      }
    } catch { el.taskBar.hidden = true; }
  }

  async function startTask(description) {
    const { task } = await api("/api/task/start", {
      method: "POST",
      body: JSON.stringify({ thread_id: state.threadId, description }),
    });
    toast(`Task branch ${task.branch}`);
    await refreshTask();
    await loadBranches();
  }

  async function reviewDiff() {
    try {
      const review = await api(`/api/task/review?thread_id=${encodeURIComponent(state.threadId)}`);
      const diffHtml = review.diff
        ? `<pre class="diff">${colourDiff(review.diff)}</pre>`
        : `<p class="hint">No changes yet.</p>`;

      const checks = (review.task.checks || []).map((check) =>
        `<li class="${check.ok ? "good" : "bad"}">${check.ok ? "✓" : "✕"}
          <code>${escapeHtml(check.command)}</code></li>`).join("");

      const buttons = ["commit", "push", "pull_request"].map((action) => {
        const button = document.createElement("button");
        button.className = action === "pull_request" ? "btn btn-primary" : "btn";
        button.textContent = { commit: "Commit", push: "Commit & push",
                               pull_request: "Open pull request" }[action];
        button.disabled = !review.has_changes;
        button.addEventListener("click", () => approve(action));
        return button;
      });

      openModal(
        `Review — ${review.task.branch}`,
        `${checks ? `<ul class="checks">${checks}</ul>` : ""}
         <pre class="stat">${escapeHtml(review.diff_stat || "")}</pre>${diffHtml}`,
        buttons);
    } catch (error) {
      toast(error.message, "error");
    }
  }

  const colourDiff = (diff) => diff.split("\n").map((line) => {
    const cls = line.startsWith("+++") || line.startsWith("---") ? "meta"
      : line.startsWith("+") ? "add"
      : line.startsWith("-") ? "del"
      : line.startsWith("@@") ? "hunk" : "";
    return `<span class="${cls}">${escapeHtml(line)}</span>`;
  }).join("\n");

  async function approve(action) {
    const label = { commit: "commit", push: "commit and push",
                    pull_request: "open a pull request" }[action];
    if (!confirm(`Approve: ${label}?\n\nThis publishes the reviewed change.`)) return;

    try {
      const outcome = await api("/api/task/approve", {
        method: "POST",
        body: JSON.stringify({ thread_id: state.threadId, action, confirmed: true }),
      });
      closeModal();
      toast(outcome.pull_request_url
        ? `Pull request opened: ${outcome.pull_request_url}`
        : `Done: ${action}`, "success");
      if (outcome.pull_request_url) window.open(outcome.pull_request_url, "_blank", "noopener");
      await refreshTask();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  // ── wiring ───────────────────────────────────────────────────────────
  el.composer.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = el.prompt.value.trim();
    if (!question) return;

    // A coding run needs a task branch; create one from the first instruction.
    if (state.mode === "code") {
      try {
        const { active } = await api(
          `/api/task/status?thread_id=${encodeURIComponent(state.threadId)}`);
        if (!active) await startTask(question);
      } catch (error) { return toast(error.message, "error"); }
    }
    send(question);
  });

  el.prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el.composer.requestSubmit();
    }
  });

  el.prompt.addEventListener("input", () => {
    el.prompt.style.height = "auto";
    el.prompt.style.height = `${Math.min(el.prompt.scrollHeight, 180)}px`;
  });

  document.querySelectorAll(".mode").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".mode").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      state.mode = button.dataset.mode;
      el.prompt.placeholder = state.mode === "code"
        ? "Describe the change to make…"
        : "Ask a question about the repository…";
    });
  });

  el.stop.addEventListener("click", () => state.controller?.abort());
  el.browseProfile.addEventListener("click", browseProfile);
  el.loadRepo.addEventListener("click", () => loadRepository());
  el.githubInput.addEventListener("keydown", (e) => { if (e.key === "Enter") loadRepository(); });
  el.branchSelect.addEventListener("change", (e) => switchBranch(e.target.value));
  el.unloadRepo.addEventListener("click", async () => {
    await api("/api/repo/unload", {
      method: "POST", body: JSON.stringify({ thread_id: state.threadId }),
    }).catch(() => {});
    clearRepo();
    toast("Repository unloaded");
  });
  el.newThread.addEventListener("click", () => openThread(newThreadId()));
  el.reviewDiff.addEventListener("click", reviewDiff);
  el.discardTask.addEventListener("click", async () => {
    if (!confirm("Discard this task and all its uncommitted changes?")) return;
    await api("/api/task/discard", {
      method: "POST", body: JSON.stringify({ thread_id: state.threadId }),
    }).catch((error) => toast(error.message, "error"));
    await refreshTask();
    await loadBranches();
  });
  el.modalClose.addEventListener("click", closeModal);
  el.modal.addEventListener("click", (e) => { if (e.target === el.modal) closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

  document.querySelectorAll(".examples li").forEach((item) => {
    item.addEventListener("click", () => {
      el.prompt.value = item.dataset.example;
      el.prompt.focus();
    });
  });

  // ── boot ─────────────────────────────────────────────────────────────
  (async function boot() {
    try {
      const config = await api("/api/chat/config");
      el.modelBadge.textContent = config.model_configured
        ? `${config.model} · sandbox: ${config.sandbox_backend}${config.sandbox_isolated ? "" : " (not isolated)"}`
        : "no API key configured";
      if (!config.model_configured) toast("OPENAI_API_KEY is not set — runs will fail.", "warn");
    } catch {
      el.modelBadge.textContent = "offline";
    }
    if (!librariesReady()) {
      toast("Markdown libraries did not load; answers render as plain text.", "warn");
    }
    await openThread(localStorage.getItem(ACTIVE_KEY) || newThreadId());
  })();
})();

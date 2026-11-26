/* MedAugmentX Policy Studio front-end. No external dependencies. */
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const ROOT_TEMPLATE = { name: "Compose", params: { transforms: [], p: 1.0, seed: null } };
  const HISTORY_LIMIT = 100;

  const S = {
    token: null,
    catalog: null,
    byName: {},
    policy: structuredClone(ROOT_TEMPLATE),
    selected: null,
    validity: { ok: true },
    data: { loaded: false },
    axis: null,
    history: [],
    future: [],
    preview: null,
    zoom: 1,
    pan: { x: 0, y: 0 },
    rawFields: new Set(),
    previewSeq: 0,
    timers: {},
    exportFormat: "json",
    palette: null,
  };

  // ------------------------------------------------------------- utilities
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") el.className = value;
      else if (key === "text") el.textContent = value;
      else if (key === "dataset") Object.assign(el.dataset, value);
      // CSSOM writes are allowed by the strict CSP; style attributes are not.
      else if (key === "style") el.style.cssText = value;
      else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
      else if (value === true) el.setAttribute(key, "");
      else el.setAttribute(key, String(value));
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
  }

  function debounce(name, ms, fn) {
    clearTimeout(S.timers[name]);
    S.timers[name] = setTimeout(fn, ms);
  }

  function toast(message, isError = false) {
    const el = $("toast");
    el.textContent = message;
    el.className = "toast" + (isError ? " error" : "");
    el.hidden = false;
    debounce("toast", isError ? 6000 : 2600, () => { el.hidden = true; });
  }

  function fmt(value, digits = 3) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "number") {
      if (!Number.isFinite(value)) return String(value);
      if (Number.isInteger(value)) return String(value);
      const abs = Math.abs(value);
      return abs !== 0 && (abs < 1e-3 || abs >= 1e5) ? value.toExponential(2) : value.toFixed(digits).replace(/\.?0+$/, "");
    }
    return JSON.stringify(value);
  }

  function compact(value) {
    if (Array.isArray(value)) return "[" + value.map(compact).join(", ") + "]";
    if (value && typeof value === "object") return "{…}";
    return fmt(value);
  }

  function download(filename, text, type = "application/json") {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const link = h("a", { href: url, download: filename });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function catClass(category) {
    return "cat-" + String(category || "Custom").split(" ")[0];
  }

  function equalJSON(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
  }

  function isTyping(event) {
    const t = event.target;
    return t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement || t.isContentEditable;
  }

  // ------------------------------------------------------------------- API
  async function api(path, body) {
    const options = { method: body === undefined ? "GET" : "POST", headers: { "X-Studio-Token": S.token } };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);
    let data;
    try { data = await response.json(); } catch { data = { error: response.statusText }; }
    if (!response.ok) {
      const error = new Error(data.error || response.statusText || "Request failed");
      error.path = data.path;
      error.status = response.status;
      throw error;
    }
    return data;
  }

  async function upload(role, file, extra = {}) {
    const query = new URLSearchParams({ role, name: file.name, ...extra });
    const response = await fetch(`/api/upload?${query}`, {
      method: "POST",
      headers: { "X-Studio-Token": S.token, "Content-Type": "application/octet-stream" },
      body: file,
    });
    const data = await response.json().catch(() => ({ error: response.statusText }));
    if (!response.ok) throw new Error(data.error || "Upload failed");
    return data;
  }

  // ------------------------------------------------------------ tree model
  function nestKey(node) {
    return node && S.byName[node.name] ? S.byName[node.name].container : null;
  }

  function getNode(path) {
    let node = S.policy;
    for (const token of path || []) {
      if (!node || !node.params) return null;
      node = typeof token === "number" ? (node.params.transforms || [])[token] : node.params.transform;
    }
    return node || null;
  }

  function setNode(path, value) {
    if (!path.length) { S.policy = value; return; }
    const parent = getNode(path.slice(0, -1));
    const last = path[path.length - 1];
    if (typeof last === "number") parent.params.transforms[last] = value;
    else parent.params.transform = value;
  }

  const pathKey = (path) => JSON.stringify(path || []);
  const samePath = (a, b) => a !== null && b !== null && pathKey(a) === pathKey(b);

  // OneOf keeps one weight per child; every list edit must keep them aligned.
  function listWeights(container) {
    const weights = container.name === "OneOf" ? container.params.weights : null;
    return Array.isArray(weights) && weights.length === (container.params.transforms || []).length ? weights : null;
  }

  function insertChildren(container, index, nodes, weights) {
    container.params.transforms = container.params.transforms || [];
    const existing = listWeights(container);
    container.params.transforms.splice(index, 0, ...nodes);
    if (existing) {
      const fill = existing.length ? existing.reduce((a, b) => a + b, 0) / existing.length : 1;
      existing.splice(index, 0, ...nodes.map((_, i) => (weights && weights[i] !== undefined ? weights[i] : fill)));
    } else if (container.name === "OneOf") {
      container.params.weights = null;
    }
  }

  function removeChild(container, index) {
    const weights = listWeights(container);
    const [removed] = container.params.transforms.splice(index, 1);
    const weight = weights ? weights.splice(index, 1)[0] : undefined;
    if (weights && weights.length && !weights.some((w) => w > 0)) container.params.weights = null;
    return { node: removed, weight };
  }

  function swapChildren(container, a, b) {
    const list = container.params.transforms;
    [list[a], list[b]] = [list[b], list[a]];
    const weights = listWeights(container);
    if (weights) [weights[a], weights[b]] = [weights[b], weights[a]];
  }

  function commit(mutator, { select, deferInspector = false } = {}) {
    S.history.push(JSON.stringify(S.policy));
    if (S.history.length > HISTORY_LIMIT) S.history.shift();
    S.future = [];
    mutator();
    if (select !== undefined) S.selected = select;
    if (S.selected && !getNode(S.selected)) S.selected = null;
    afterPolicyChange({ deferInspector });
  }

  function undo() {
    if (!S.history.length) return;
    S.future.push(JSON.stringify(S.policy));
    S.policy = JSON.parse(S.history.pop());
    if (S.selected && !getNode(S.selected)) S.selected = null;
    afterPolicyChange();
  }

  function redo() {
    if (!S.future.length) return;
    S.history.push(JSON.stringify(S.policy));
    S.policy = JSON.parse(S.future.pop());
    if (S.selected && !getNode(S.selected)) S.selected = null;
    afterPolicyChange();
  }

  function afterPolicyChange({ deferInspector = false } = {}) {
    $("btn-undo").disabled = !S.history.length;
    $("btn-redo").disabled = !S.future.length;
    // Any preview still in flight was computed for the previous policy.
    S.previewSeq += 1;
    $("preview-busy").hidden = true;
    renderTree();
    // Field edits commit on "change", which fires before focus moves on; render
    // after the move so Tab lands on the next field of the new panel.
    if (deferInspector) setTimeout(renderInspector, 0);
    else renderInspector();
    debounce("sync", 250, syncPolicy);
  }

  async function syncPolicy() {
    const seq = (S.syncSeq = (S.syncSeq || 0) + 1);
    let validity;
    try {
      validity = await api("/api/policy", { policy: S.policy });
    } catch (error) {
      validity = { ok: false, error: error.message, path: error.path || [] };
    }
    if (seq !== S.syncSeq) return; // a newer edit has already been sent
    S.validity = validity;
    renderStatus();
    renderTree();
    renderInspector();
    if (S.validity.ok) schedulePreview(0);
    else showPreviewError(S.validity.error);
  }

  function renderStatus() {
    const el = $("policy-status");
    const steps = countSteps(S.policy);
    if (S.validity.ok) {
      el.className = "policy-status ok";
      el.textContent = `Valid · ${steps} step${steps === 1 ? "" : "s"}`;
      el.title = "";
    } else {
      el.className = "policy-status bad";
      el.textContent = "Invalid policy";
      el.title = S.validity.error || "";
    }
  }

  function countSteps(node) {
    let total = 0;
    const visit = (n) => {
      const key = nestKey(n);
      if (key === "transforms") (n.params.transforms || []).forEach(visit);
      else if (key === "transform") visit(n.params.transform);
      else total += 1;
    };
    visit(node);
    return total;
  }

  // ------------------------------------------------------------ tree view
  function nodeSubtitle(node) {
    const key = nestKey(node);
    if (key === "transforms") {
      const n = (node.params.transforms || []).length;
      const extra = node.name === "SomeOf" ? ` · n=${compact(node.params.n)}` : "";
      return `${n} child${n === 1 ? "" : "ren"}${extra}`;
    }
    const meta = S.byName[node.name];
    const parts = [];
    for (const param of meta ? meta.params : []) {
      if (["p", "seed"].includes(param.name) || !(param.name in node.params)) continue;
      const value = node.params[param.name];
      if (!param.required && equalJSON(value, param.default)) continue;
      parts.push(`${param.name}=${compact(value)}`);
      if (parts.length === 3) break;
    }
    if (key === "transform" && node.params.shared) parts.unshift("shared");
    return parts.join("  ");
  }

  function renderTree() {
    const tree = $("tree");
    tree.replaceChildren();
    const root = S.policy;
    const rootKey = nestKey(root);
    if (rootKey === "transforms" && root.name === "Compose") {
      const children = root.params.transforms || [];
      if (!children.length) tree.append(h("div", { class: "tree-empty" }, "No steps yet. Add a step or start from a preset."));
      children.forEach((child, index) => tree.append(renderNode(child, [index])));
      if (!samePath(S.selected, [])) {
        const rootInfo = h("button", { class: "tree-add", type: "button", onclick: () => select([]) }, "Pipeline settings (root Compose)");
        tree.append(rootInfo);
      }
    } else {
      tree.append(renderNode(root, []));
    }
  }

  function renderNode(node, path) {
    const meta = S.byName[node.name];
    const invalid = !S.validity.ok && samePath(S.validity.path || [], path);
    const p = node.params ? node.params.p : undefined;
    const row = h(
      "div",
      {
        class: "tree-node" + (samePath(S.selected, path) ? " selected" : "") + (invalid ? " invalid" : ""),
        role: "treeitem",
        tabindex: "0",
        "aria-selected": samePath(S.selected, path) ? "true" : "false",
        onclick: (event) => { event.stopPropagation(); select(path); },
        onkeydown: (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(path); } },
      },
      h("span", { class: "dot " + catClass(meta ? meta.category : "Custom") }),
      h("div", { class: "node-main" },
        h("div", { class: "node-name", text: node.name }),
        h("div", { class: "node-sub", text: nodeSubtitle(node) })),
      typeof p === "number" && p < 1 ? h("span", { class: "badge", title: "Probability of applying" }, `p ${fmt(p, 2)}`) : null,
    );
    const key = nestKey(node);
    if (!key) return row;
    const box = h("div", {}, row);
    const children = h("div", { class: "tree-children" });
    if (key === "transforms") {
      (node.params.transforms || []).forEach((child, index) => children.append(renderNode(child, [...path, index])));
      children.append(h("button", { class: "tree-add", type: "button", onclick: () => openPalette({ mode: "append", path }) }, `+ Add to ${node.name}`));
    } else if (node.params.transform) {
      children.append(renderNode(node.params.transform, [...path, "transform"]));
    }
    box.append(children);
    return box;
  }

  function select(path) {
    S.selected = path;
    renderTree();
    renderInspector();
  }

  // ------------------------------------------------------------ inspector
  function renderInspector() {
    const panel = $("inspector");
    // Edits commit on "change", which fires as focus moves to the next field;
    // remember that field so re-rendering does not drop keyboard focus.
    const active = panel.contains(document.activeElement) ? document.activeElement.getAttribute("aria-label") : null;
    panel.replaceChildren();
    if (active) {
      requestAnimationFrame(() => {
        const match = [...panel.querySelectorAll("[aria-label]")].find((el) => el.getAttribute("aria-label") === active);
        if (match) match.focus();
      });
    }
    const body = h("div", { class: "inspector-body" });
    panel.append(body);
    const node = S.selected ? getNode(S.selected) : null;
    if (!node) {
      body.append(h("div", { class: "title-row" }, h("h2", { text: "Pipeline" })));
      if (!S.validity.ok) body.append(h("div", { class: "node-error", text: S.validity.error }));
      body.append(
        h("p", { class: "summary" }, "Select a step to edit its parameters. Steps run top to bottom; containers choose among their children."),
        S.validity.summary ? h("pre", { text: S.validity.summary }) : null,
      );
      return;
    }
    const meta = S.byName[node.name] || { params: [], category: "Custom", summary: "", card: null };
    const path = S.selected;
    body.append(
      h("div", { class: "title-row" }, h("span", { class: "dot " + catClass(meta.category), style: "width:10px;height:10px;border-radius:50%" }), h("h2", { text: node.name })),
      h("p", { class: "summary", text: `${meta.category} · ${meta.summary}` }),
    );
    if (!S.validity.ok && samePath(S.validity.path || [], path)) body.append(h("div", { class: "node-error", text: S.validity.error }));
    body.append(renderNodeActions(node, path));

    const nested = meta.container;
    const known = new Set();
    for (const param of meta.params) {
      known.add(param.name);
      body.append(renderField(node, param, path));
    }
    for (const name of Object.keys(node.params || {})) {
      if (known.has(name) || name === nested) continue;
      body.append(renderField(node, { name, kind: "json", default: null, description: "Parameter not described by this transform's signature.", required: false }, path));
    }
    if (meta.card) body.append(renderCard(meta.card));
  }

  function renderNodeActions(node, path) {
    const actions = h("div", { class: "actions" });
    const parentPath = path.slice(0, -1);
    const last = path[path.length - 1];
    const inList = typeof last === "number";
    const parent = path.length ? getNode(parentPath) : null;
    const siblings = inList ? parent.params.transforms : null;
    const btn = (label, handler, opts = {}) => h("button", { type: "button", class: "btn small" + (opts.danger ? " danger" : ""), disabled: opts.disabled, title: opts.title, onclick: handler }, label);

    if (inList) {
      actions.append(
        btn("↑", () => moveNode(path, -1), { disabled: last === 0, title: "Move up" }),
        btn("↓", () => moveNode(path, 1), { disabled: last === siblings.length - 1, title: "Move down" }),
        btn("Duplicate", () => commit(() => {
          const weights = listWeights(parent);
          insertChildren(parent, last + 1, [structuredClone(node)], weights ? [weights[last]] : null);
        }, { select: [...parentPath, last + 1] })),
      );
    }
    if (path.length) {
      actions.append(btn("Delete", () => deleteNode(path), { danger: true }));
    }
    const key = nestKey(node);
    if (key === "transform") {
      actions.append(btn("Replace child", () => openPalette({ mode: "replace", path: [...path, "transform"] })));
    }
    if (key === "transforms") {
      actions.append(btn("+ Child", () => openPalette({ mode: "append", path })));
    }
    if (path.length && key && canUnwrap(node)) {
      actions.append(btn("Unwrap", () => unwrapNode(path), { title: "Replace this container with its children" }));
    }
    const wrap = h("select", { "aria-label": "Wrap in container", onchange: async (event) => {
      const name = event.target.value;
      event.target.value = "";
      if (name) await wrapNode(path, name);
    } }, h("option", { value: "" }, "Wrap in…"), ...["Compose", "OneOf", "SomeOf", "Guard", "PerChannel"].map((name) => h("option", { value: name }, name)));
    actions.append(wrap);
    return actions;
  }

  function moveNode(path, delta) {
    const parent = getNode(path.slice(0, -1));
    const index = path[path.length - 1];
    const target = index + delta;
    if (target < 0 || target >= parent.params.transforms.length) return;
    commit(() => swapChildren(parent, index, target), { select: [...path.slice(0, -1), target] });
  }

  function deleteNode(path) {
    const last = path[path.length - 1];
    const parentPath = path.slice(0, -1);
    commit(() => {
      if (typeof last === "number") removeChild(getNode(parentPath), last);
      else getNode(parentPath).params.transform = structuredClone(ROOT_TEMPLATE);
    }, { select: parentPath.length || typeof last !== "number" ? parentPath : null });
  }

  function canUnwrap(node) {
    const key = nestKey(node);
    if (key === "transform") return Boolean(node.params.transform);
    return key === "transforms" && (node.params.transforms || []).length > 0;
  }

  function unwrapNode(path) {
    const node = getNode(path);
    const key = nestKey(node);
    const last = path[path.length - 1];
    const children = key === "transform" ? [node.params.transform] : node.params.transforms;
    commit(() => {
      if (typeof last === "number") {
        const parent = getNode(path.slice(0, -1));
        const { weight } = removeChild(parent, last);
        const inherited = weight === undefined ? null : children.map(() => weight / children.length);
        insertChildren(parent, last, structuredClone(children), inherited);
      } else if (children.length === 1) setNode(path, structuredClone(children[0]));
      else setNode(path, { name: "Compose", params: { transforms: structuredClone(children), p: 1.0, seed: null } });
    }, { select: path });
  }

  async function wrapNode(path, name) {
    try {
      const template = await api("/api/step", { name });
      const node = structuredClone(getNode(path));
      if (S.byName[name].container === "transforms") template.params.transforms = [node];
      else template.params.transform = node;
      commit(() => setNode(path, template), { select: path });
    } catch (error) {
      toast(error.message, true);
    }
  }

  function setParam(path, name, value) {
    commit(() => {
      const node = getNode(path);
      node.params = node.params || {};
      node.params[name] = value;
    }, { deferInspector: true });
  }

  function renderField(node, meta, path) {
    const params = node.params || {};
    const has = Object.prototype.hasOwnProperty.call(params, meta.name);
    const value = has ? params[meta.name] : meta.default;
    const fieldId = pathKey(path) + ":" + meta.name;
    const raw = S.rawFields.has(fieldId);
    const isDefault = equalJSON(value, meta.default);
    const head = h("div", { class: "field-head" },
      h("span", { class: "field-name", text: meta.name }),
      meta.required ? h("span", { class: "req", text: "required" }) : null,
      h("span", { class: "spacer" }),
      h("button", { type: "button", title: raw ? "Use form input" : "Edit as JSON (e.g. turn a value into a range)", onclick: () => {
        raw ? S.rawFields.delete(fieldId) : S.rawFields.add(fieldId);
        renderInspector();
      } }, raw ? "form" : "{ }"),
      !meta.required && !isDefault ? h("button", { type: "button", title: `Reset to ${compact(meta.default)}`, onclick: () => setParam(path, meta.name, structuredClone(meta.default)) }, "reset") : null,
    );
    const input = h("div", { class: "field-input" });
    const error = h("div", { class: "field-error" });
    const commitValue = (v) => setParam(path, meta.name, v);

    const jsonInput = () => {
      const text = JSON.stringify(value);
      const multiline = text.length > 48 || (value && typeof value === "object" && !Array.isArray(value));
      const el = multiline
        ? h("textarea", { spellcheck: "false", "aria-label": meta.name })
        : h("input", { type: "text", spellcheck: "false", "aria-label": meta.name });
      el.value = multiline ? JSON.stringify(value, null, 1) : text;
      el.addEventListener("change", () => {
        try {
          const parsed = JSON.parse(el.value);
          el.classList.remove("invalid");
          error.textContent = "";
          commitValue(parsed);
        } catch {
          el.classList.add("invalid");
          error.textContent = "Not valid JSON — use [1, 2], true, null, \"text\" or numbers.";
        }
      });
      return el;
    };

    if (raw) {
      input.append(jsonInput());
    } else if (meta.name === "p" && typeof value === "number") {
      const number = h("input", { type: "number", min: "0", max: "1", step: "0.05", value: String(value), "aria-label": "p" });
      const range = h("input", { type: "range", min: "0", max: "1", step: "0.01", value: String(value), "aria-label": "p slider" });
      range.addEventListener("input", () => { number.value = range.value; });
      range.addEventListener("change", () => commitValue(Number(range.value)));
      number.addEventListener("change", () => {
        const v = Number(number.value);
        if (Number.isFinite(v) && v >= 0 && v <= 1) commitValue(v);
        else number.classList.add("invalid");
      });
      input.append(range, number);
    } else if (meta.name === "seed") {
      const el = h("input", { type: "text", inputmode: "numeric", placeholder: "none (derived from parent)", value: value === null || value === undefined ? "" : String(value), "aria-label": "seed" });
      el.addEventListener("change", () => {
        const text = el.value.trim();
        if (!text) return commitValue(null);
        if (/^\d+$/.test(text) && Number.isSafeInteger(Number(text))) return commitValue(Number(text));
        el.classList.add("invalid");
        error.textContent = /^\d+$/.test(text)
          ? "Seeds above 2^53 cannot be represented exactly in the browser; set them in Python."
          : "Seed must be a non-negative integer or empty.";
      });
      input.append(el);
    } else if (typeof value === "boolean") {
      const el = h("input", { type: "checkbox", "aria-label": meta.name });
      el.checked = value;
      el.addEventListener("change", () => commitValue(el.checked));
      input.append(h("label", { class: "check" }, el, value ? "enabled" : "disabled"));
    } else if (typeof value === "number") {
      const integer = meta.kind === "int" && Number.isInteger(value);
      const el = h("input", { type: "number", step: integer ? "1" : "any", value: String(value), "aria-label": meta.name });
      el.addEventListener("change", () => {
        const v = Number(el.value);
        if (el.value.trim() === "" || !Number.isFinite(v)) { el.classList.add("invalid"); return; }
        commitValue(integer ? Math.round(v) : v);
      });
      input.append(el);
    } else if (typeof value === "string" && meta.choices) {
      const el = h("select", { "aria-label": meta.name }, ...meta.choices.map((c) => h("option", { value: c, selected: c === value }, c)));
      if (!meta.choices.includes(value)) el.append(h("option", { value, selected: true }, value));
      el.addEventListener("change", () => commitValue(el.value));
      input.append(el);
    } else if (typeof value === "string") {
      const el = h("input", { type: "text", value, "aria-label": meta.name, spellcheck: "false" });
      el.addEventListener("change", () => commitValue(el.value));
      input.append(el);
    } else if (Array.isArray(value) && value.length > 0 && value.length <= 6 && value.every((v) => typeof v === "number")) {
      value.forEach((item, index) => {
        const el = h("input", { type: "number", step: "any", value: String(item), "aria-label": `${meta.name}[${index}]` });
        el.addEventListener("change", () => {
          const v = Number(el.value);
          if (el.value.trim() === "" || !Number.isFinite(v)) { el.classList.add("invalid"); return; }
          const next = value.slice();
          next[index] = v;
          commitValue(next);
        });
        input.append(el);
      });
    } else {
      input.append(jsonInput());
    }
    const helpBits = [meta.description, !meta.required ? `Default: ${compact(meta.default)}` : null].filter(Boolean);
    return h("div", { class: "field" }, head, input, error, helpBits.length ? h("div", { class: "field-help", text: helpBits.join(" · ") }) : null);
  }

  function renderCard(card) {
    const list = h("dl");
    for (const [key, value] of Object.entries(card)) {
      list.append(h("dt", { text: key.replace(/_/g, " ") }));
      if (key === "citations" && Array.isArray(value)) {
        list.append(h("dd", {}, ...value.map((url) => h("div", { text: url }))));
      } else {
        list.append(h("dd", { text: typeof value === "string" ? value : JSON.stringify(value) }));
      }
    }
    return h("details", { class: "card model-card" }, h("summary", { text: "Model card" }), list);
  }

  // --------------------------------------------------------------- palette
  function openPalette(target) {
    S.palette = target;
    const title = target.mode === "replace" ? "Replace child" : "Add step";
    $("palette-title").textContent = title;
    $("palette-search").value = "";
    renderPalette("");
    $("palette").showModal();
    $("palette-search").focus();
  }

  function renderPalette(query) {
    const list = $("palette-list");
    list.replaceChildren();
    const q = query.trim().toLowerCase();
    let first = true;
    for (const category of S.catalog.categories) {
      const items = S.catalog.transforms.filter((t) => t.category === category && (!q || t.name.toLowerCase().includes(q) || t.summary.toLowerCase().includes(q) || category.toLowerCase().includes(q)));
      if (!items.length) continue;
      const group = h("div", { class: "palette-group" }, h("h4", { text: category }));
      for (const item of items) {
        group.append(h("button", {
          type: "button",
          class: "palette-item" + (first ? " active" : ""),
          dataset: { name: item.name },
          onclick: () => pickTransform(item.name),
        }, h("span", { class: "dot " + catClass(item.category) }), h("span", {}, h("strong", { text: item.name }), h("span", { class: "desc", text: item.summary }))));
        first = false;
      }
      list.append(group);
    }
    if (first) list.append(h("p", { class: "muted", text: "No transform matches that search." }));
  }

  async function pickTransform(name) {
    $("palette").close();
    const target = S.palette;
    try {
      const node = await api("/api/step", { name });
      if (target.mode === "replace") {
        commit(() => setNode(target.path, node), { select: target.path });
        return;
      }
      let containerPath = target.path;
      let container = getNode(containerPath);
      // The header button adds a step in sequence; only Compose roots take it as a child.
      const sequential = containerPath.length === 0 && container.name !== "Compose";
      if (sequential || nestKey(container) !== "transforms") {
        commit(() => { S.policy = { name: "Compose", params: { transforms: [S.policy, node], p: 1.0, seed: null } }; }, { select: [1] });
        return;
      }
      const index = (container.params.transforms || []).length;
      commit(() => {
        container = getNode(containerPath);
        insertChildren(container, index, [node], null);
      }, { select: [...containerPath, index] });
    } catch (error) {
      toast(error.message, true);
    }
  }

  // ---------------------------------------------------------------- preview
  // Draw options travel as digit strings so 64-bit seeds and numeric sample IDs
  // (e.g. accession numbers) reach Python exactly, as training would use them.
  function digits(id) {
    const text = $(id).value.trim();
    return /^\d+$/.test(text) ? text : "0";
  }

  function drawOptions() {
    const sample = $("sample-id").value.trim() || "studio";
    return {
      seed: digits("seed"),
      sample_id: sample,
      numeric_sample_id: /^\d+$/.test(sample),
      epoch: digits("epoch"),
      index: digits("draw-index"),
    };
  }

  function viewOptions() {
    const options = {
      axis: S.axis,
      position: Number($("slice").value) / 1000,
      channel: $("channel").value || null,
      mask_alpha: Number($("mask-alpha").value) / 100,
      window_mode: $("window-mode").value,
    };
    if (options.window_mode === "manual") options.window = [Number($("window-low").value), Number($("window-high").value)];
    return options;
  }

  function schedulePreview(delay = 120) {
    if (!S.data.loaded) return;
    debounce("preview", delay, refreshPreview);
  }

  async function refreshPreview() {
    if (!S.data.loaded || !S.validity.ok) return;
    const seq = ++S.previewSeq;
    $("preview-busy").hidden = false;
    try {
      const result = await api("/api/preview", { policy: S.policy, ...drawOptions(), ...viewOptions() });
      if (seq !== S.previewSeq) return;
      S.preview = result;
      renderPreview();
    } catch (error) {
      if (seq !== S.previewSeq) return;
      if (error.path) S.validity = { ok: false, error: error.message, path: error.path };
      showPreviewError(error.message);
      renderStatus();
    } finally {
      if (seq === S.previewSeq) $("preview-busy").hidden = true;
    }
  }

  function showPreviewError(message) {
    const el = $("output-error");
    el.textContent = message;
    el.hidden = false;
  }

  const imageCache = new Map();
  function loadImage(url) {
    if (!imageCache.has(url)) {
      if (imageCache.size > 64) imageCache.delete(imageCache.keys().next().value);
      imageCache.set(url, new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = url;
      }));
    }
    return imageCache.get(url);
  }

  const geometry = new WeakMap();
  async function drawCanvas(canvas, view, frameAspect) {
    const wrap = canvas.parentElement;
    wrap.style.aspectRatio = String(frameAspect);
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = wrap.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!view) { geometry.delete(canvas); return; }
    const image = await loadImage(view.png);
    const cw = canvas.width;
    const ch = canvas.height;
    const physicalW = view.width;
    const physicalH = view.height * view.aspect;
    const scale = Math.min(cw / physicalW, ch / physicalH) * S.zoom;
    const dw = physicalW * scale;
    const dh = physicalH * scale;
    const ox = (cw - dw) / 2 + S.pan.x * dw;
    const oy = (ch - dh) / 2 + S.pan.y * dh;
    geometry.set(canvas, { ox, oy, dw, dh, cw, ch, dpr });
    ctx.clearRect(0, 0, cw, ch);
    ctx.imageSmoothingEnabled = dw / view.width < 3;
    ctx.drawImage(image, ox, oy, dw, dh);
    if (!$("show-targets").checked || !view.targets) return;
    const sx = dw / view.width;
    const sy = dh / view.height;
    ctx.lineWidth = 1.5 * dpr;
    ctx.font = `${11 * dpr}px system-ui, sans-serif`;
    ctx.textBaseline = "bottom";
    for (const box of view.targets.boxes) {
      const x0 = ox + (Math.min(box.col0, box.col1) + 0.5) * sx;
      const y0 = oy + (Math.min(box.row0, box.row1) + 0.5) * sy;
      const x1 = ox + (Math.max(box.col0, box.col1) + 0.5) * sx;
      const y1 = oy + (Math.max(box.row0, box.row1) + 0.5) * sy;
      ctx.strokeStyle = "#ffd166";
      ctx.setLineDash([5 * dpr, 3 * dpr]);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]);
      if (box.label) { ctx.fillStyle = "#ffd166"; ctx.fillText(box.label, x0 + 2 * dpr, y0 - 2 * dpr); }
    }
    for (const point of view.targets.keypoints) {
      const x = ox + (point.col + 0.5) * sx;
      const y = oy + (point.row + 0.5) * sy;
      ctx.fillStyle = "#ff4d6d";
      ctx.strokeStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(x, y, 4 * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      if (point.label) { ctx.fillStyle = "#ffffff"; ctx.fillText(point.label, x + 6 * dpr, y - 3 * dpr); }
    }
  }

  function redrawViews() {
    const result = S.preview;
    const input = result ? result.input : null;
    const frame = input ? input.width / (input.height * input.aspect) : 1;
    drawCanvas($("canvas-input"), input, frame);
    drawCanvas($("canvas-output"), result ? result.output : null, frame);
    const showDiff = $("show-diff").checked && result && result.diff;
    $("diff-view").hidden = !showDiff;
    if (showDiff) drawCanvas($("canvas-diff"), { ...input, png: result.diff, targets: null }, frame);
  }

  function renderPreview() {
    const r = S.preview;
    $("output-error").hidden = !r.error;
    if (r.error) $("output-error").textContent = `${r.error.type}: ${r.error.message}`;
    const axisLabel = S.data.ndim === 3 ? `${S.axis}=${r.input.index}/${r.input.length - 1}` : "2D";
    $("slice-label").textContent = S.data.ndim === 3 ? `${r.input.index} / ${r.input.length - 1}` : "—";
    $("input-meta").textContent = `${r.summary.input.shape.join("×")} · ${axisLabel}`;
    $("output-meta").textContent = r.output
      ? `${r.summary.output.shape.join("×")}` + (S.data.ndim === 3 ? ` · ${S.axis}=${r.output.index}/${r.output.length - 1}` : "")
      : "failed";
    if ($("window-mode").value !== "manual") {
      $("window-low").value = fmt(r.window.input[0], 4);
      $("window-high").value = fmt(r.window.input[1], 4);
    }
    redrawViews();
    renderTrace(r);
    renderIssues(r);
    renderFacts(r);
  }

  function tracePathToTree(tracePath) {
    const tokens = tracePath.split(".").slice(1);
    const path = [];
    let node = S.policy;
    for (const token of tokens) {
      if (token === "inner") break;
      if (token === "guarded") { path.push("transform"); node = node && node.params.transform; continue; }
      const index = Number(token);
      path.push(index);
      node = node && node.params.transforms ? node.params.transforms[index] : null;
    }
    return node ? path : null;
  }

  function renderTrace(r) {
    const body = $("trace-table").tBodies[0];
    body.replaceChildren();
    $("elapsed").textContent = `· ${fmt(r.elapsed_ms, 1)} ms`;
    if (!r.trace.length) body.append(h("tr", {}, h("td", { class: "muted", text: "The policy has no steps." })));
    for (const event of r.trace) {
      const depth = event.path.split(".").length - 1;
      const treePath = tracePathToTree(event.path);
      body.append(h("tr", {
        class: event.action === "skipped" ? "skipped" : "",
        style: treePath ? "cursor:pointer" : null,
        onclick: treePath ? () => select(treePath) : null,
      },
      h("td", { style: `padding-left:${6 + depth * 12}px`, text: event.name }),
      h("td", {}, h("span", { class: `pill ${event.action}`, text: event.action })),
      h("td", { class: "num", text: event.action === "applied" ? `${fmt(event.ms, 1)} ms` : "" })));
    }
  }

  function renderIssues(r) {
    const list = $("issues");
    list.replaceChildren();
    if (r.error) { list.append(h("li", { class: "error", text: `Transform raised ${r.error.type}` })); return; }
    if (!r.issues.length) { list.append(h("li", { class: "ok", text: "No validation issues for this draw." })); return; }
    for (const issue of r.issues) list.append(h("li", { class: issue.severity, text: `${issue.check}: ${issue.message}` }));
  }

  function renderFacts(r) {
    const dl = $("output-facts");
    dl.replaceChildren();
    const a = r.summary.input;
    const b = r.summary.output;
    const row = (label, before, after) => {
      const changed = after !== undefined && after !== before;
      dl.append(h("dt", { text: label }), h("dd", { class: changed ? "changed" : "", text: after === undefined ? before : (changed ? `${before} → ${after}` : after) }));
    };
    const range = (s) => (s && s.intensity ? `${fmt(s.intensity.min)} … ${fmt(s.intensity.max)}` : "—");
    row("Sample seed", r.sample_seed);
    if (!b) { row("Shape", a.shape.join("×")); return; }
    row("Shape", a.shape.join("×"), b.shape.join("×"));
    row("Spacing", a.spacing.map((s) => fmt(s, 3)).join(", "), b.spacing.map((s) => fmt(s, 3)).join(", "));
    row("Intensity", range(a), range(b));
    row("Labels", a.labels.join(", ") || "none", b.labels.join(", ") || "none");
    if (a.keypoints || b.keypoints) row("Keypoints", String(a.keypoints), String(b.keypoints));
    if (a.bboxes || b.bboxes) row("Boxes", String(a.bboxes), String(b.bboxes));
    if (a.channels.length) row("Channels", a.channels.join(", "), b.channels.join(", "));
  }

  // ----------------------------------------------------------- zoom & pan
  function bindViewerInteractions(canvas) {
    let drag = null;
    canvas.addEventListener("wheel", (event) => {
      const g = geometry.get(canvas);
      if (!g) return;
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const px = (event.clientX - rect.left) * g.dpr;
      const py = (event.clientY - rect.top) * g.dpr;
      const u = (px - g.ox) / g.dw;
      const v = (py - g.oy) / g.dh;
      const next = Math.min(16, Math.max(1, S.zoom * Math.exp(-event.deltaY * 0.0015)));
      const factor = next / S.zoom;
      const dw = g.dw * factor;
      const dh = g.dh * factor;
      S.pan.x = (px - u * dw - (g.cw - dw) / 2) / dw;
      S.pan.y = (py - v * dh - (g.ch - dh) / 2) / dh;
      S.zoom = next;
      if (next === 1) S.pan = { x: 0, y: 0 };
      redrawViews();
    }, { passive: false });
    canvas.addEventListener("pointerdown", (event) => {
      const g = geometry.get(canvas);
      if (!g) return;
      drag = { x: event.clientX, y: event.clientY, pan: { ...S.pan }, g };
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add("dragging");
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!drag) return;
      S.pan.x = drag.pan.x + ((event.clientX - drag.x) * drag.g.dpr) / drag.g.dw;
      S.pan.y = drag.pan.y + ((event.clientY - drag.y) * drag.g.dpr) / drag.g.dh;
      redrawViews();
    });
    const end = () => { drag = null; canvas.classList.remove("dragging"); };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
    canvas.addEventListener("dblclick", () => { S.zoom = 1; S.pan = { x: 0, y: 0 }; redrawViews(); });
  }

  // ---------------------------------------------------------------- gallery
  async function renderGallery() {
    if (!S.data.loaded) return toast("Load data first", true);
    if (!S.validity.ok) return toast("Fix the policy first", true);
    const button = $("btn-gallery");
    button.disabled = true;
    button.textContent = "Rendering…";
    try {
      const result = await api("/api/gallery", { policy: S.policy, ...drawOptions(), ...viewOptions(), count: Number($("gallery-count").value) });
      const grid = $("gallery");
      grid.replaceChildren();
      for (const item of result.items) {
        let picture = null;
        if (item.view) {
          // Respect physical pixel aspect (anisotropic spacing), like the preview.
          const ratio = item.view.width / (item.view.height * item.view.aspect);
          picture = h("img", { src: item.view.png, alt: `Draw ${item.index}` });
          picture.style.objectFit = "fill";
          picture.style.width = ratio >= 1 ? "100%" : `${ratio * 100}%`;
          picture.style.height = ratio >= 1 ? `${100 / ratio}%` : "100%";
        }
        const thumb = h("div", { class: "thumb" }, picture || h("div", { class: "tile-error", text: item.error ? `${item.error.type}: ${item.error.message}` : "Failed" }));
        const status = item.error ? h("span", { class: "pill applied", style: "background:var(--danger-soft);color:var(--danger)", text: "error" })
          : item.errors ? h("span", { class: "pill", style: "background:var(--danger-soft);color:var(--danger)", text: `${item.errors} error${item.errors > 1 ? "s" : ""}` })
          : item.warnings ? h("span", { class: "pill", style: "background:var(--warn-soft);color:var(--warn)", text: `${item.warnings} warning${item.warnings > 1 ? "s" : ""}` })
          : h("span", { class: "pill applied", text: "ok" });
        grid.append(h("button", { type: "button", class: "tile", title: item.applied.join(" → ") || "No steps applied", onclick: () => {
          $("draw-index").value = item.index;
          switchTab("preview");
          schedulePreview(0);
        } }, thumb, h("div", { class: "tile-meta" }, h("span", { text: `Draw ${item.index}` }), status), h("div", { class: "tile-applied", text: item.applied.filter((n) => !S.byName[n] || !S.byName[n].container).join(", ") || "no change" })));
      }
    } catch (error) {
      toast(error.message, true);
    } finally {
      button.disabled = false;
      button.textContent = "Render gallery";
    }
  }

  // ----------------------------------------------------------------- stress
  function renderValidatorForm() {
    const form = $("validator-form");
    form.replaceChildren();
    for (const [key, value] of Object.entries(S.catalog.validator)) {
      let input;
      if (typeof value === "boolean") {
        input = h("input", { type: "checkbox", dataset: { key, kind: "bool" } });
        input.checked = value;
        form.append(h("label", { class: "check" }, input, key));
      } else {
        input = h("input", { type: "text", dataset: { key, kind: "json" }, value: JSON.stringify(value), spellcheck: "false" });
        form.append(h("label", { class: "stack" }, key, input));
      }
    }
  }

  function validatorConfig() {
    const config = {};
    for (const input of $("validator-form").querySelectorAll("input")) {
      const { key, kind } = input.dataset;
      if (kind === "bool") config[key] = input.checked;
      else {
        try { config[key] = JSON.parse(input.value); input.classList.remove("invalid"); } catch { input.classList.add("invalid"); throw new Error(`Validator setting ${key} is not valid JSON`); }
      }
    }
    return config;
  }

  async function runStress() {
    if (!S.data.loaded) return toast("Load data first", true);
    if (!S.validity.ok) return toast("Fix the policy first", true);
    const button = $("btn-stress");
    const out = $("stress-results");
    let validator;
    try { validator = validatorConfig(); } catch (error) { return toast(error.message, true); }
    button.disabled = true;
    button.textContent = "Running…";
    out.replaceChildren(h("p", { class: "muted", text: "Running draws — large volumes can take a while." }));
    try {
      const r = await api("/api/stress", { policy: S.policy, ...drawOptions(), draws: Number($("stress-draws").value), validator });
      const pct = (n) => (r.completed ? `${Math.round((100 * n) / r.completed)}%` : "—");
      const stat = (cls, value, label) => h("div", { class: `stat ${cls}` }, h("div", { class: "value", text: value }), h("div", { class: "label", text: label }));
      const nodes = [
        h("div", { class: "stat-row" },
          stat("ok", pct(r.passed), `passed (${r.passed}/${r.completed})`),
          stat(r.rejected ? "bad" : "", pct(r.rejected), `rejected by validator (${r.rejected})`),
          stat(r.exceptions ? "bad" : "", String(r.exceptions), "transform exceptions"),
          stat("", `${fmt(r.timing_ms.mean, 1)} ms`, "mean time per draw"),
          stat("", `${fmt(r.timing_ms.p95, 1)} ms`, "95th percentile")),
      ];
      if (r.truncated) nodes.push(h("p", { class: "muted", text: `Stopped after ${r.completed} of ${r.requested} draws (time budget).` }));
      const table = (headers, rows) => h("table", { class: "table" }, h("thead", {}, h("tr", {}, ...headers.map((t, i) => h("th", { class: i ? "num" : "", text: t })))), h("tbody", {}, ...rows));
      const applied = r.applied.map((entry) => {
        const treePath = tracePathToTree(entry.path);
        const node = treePath ? getNode(treePath) : null;
        const configured = node && node.params && typeof node.params.p === "number" ? fmt(node.params.p, 2) : "—";
        return h("tr", {}, h("td", { style: `padding-left:${6 + (entry.path.split(".").length - 2) * 12}px`, text: entry.name || entry.path }),
          h("td", { class: "num", text: configured }),
          h("td", { class: "num", text: `${Math.round(entry.rate * 100)}%` }),
          h("td", {}, h("div", { class: "bar" }, h("span", { style: `width:${Math.round(entry.rate * 100)}%` }))));
      });
      nodes.push(h("section", { class: "card" }, h("h3", { text: "How often each step ran" }), h("p", { class: "muted", text: "Observed rates include container choices, so children of OneOf/SomeOf run less often than their own p." }), table(["Step", "Configured p", "Observed", ""], applied)));
      nodes.push(h("section", { class: "card" }, h("h3", { text: "Validation findings" }), r.issues.length
        ? table(["Check", "Severity", "Draws"], r.issues.map((i) => h("tr", {}, h("td", { text: i.check }), h("td", { class: "num", text: i.severity }), h("td", { class: "num", text: String(i.count) }))))
        : h("p", { class: "muted", text: "No validation issues across these draws." })));
      if (r.exception_messages.length) {
        nodes.push(h("section", { class: "card" }, h("h3", { text: "Exceptions" }), table(["Message", "Draws"], r.exception_messages.map((e) => h("tr", {}, h("td", { text: e.message }), h("td", { class: "num", text: String(e.count) }))))));
      }
      nodes.push(h("section", { class: "card" }, h("h3", { text: "Output shapes" }), table(["Shape", "Draws"], r.shapes.map((s) => h("tr", {}, h("td", { text: s.shape }), h("td", { class: "num", text: String(s.count) }))))));
      out.replaceChildren(...nodes);
    } catch (error) {
      out.replaceChildren(h("p", { class: "error-text", text: error.message }));
    } finally {
      button.disabled = false;
      button.textContent = "Run stress test";
    }
  }

  // ------------------------------------------------------------------- data
  function applyData(state) {
    S.data = state;
    if (state.policy && !S.policy) S.policy = state.policy;
    const chip = $("data-chip");
    if (!state.loaded) {
      chip.textContent = "No data loaded";
      $("viewer-empty").hidden = false;
      return;
    }
    const s = state.summary;
    chip.textContent = `${state.source} · ${s.shape.join("×")}${state.channels.length ? ` · +${state.channels.length} ch` : ""}`;
    chip.title = chip.textContent;
    $("viewer-empty").hidden = true;
    if (!state.axes.includes(S.axis)) S.axis = state.axes[0];
    const axisBox = $("axis-buttons");
    axisBox.replaceChildren();
    axisBox.hidden = state.ndim !== 3;
    $("slice").disabled = state.ndim !== 3;
    for (const axis of state.axes) {
      axisBox.append(h("button", { type: "button", role: "radio", "aria-checked": axis === S.axis ? "true" : "false", class: axis === S.axis ? "active" : "", onclick: () => {
        S.axis = axis;
        S.zoom = 1;
        S.pan = { x: 0, y: 0 };
        applyData(S.data);
        schedulePreview(0);
      } }, axis));
    }
    const channel = $("channel");
    const previous = channel.value;
    channel.replaceChildren(h("option", { value: "" }, "primary"), ...state.channels.map((name) => h("option", { value: name }, name)));
    channel.value = state.channels.includes(previous) ? previous : "";
    $("channel-wrap").hidden = !state.channels.length;
    const list = $("channel-list");
    list.replaceChildren(...state.channels.map((name) => h("li", {}, name, h("button", { type: "button", title: `Remove ${name}`, "aria-label": `Remove ${name}`, onclick: () => dataAction(() => api("/api/channel/remove", { name })) }, "×"))));
    if (!state.channels.length) list.append(h("li", { class: "muted", style: "border:0;padding-left:0", text: "No extra channels." }));
    renderSummary(state);
  }

  function renderSummary(state) {
    const s = state.summary;
    const facts = h("dl", { class: "facts" });
    const add = (k, v) => facts.append(h("dt", { text: k }), h("dd", { text: v }));
    add("Source", state.source);
    add("Modality", state.modality || "unknown");
    add("Shape", s.shape.join(" × "));
    add("Spacing (mm)", s.spacing_mm.map((v) => fmt(v, 4)).join(", "));
    add("Data type", s.dtype);
    add("Patient geometry", s.geometry_known ? "known" : "unknown");
    if (s.intensity) add("Intensity", `min ${fmt(s.intensity.min)} · max ${fmt(s.intensity.max)} · mean ${fmt(s.intensity.mean)} · sd ${fmt(s.intensity.std)}`);
    add("Finite values", `${fmt(100 * s.finite_fraction, 2)}%`);
    add("Keypoints / boxes", `${s.keypoints} / ${s.bboxes}`);
    if (state.channels.length) add("Channels", state.channels.join(", "));
    const nodes = [facts];
    if (s.labels.length) {
      nodes.push(h("table", { class: "table" }, h("thead", {}, h("tr", {}, h("th", { text: "Label" }), h("th", { class: "num", text: "Voxels" }), h("th", { class: "num", text: state.ndim === 3 ? "Volume (mm³)" : "Area (mm²)" }))),
        h("tbody", {}, ...s.labels.map((l) => h("tr", {}, h("td", { text: String(l.id) }), h("td", { class: "num", text: String(l.voxels) }), h("td", { class: "num", text: fmt(l.physical_measure, 1) }))))));
    } else {
      nodes.push(h("p", { class: "muted", text: "No mask loaded." }));
    }
    if (s.issues.length) nodes.push(h("ul", { class: "issues" }, ...s.issues.map((i) => h("li", { class: i.severity, text: `${i.check}: ${i.message}` }))));
    $("data-summary").replaceChildren(...nodes);
    $("data-summary").classList.remove("muted");
  }

  async function dataAction(request, successMessage) {
    try {
      const state = await request();
      S.zoom = 1;
      S.pan = { x: 0, y: 0 };
      applyData(state);
      if (successMessage) toast(successMessage);
      // Volume-aware suggestions (crop sizes, grids) may differ, so revalidate.
      debounce("sync", 0, syncPolicy);
    } catch (error) {
      toast(error.message, true);
    }
  }

  // ---------------------------------------------------------- import/export
  const EXPORT_HINTS = {
    json: "Load with medaugmentx.serialization.from_json(text). Includes schema version, library version and a SHA-256 policy hash.",
    yaml: "Load with medaugmentx.serialization.from_yaml(text) (requires PyYAML).",
    python: "Readable constructor code; values equal to defaults are omitted. Prefer JSON/YAML for exact, hash-verified reproduction.",
  };
  let exportCache = null;

  async function showExport(format) {
    S.exportFormat = format;
    for (const button of $("export-formats").querySelectorAll("button")) button.classList.toggle("active", button.dataset.format === format);
    $("export-hint").textContent = EXPORT_HINTS[format];
    $("export-text").value = "Generating…";
    try {
      exportCache = await api("/api/export", { policy: S.policy, format });
      $("export-text").value = exportCache.text;
    } catch (error) {
      exportCache = null;
      $("export-text").value = error.message;
    }
  }

  async function importPolicy() {
    const text = $("import-text").value;
    try {
      const result = await api("/api/import", { text });
      commit(() => { S.policy = result.policy; }, { select: null });
      $("import-dialog").close();
      toast(result.notices.length ? result.notices.join(" ") : "Policy imported");
    } catch (error) {
      $("import-error").textContent = error.message;
    }
  }

  // ------------------------------------------------------------------- tabs
  function switchTab(name) {
    for (const tab of document.querySelectorAll(".tab")) {
      const active = tab.dataset.tab === name;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    }
    for (const page of document.querySelectorAll(".tab-page")) page.classList.toggle("active", page.id === `tab-${name}`);
    if (name === "preview") requestAnimationFrame(redrawViews);
  }

  function stepSlice(direction) {
    if (!S.preview || S.data.ndim !== 3) return;
    const length = S.preview.input.length;
    if (length < 2) return;
    const index = Math.min(length - 1, Math.max(0, S.preview.input.index + direction));
    $("slice").value = String(Math.round((index / (length - 1)) * 1000));
    schedulePreview(0);
  }

  function stepDraw(direction) {
    const input = $("draw-index");
    input.value = String(Math.max(0, (Number(input.value) || 0) + direction));
    schedulePreview(0);
  }

  // ------------------------------------------------------------------- init
  function bindEvents() {
    for (const tab of document.querySelectorAll(".tab")) tab.addEventListener("click", () => switchTab(tab.dataset.tab));
    $("data-chip").addEventListener("click", () => switchTab("data"));
    $("btn-add-root").addEventListener("click", () => openPalette({ mode: "append", path: [] }));
    $("btn-undo").addEventListener("click", undo);
    $("btn-redo").addEventListener("click", redo);
    $("btn-clear").addEventListener("click", () => {
      if (countSteps(S.policy) && !confirm("Remove every step from the policy?")) return;
      commit(() => { S.policy = structuredClone(ROOT_TEMPLATE); }, { select: null });
    });

    $("preset-select").addEventListener("change", () => {
      const name = $("preset-select").value;
      const box = $("preset-options");
      box.replaceChildren();
      box.hidden = !name;
      if (!name) return;
      const preset = S.catalog.presets.find((p) => p.name === name);
      box.append(h("span", { class: "muted", text: preset.summary }));
      for (const option of preset.options) box.append(h("label", { class: "check" }, h("input", { type: "checkbox", dataset: { option } }), option));
      box.append(h("button", { type: "button", class: "btn small primary", onclick: async () => {
        const options = {};
        for (const input of box.querySelectorAll("input[data-option]")) options[input.dataset.option] = input.checked;
        if (countSteps(S.policy) && !confirm("Replace the current policy with this preset?")) return;
        try {
          const node = await api("/api/preset", { name, options });
          commit(() => { S.policy = node; }, { select: null });
          $("preset-select").value = "";
          box.hidden = true;
          toast(`Loaded ${name.toUpperCase()} preset`);
        } catch (error) { toast(error.message, true); }
      } }, "Apply"), h("button", { type: "button", class: "btn small ghost", onclick: () => { $("preset-select").value = ""; box.hidden = true; } }, "Cancel"));
    });

    for (const id of ["seed", "sample-id", "epoch", "draw-index"]) $(id).addEventListener("change", () => schedulePreview(0));
    $("btn-prev-draw").addEventListener("click", () => stepDraw(-1));
    $("btn-next-draw").addEventListener("click", () => stepDraw(1));
    $("btn-random-seed").addEventListener("click", () => { $("seed").value = String(Math.floor(Math.random() * 2 ** 31)); schedulePreview(0); });
    $("slice").addEventListener("input", () => schedulePreview(60));
    $("channel").addEventListener("change", () => schedulePreview(0));
    $("mask-alpha").addEventListener("input", () => schedulePreview(80));
    $("show-targets").addEventListener("change", redrawViews);
    $("show-diff").addEventListener("change", redrawViews);
    $("window-mode").addEventListener("change", () => { $("manual-window").hidden = $("window-mode").value !== "manual"; schedulePreview(0); });
    $("window-low").addEventListener("change", () => schedulePreview(0));
    $("window-high").addEventListener("change", () => schedulePreview(0));
    $("btn-quick-phantom").addEventListener("click", () => dataAction(() => api("/api/load/phantom", { name: "mri", demo_targets: true }), "Loaded MRI phantom"));
    $("btn-record").addEventListener("click", async () => {
      try {
        const record = await api("/api/record", { policy: S.policy, ...drawOptions() });
        download("augmentation-record.json", JSON.stringify(record, null, 2));
      } catch (error) { toast(error.message, true); }
    });
    for (const id of ["canvas-input", "canvas-output", "canvas-diff"]) bindViewerInteractions($(id));
    new ResizeObserver(() => debounce("resize", 60, redrawViews)).observe($("viewer"));

    $("btn-gallery").addEventListener("click", renderGallery);
    $("btn-stress").addEventListener("click", runStress);

    $("btn-phantom").addEventListener("click", () => dataAction(() => api("/api/load/phantom", { name: $("phantom-select").value, demo_targets: $("phantom-demo").checked }), "Phantom loaded"));
    $("btn-path").addEventListener("click", () => dataAction(() => api("/api/load/path", {
      image: $("path-image").value, mask: $("path-mask").value || null,
      resample_mask: $("path-resample").checked, strict: $("path-strict").checked,
    }), "Data loaded"));
    $("upload-image").addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) dataAction(() => upload("image", file), `Loaded ${file.name}`);
      event.target.value = "";
    });
    $("upload-mask").addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) dataAction(() => upload("mask", file, { resample_mask: $("path-resample").checked ? "1" : "0" }), `Loaded mask ${file.name}`);
      event.target.value = "";
    });
    $("btn-channel").addEventListener("click", () => dataAction(() => api("/api/channel/add", { name: $("channel-name").value.trim(), path: $("channel-path").value }), "Channel added"));
    $("upload-channel").addEventListener("change", (event) => {
      const file = event.target.files[0];
      const name = $("channel-name").value.trim();
      if (file && !name) toast("Enter a channel name before uploading", true);
      else if (file) dataAction(() => upload("channel", file, { channel: name }), `Added channel ${name}`);
      event.target.value = "";
    });

    $("btn-export").addEventListener("click", () => {
      if (!S.validity.ok) return toast(`Fix the policy first: ${S.validity.error}`, true);
      $("export-dialog").showModal();
      showExport(S.exportFormat);
    });
    for (const button of $("export-formats").querySelectorAll("button")) button.addEventListener("click", () => showExport(button.dataset.format));
    $("btn-copy").addEventListener("click", async () => {
      try { await navigator.clipboard.writeText($("export-text").value); toast("Copied to clipboard"); }
      catch { $("export-text").select(); toast("Press Ctrl/Cmd+C to copy", true); }
    });
    $("btn-download").addEventListener("click", () => { if (exportCache) download(exportCache.filename, exportCache.text, "text/plain"); });
    $("btn-import").addEventListener("click", () => { $("import-error").textContent = ""; $("import-dialog").showModal(); });
    $("import-file").addEventListener("change", async (event) => {
      const file = event.target.files[0];
      if (file) $("import-text").value = await file.text();
    });
    $("btn-import-apply").addEventListener("click", importPolicy);

    $("palette-search").addEventListener("input", (event) => renderPalette(event.target.value));
    $("palette-search").addEventListener("keydown", (event) => {
      const items = [...$("palette-list").querySelectorAll(".palette-item")];
      if (!items.length) return;
      let index = items.findIndex((item) => item.classList.contains("active"));
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        items[index]?.classList.remove("active");
        index = (index + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
        items[index].classList.add("active");
        items[index].scrollIntoView({ block: "nearest" });
      } else if (event.key === "Enter") {
        event.preventDefault();
        pickTransform(items[Math.max(0, index)].dataset.name);
      }
    });

    document.addEventListener("keydown", (event) => {
      const mod = event.metaKey || event.ctrlKey;
      if (mod && event.key.toLowerCase() === "z" && !isTyping(event)) {
        event.preventDefault();
        event.shiftKey ? redo() : undo();
        return;
      }
      if (isTyping(event) || mod || document.querySelector("dialog[open]")) return;
      if (event.key === "ArrowLeft") { event.preventDefault(); stepSlice(-1); }
      else if (event.key === "ArrowRight") { event.preventDefault(); stepSlice(1); }
      else if (event.key === "n" || event.key === "N") stepDraw(1);
      else if (event.key === "p" || event.key === "P") stepDraw(-1);
      else if (event.key === "r" || event.key === "R") $("btn-random-seed").click();
    });
  }

  function readToken() {
    const params = new URLSearchParams(location.search);
    const token = params.get("token");
    if (token) {
      try { sessionStorage.setItem("medaugmentx-studio-token", token); } catch { /* storage unavailable */ }
      history.replaceState(null, "", location.pathname);
      return token;
    }
    try { return sessionStorage.getItem("medaugmentx-studio-token"); } catch { return null; }
  }

  async function init() {
    S.token = readToken();
    if (!S.token) { $("locked").hidden = false; return; }
    try {
      S.catalog = await api("/api/catalog");
    } catch (error) {
      $("locked").hidden = false;
      if (error.status !== 401) $("locked").querySelector("p").textContent = error.message;
      try { sessionStorage.removeItem("medaugmentx-studio-token"); } catch { /* ignore */ }
      return;
    }
    for (const entry of S.catalog.transforms) S.byName[entry.name] = entry;
    $("version").textContent = `v${S.catalog.version}`;
    $("preset-select").append(...S.catalog.presets.map((p) => h("option", { value: p.name }, `${p.name.toUpperCase()} preset`)));
    $("phantom-select").replaceChildren(...S.catalog.phantoms.map((p) => h("option", { value: p }, `${p.toUpperCase()} phantom`)));
    renderValidatorForm();
    bindEvents();
    const state = await api("/api/state");
    S.policy = state.policy || structuredClone(ROOT_TEMPLATE);
    applyData(state);
    afterPolicyChange();
    switchTab("preview");
  }

  init().catch((error) => toast(error.message, true));
})();

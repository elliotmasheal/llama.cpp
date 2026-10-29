(function () {
  "use strict";

  const M = window.ARCH_MODEL;
  if (!M) {
    document.body.insertAdjacentHTML("afterbegin", "<p style='padding:20px;color:#ff6b6b'>architecture-model.js 未加载</p>");
    return;
  }

  const NS = "http://www.w3.org/2000/svg";
  const SANS = '"Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, sans-serif';
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const TYPE_COLOR = {
    ext: "#7d8b99",
    tool: "#4cc9f0",
    common: "#6ea8fe",
    core: "#b388ff",
    ggml: "#ffb86b",
    hw: "#7ee787",
    meta: "#f778ba"
  };
  const TYPE_LABEL = {
    ext: "外部",
    tool: "工具 / 入口",
    common: "common 编排",
    core: "llama core",
    ggml: "ggml 运行时",
    hw: "硬件层",
    meta: "验证 / 账本"
  };
  const STATUS_LABEL = {
    current: "已落地",
    experimental: "实验项",
    planned: "规划项",
    partial: "部分落地"
  };
  const STATUS_COLOR = {
    current: "#7ee787",
    experimental: "#f4a261",
    planned: "#9fb0c2",
    partial: "#f4a261"
  };
  const EDGE_TYPE_LABEL = {
    call: "调用 / 回调",
    data: "数据流",
    config: "参数 / 预算",
    tier: "跨层放置 / 搬运",
    validation: "门禁"
  };
  const EDGE_DESC = {
    call: "源方发起调用或回调, 目标方执行并可能回传结果。",
    data: "张量、token、batch 或拷贝结果沿这条边移动。",
    config: "启动期或实验轮次的参数、预算与判决回灌, 不是每 token 的数据。",
    tier: "对象在 VRAM / RAM / SSD 之间放置或搬运, 受账本预算约束。",
    validation: "测试或质量门验证目标组件, 不参与推理数据流。"
  };

  const $ = (sel, root) => (root || document).querySelector(sel);
  const svg = $("#diagram");
  const gBands = $("#bands");
  const gEdges = $("#edges");
  const gParticles = $("#particles");
  const gNodes = $("#nodes");
  const gEdgeLabels = $("#edgeLabels");
  const canvasWrap = $("#canvasWrap");
  const tooltip = $("#tooltip");
  const panel = $("#panel");
  const chipsEl = $("#scenarios");
  const stepDots = $("#stepDots");
  const storyScenario = $("#storyScenario");
  const storyStep = $("#storyStep");
  const storyText = $("#storyText");
  const stepCounter = $("#stepCounter");
  const playBtn = $("#playBtn");
  const prevBtn = $("#prevBtn");
  const nextBtn = $("#nextBtn");
  const speedSel = $("#speedSel");
  const followChk = $("#followChk");
  const showPlannedChk = $("#showPlannedChk");
  const searchInput = $("#searchInput");

  const nodeById = new Map(M.nodes.map(n => [n.id, n]));
  const edgeById = new Map(M.edges.map(e => [e.id, e]));
  const scenarioById = new Map(M.scenarios.map(s => [s.id, s]));
  const nodeEls = new Map();
  const edgeEls = new Map();
  const edgeRuntime = new Map();

  let scenarioId = "overview";
  let stepIndex = 0;
  let playing = false;
  let speed = 1;
  let follow = true;
  let showNonCurrent = true;
  let selected = null;
  let focusNodes = new Set();
  let activeEdges = new Set();
  let trailEdges = new Set();
  let lastStepAt = performance.now();
  let view = { x: 0, y: 0, w: 1680, h: 1200 };
  let viewAnim = null;
  let dragging = false;
  let suppressClick = false;
  let lastX = 0;
  let lastY = 0;
  let moved = 0;
  let hoveredEdge = null;

  const measureCtx = document.createElement("canvas").getContext("2d");

  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, ch => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[ch]));
  }

  function svgEl(tag, attrs, parent) {
    const el = document.createElementNS(NS, tag);
    if (attrs) {
      for (const key of Object.keys(attrs)) {
        const value = attrs[key];
        if (value !== undefined && value !== null) el.setAttribute(key, String(value));
      }
    }
    if (parent) parent.appendChild(el);
    return el;
  }

  function addText(parent, x, y, text, cls) {
    const el = svgEl("text", { x, y, class: cls }, parent);
    el.textContent = text;
    return el;
  }

  function wrapText(text, maxWidth, font) {
    if (!text) return [];
    measureCtx.font = font;
    const tokens = String(text).split(/(\s+)/).filter(t => t.length);
    const lines = [];
    let cur = "";
    const flush = () => {
      if (cur.trim()) lines.push(cur.trim());
      cur = "";
    };
    for (const tok of tokens) {
      if (!cur && /^\s+$/.test(tok)) continue;
      if (measureCtx.measureText(cur + tok).width <= maxWidth) {
        cur += tok;
        continue;
      }
      if (cur.trim()) flush();
      if (measureCtx.measureText(tok).width <= maxWidth) {
        cur = tok.trimStart();
        continue;
      }
      let part = "";
      for (const ch of tok) {
        if (measureCtx.measureText(part + ch).width > maxWidth && part) {
          lines.push(part);
          part = ch;
        } else {
          part += ch;
        }
      }
      cur = part;
    }
    flush();
    return lines;
  }

  /* ---------- render ---------- */

  function renderBands() {
    for (const band of M.bands) {
      svgEl("rect", { class: "band", x: band.x, y: band.y, width: band.w, height: band.h, rx: 12 }, gBands);
      addText(gBands, band.x + 14, band.y + 20, band.label, "band-title");
      addText(gBands, band.x + 14, band.y + 34, band.sub, "band-sub");
    }
  }

  function anchors(a, b) {
    const acx = a.x + a.w / 2;
    const acy = a.y + a.h / 2;
    const bcx = b.x + b.w / 2;
    const bcy = b.y + b.h / 2;
    const dx = bcx - acx;
    const dy = bcy - acy;
    if (Math.abs(dx) > Math.abs(dy) * 0.9) {
      return dx >= 0
        ? [{ x: a.x + a.w, y: acy, n: "r" }, { x: b.x, y: bcy, n: "l" }]
        : [{ x: a.x, y: acy, n: "l" }, { x: b.x + b.w, y: bcy, n: "r" }];
    }
    return dy >= 0
      ? [{ x: acx, y: a.y + a.h, n: "b" }, { x: bcx, y: b.y, n: "t" }]
      : [{ x: acx, y: a.y, n: "t" }, { x: bcx, y: b.y + b.h, n: "b" }];
  }

  function pathD(a, b) {
    const pair = anchors(a, b);
    const p = pair[0];
    const q = pair[1];
    const dist = Math.hypot(q.x - p.x, q.y - p.y);
    const off = Math.max(26, Math.min(112, dist * 0.38));
    if (p.n === "t" || p.n === "b") {
      const cy1 = p.y + (p.n === "b" ? off : -off);
      const cy2 = q.y + (q.n === "b" ? off : -off);
      return `M ${p.x} ${p.y} C ${p.x} ${cy1}, ${q.x} ${cy2}, ${q.x} ${q.y}`;
    }
    const cx1 = p.x + (p.n === "r" ? off : -off);
    const cx2 = q.x + (q.n === "r" ? off : -off);
    return `M ${p.x} ${p.y} C ${cx1} ${p.y}, ${cx2} ${q.y}, ${q.x} ${q.y}`;
  }

  function renderEdges() {
    for (const edge of M.edges) {
      const a = nodeById.get(edge.from);
      const b = nodeById.get(edge.to);
      if (!a || !b) {
        console.warn("missing edge endpoint", edge);
        continue;
      }
      const g = svgEl("g", {
        class: "edge",
        "data-id": edge.id,
        "data-type": edge.type,
        "data-status": edge.status
      }, gEdges);
      const path = svgEl("path", { class: "path", d: pathD(a, b) }, g);
      const label = svgEl("text", { class: "edge-label", "text-anchor": "middle" }, gEdgeLabels);
      label.textContent = edge.label;
      const len = path.getTotalLength();
      if (len > 0) {
        const mid = path.getPointAtLength(len * 0.52);
        label.setAttribute("x", mid.x);
        label.setAttribute("y", mid.y - 6);
      }
      edgeEls.set(edge.id, g);
      edgeRuntime.set(edge.id, { path, len, labelEl: label, particles: [] });

      const tip = () => `<b>${esc(a.label)} -> ${esc(b.label)}</b>${esc(edge.label)}<br><span style="color:var(--dim)">${EDGE_TYPE_LABEL[edge.type] || edge.type} / ${STATUS_LABEL[edge.status]}</span>`;
      g.addEventListener("click", ev => {
        ev.stopPropagation();
        selectEdge(edge.id);
      });
      g.addEventListener("pointerenter", ev => {
        hoveredEdge = edge.id;
        syncLabels();
        showTip(tip(), ev);
      });
      g.addEventListener("pointermove", moveTip);
      g.addEventListener("pointerleave", () => {
        hoveredEdge = null;
        syncLabels();
        hideTip();
      });
    }
  }

  function syncLabels() {
    const scenarioMode = scenarioId !== "overview";
    for (const edge of M.edges) {
      const rt = edgeRuntime.get(edge.id);
      if (!rt) continue;
      const hidden = !showNonCurrent && (edge.status === "planned" || edge.status === "experimental");
      const isSelected = selected && selected.type === "edge" && selected.id === edge.id;
      const show = !hidden && (isSelected || hoveredEdge === edge.id || (scenarioMode && activeEdges.has(edge.id)));
      rt.labelEl.classList.toggle("show", !!show);
    }
  }

  function renderNodes() {
    for (const n of M.nodes) {
      const g = svgEl("g", {
        class: "node",
        "data-id": n.id,
        "data-type": n.type,
        "data-status": n.status,
        tabindex: "0",
        role: "button",
        "aria-label": n.label
      }, gNodes);

      svgEl("rect", { class: "card", x: n.x, y: n.y, width: n.w, height: n.h, rx: 9 }, g);
      svgEl("rect", {
        x: n.x + 1, y: n.y + 7, width: 3, height: Math.max(6, n.h - 14), rx: 1.5,
        fill: TYPE_COLOR[n.type] || "#7d8b99", opacity: 0.9
      }, g);

      if (n.status !== "current") {
        svgEl("circle", {
          cx: n.x + n.w - 11, cy: n.y + 11, r: 3,
          fill: STATUS_COLOR[n.status] || "#9fb0c2", opacity: 0.95
        }, g);
      }

      const padX = 13;
      const maxW = n.w - padX - 12;
      let y = n.y + 20;
      const labelLines = wrapText(n.label, maxW, "650 14.5px " + SANS).slice(0, 2);
      for (const line of labelLines) {
        addText(g, n.x + padX, y, line, "label");
        y += 17;
      }
      y += 1;

      const subAll = wrapText(n.sub, maxW, "400 10.6px " + SANS);
      const subMax = Math.max(0, Math.floor((n.h - (y - n.y) - 2) / 13) + 1);
      const subLines = subAll.slice(0, subMax);
      if (subAll.length > subMax && subLines.length) {
        subLines[subLines.length - 1] += "...";
      }
      for (const line of subLines) {
        addText(g, n.x + padX, y, line, "sub");
        y += 13;
      }

      if (n.extra && n.h >= 50) {
        const extraAll = wrapText(n.extra, maxW, "400 10.2px " + SANS);
        const extraMax = Math.max(0, Math.floor((n.h - (y - n.y) - 4) / 12) + 1);
        const extraLines = extraAll.slice(0, extraMax);
        if (extraAll.length > extraMax && extraLines.length) {
          extraLines[extraLines.length - 1] += "...";
        }
        for (const line of extraLines) {
          addText(g, n.x + padX, y + 3, line, "extra");
          y += 12;
        }
      }

      nodeEls.set(n.id, g);

      const tip = () => `<b>${esc(n.label)}</b>${esc(n.sub)}<br><span style="color:var(--dim)">${TYPE_LABEL[n.type]} / ${STATUS_LABEL[n.status]} - 点击查看证据</span>`;
      g.addEventListener("click", ev => {
        ev.stopPropagation();
        selectNode(n.id);
      });
      g.addEventListener("keydown", ev => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          ev.stopPropagation();
          selectNode(n.id);
        }
      });
      g.addEventListener("pointerenter", ev => showTip(tip(), ev));
      g.addEventListener("pointermove", moveTip);
      g.addEventListener("pointerleave", hideTip);
      g.addEventListener("dblclick", ev => {
        ev.stopPropagation();
        fitToNodes([n.id], 80, true);
      });
    }
  }

  const NOTES = [
    {
      x: 420, y: 290, w: 350, h: 130,
      title: "工程红线 (硬约束)",
      lines: [
        "容量不足必须退化, 不得 abort",
        "SSD 只做容量层, 不进 decode 单步",
        "开关只用于 A/B: 赢了折叠, 输了删除",
        "结论只进 bench/ 四份文档"
      ]
    },
    {
      x: 824, y: 606, w: 342, h: 96,
      title: "qwen35moe 每层",
      lines: [
        "attn_norm -> [GDN linear | full attn]",
        "-> post_norm -> MoE: router top-8",
        "-> MUL_MAT_ID -> sum + shared expert",
        "-> residual"
      ]
    },
    {
      x: 1326, y: 506, w: 316, h: 58,
      title: "调度回调",
      lines: [
        "copy / stage_inputs / wait_deferred",
        "pre_split -> 预取环与 hybrid 汇合"
      ]
    }
  ];

  function renderNotes() {
    const gNotes = svgEl("g", { id: "notes", "pointer-events": "none" }, null);
    gNodes.parentNode.insertBefore(gNotes, gNodes);
    for (const note of NOTES) {
      svgEl("rect", { class: "note", x: note.x, y: note.y, width: note.w, height: note.h }, gNotes);
      addText(gNotes, note.x + 12, note.y + 20, note.title, "note-title");
      let y = note.y + 37;
      for (const line of note.lines) {
        const wrapped = wrapText(line, note.w - 24, "400 10.2px " + SANS);
        for (const part of wrapped) {
          addText(gNotes, note.x + 12, y, part, "note-line");
          y += 13.5;
        }
      }
    }
  }

  function renderMetrics() {
    $("#metrics").innerHTML = M.metrics.map(m =>
      `<div class="metric ${m.cls || ""}"><div class="v">${esc(m.v)}${m.u ? `<span style="font-size:10px;color:var(--muted)"> ${esc(m.u)}</span>` : ""}</div><div class="k">${esc(m.k)}</div></div>`
    ).join("");
  }

  function renderChips() {
    chipsEl.innerHTML = M.scenarios.map(s =>
      `<button class="chip ${s.id === scenarioId ? "active" : ""}" data-id="${esc(s.id)}" title="快捷键 ${s.key}"><span class="kbd">${s.key}</span>${esc(s.name)}</button>`
    ).join("");
  }

  /* ---------- view ---------- */

  function applyView() {
    svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.w} ${view.h}`);
  }

  function stopViewAnim() {
    if (viewAnim) {
      cancelAnimationFrame(viewAnim);
      viewAnim = null;
    }
  }

  function animateView(target, duration) {
    stopViewAnim();
    if (!duration) {
      view = { ...target };
      applyView();
      return;
    }
    const start = { ...view };
    const t0 = performance.now();
    const step = now => {
      const k = Math.min(1, (now - t0) / duration);
      const e = 1 - Math.pow(1 - k, 3);
      view = {
        x: start.x + (target.x - start.x) * e,
        y: start.y + (target.y - start.y) * e,
        w: start.w + (target.w - start.w) * e,
        h: start.h + (target.h - start.h) * e
      };
      applyView();
      if (k < 1) {
        viewAnim = requestAnimationFrame(step);
      } else {
        viewAnim = null;
      }
    };
    viewAnim = requestAnimationFrame(step);
  }

  function fitBox(x0, y0, x1, y1, pad, animate) {
    const cw = svg.clientWidth || 1000;
    const ch = svg.clientHeight || 700;
    const aspect = cw / ch;
    let w = x1 - x0 + pad * 2;
    let h = y1 - y0 + pad * 2;
    let x = x0 - pad;
    let y = y0 - pad;
    if (w < 640) {
      const d = (640 - w) / 2;
      x -= d;
      w = 640;
    }
    if (h < 420) {
      const d = (420 - h) / 2;
      y -= d;
      h = 420;
    }
    if (w / h < aspect) {
      const nw = h * aspect;
      x -= (nw - w) / 2;
      w = nw;
    } else {
      const nh = w / aspect;
      y -= (nh - h) / 2;
      h = nh;
    }
    animateView({ x, y, w, h }, animate ? 520 : 0);
  }

  function fitAll(animate) {
    fitBox(0, 0, 1680, 1200, 8, animate);
  }

  function fitToNodes(ids, pad, animate) {
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (const id of ids) {
      const n = nodeById.get(id);
      if (!n) continue;
      x0 = Math.min(x0, n.x);
      y0 = Math.min(y0, n.y);
      x1 = Math.max(x1, n.x + n.w);
      y1 = Math.max(y1, n.y + n.h);
    }
    if (!isFinite(x0)) return;
    fitBox(x0, y0, x1, y1, pad == null ? 90 : pad, animate);
  }

  function breakFollow() {
    if (follow) {
      follow = false;
      followChk.checked = false;
    }
  }

  /* ---------- particles ---------- */

  function rebuildParticles() {
    while (gParticles.firstChild) gParticles.removeChild(gParticles.firstChild);
    edgeRuntime.forEach(rt => { rt.particles = []; });
    if (REDUCED) return;
    for (const id of activeEdges) {
      const edge = edgeById.get(id);
      const rt = edgeRuntime.get(id);
      if (!edge || !rt || rt.len <= 0) continue;
      if (!showNonCurrent && (edge.status === "planned" || edge.status === "experimental")) continue;
      const count = Math.max(1, Math.min(4, Math.round(rt.len / 190)));
      for (let i = 0; i < count; i++) {
        const el = svgEl("circle", {
          r: 3,
          class: `particle ${edge.type}`
        }, gParticles);
        rt.particles.push({ el, phase: i / count });
      }
    }
  }

  function updateParticles(now) {
    if (REDUCED) return;
    const base = now * 0.00042 * speed;
    edgeRuntime.forEach(rt => {
      if (!rt.particles.length || rt.len <= 0) return;
      for (const p of rt.particles) {
        const t = (base + p.phase) % 1;
        const pt = rt.path.getPointAtLength(t * rt.len);
        p.el.setAttribute("cx", pt.x);
        p.el.setAttribute("cy", pt.y);
        p.el.setAttribute("opacity", String(0.3 + 0.7 * Math.sin(Math.PI * t)));
      }
    });
  }

  /* ---------- scenario ---------- */

  function setScenario(id, options) {
    const opts = options || {};
    scenarioId = scenarioById.has(id) ? id : "overview";
    stepIndex = 0;
    selected = null;
    clearSearch();
    renderChips();
    updateHash();
    if (!opts.initial) {
      playing = true;
      lastStepAt = performance.now();
    }
    applyStep(!opts.initial);
    updatePlayBtn();
    renderDefaultPanel();
  }

  function applyStep(animate) {
    const sc = scenarioById.get(scenarioId);
    if (!sc) return;
    const step = sc.steps[stepIndex] || sc.steps[0];
    focusNodes = new Set(step.f || []);
    activeEdges = new Set(step.e || []);
    trailEdges = new Set();
    for (let i = 0; i < stepIndex; i++) {
      for (const id of (sc.steps[i].e || [])) trailEdges.add(id);
    }

    const scenarioMode = scenarioId !== "overview";
    nodeEls.forEach((g, id) => {
      g.classList.toggle("focus", focusNodes.has(id));
      g.classList.toggle("dim", scenarioMode && focusNodes.size > 0 && !focusNodes.has(id));
    });
    edgeEls.forEach((g, id) => {
      const active = activeEdges.has(id);
      const trail = !active && trailEdges.has(id);
      g.classList.toggle("active", active);
      g.classList.toggle("trail", trail);
      g.classList.toggle("dim", scenarioMode && focusNodes.size > 0 && !active && !trail);
    });

    rebuildParticles();
    updateStory(sc, step);
    updateSelection();

    if (follow) {
      if (focusNodes.size) {
        fitToNodes(Array.from(focusNodes), 90, animate);
      } else {
        fitAll(animate);
      }
    }
  }

  function updateStory(sc, step) {
    storyScenario.textContent = sc.name;
    storyStep.textContent = `${stepIndex + 1}/${sc.steps.length}`;
    storyText.textContent = step.t;
    stepCounter.textContent = `${stepIndex + 1}/${sc.steps.length}`;
    stepDots.innerHTML = sc.steps.map((_, i) =>
      `<button class="dot ${i === stepIndex ? "active" : ""}" data-step="${i}" aria-label="步骤 ${i + 1}"></button>`
    ).join("");
    const single = sc.steps.length <= 1;
    playBtn.disabled = single;
    prevBtn.disabled = single;
    nextBtn.disabled = single;
  }

  function manualStep(dir) {
    const sc = scenarioById.get(scenarioId);
    if (!sc || sc.steps.length <= 1) return;
    stepIndex = (stepIndex + dir + sc.steps.length) % sc.steps.length;
    playing = false;
    lastStepAt = performance.now();
    updatePlayBtn();
    applyStep(true);
  }

  function updatePlayBtn() {
    playBtn.textContent = playing ? "暂停" : "自动播放";
    playBtn.classList.toggle("primary", !playing);
  }

  function pauseForReading() {
    if (playing) {
      playing = false;
      updatePlayBtn();
    }
  }

  function updateHash() {
    try {
      history.replaceState(null, "", "#" + scenarioId);
    } catch (_) {
      /* file:// may reject history updates */
    }
  }

  /* ---------- selection ---------- */

  function updateSelection() {
    nodeEls.forEach(g => g.classList.remove("selected", "neighbor"));
    edgeEls.forEach(g => g.classList.remove("connected", "selected-edge"));
    if (!selected) {
      syncLabels();
      return;
    }

    if (selected.type === "node") {
      const g = nodeEls.get(selected.id);
      if (g) g.classList.add("selected");
      for (const edge of M.edges) {
        if (edge.from !== selected.id && edge.to !== selected.id) continue;
        const eg = edgeEls.get(edge.id);
        if (eg) eg.classList.add("connected");
        const other = edge.from === selected.id ? edge.to : edge.from;
        const ng = nodeEls.get(other);
        if (ng) ng.classList.add("neighbor");
      }
      syncLabels();
      return;
    }

    const edge = edgeById.get(selected.id);
    if (!edge) {
      syncLabels();
      return;
    }
    const eg = edgeEls.get(edge.id);
    if (eg) eg.classList.add("selected-edge");
    for (const id of [edge.from, edge.to]) {
      const ng = nodeEls.get(id);
      if (ng) ng.classList.add("neighbor");
    }
    syncLabels();
  }

  function selectNode(id) {
    selected = { type: "node", id };
    pauseForReading();
    updateSelection();
    renderNodePanel(id);
    panel.scrollTop = 0;
  }

  function selectEdge(id) {
    selected = { type: "edge", id };
    pauseForReading();
    updateSelection();
    renderEdgePanel(id);
    panel.scrollTop = 0;
  }

  function selectNothing() {
    selected = null;
    updateSelection();
    renderDefaultPanel();
  }

  /* ---------- panel ---------- */

  function refsHtml(refs) {
    return `<ul class="refs">${refs.map(ref => {
      const m = String(ref).match(/^(.+?)(?::(\d+(?:-\d+)?))?$/);
      const path = m ? m[1] : String(ref);
      const range = m && m[2] ? ":" + m[2] : "";
      return `<li><a href="../../${esc(path)}" target="_blank" rel="noreferrer">${esc(path)}</a><span class="range">${esc(range)}</span></li>`;
    }).join("")}</ul>`;
  }

  function legendHtml() {
    return `
      <div class="legend">
        <div class="item"><span class="swatch" style="background:var(--tool)"></span>工具 / 入口</div>
        <div class="item"><span class="swatch" style="background:var(--common)"></span>common 编排</div>
        <div class="item"><span class="swatch" style="background:var(--core)"></span>llama core</div>
        <div class="item"><span class="swatch" style="background:var(--ggml)"></span>ggml 运行时</div>
        <div class="item"><span class="swatch" style="background:var(--hw)"></span>硬件层</div>
        <div class="item"><span class="swatch" style="background:var(--meta)"></span>验证 / 账本</div>
        <div class="item"><span class="line-sample active"></span>当前步骤</div>
        <div class="item"><span class="line-sample dashed"></span>实验 / 部分</div>
        <div class="item"><span class="line-sample dotted"></span>规划</div>
      </div>`;
  }

  function renderDefaultPanel() {
    const rows = M.metrics.map(m =>
      `<tr><td>${esc(m.k)}</td><td class="v">${esc(m.v)}${m.u ? " " + esc(m.u) : ""}</td></tr>`
    ).join("") +
      `<tr><td>KV 工作集</td><td class="v">约 320 MiB</td></tr>` +
      `<tr><td>resident IQ4_XS</td><td class="v">16.96 GiB</td></tr>`;
    panel.innerHTML = `
      <h2>fastLlama 架构总览</h2>
      <div class="badges">
        <span class="badge">${M.nodes.length} 节点</span>
        <span class="badge">${M.edges.length} 条关系</span>
        <span class="badge">${M.scenarios.length} 个场景</span>
        <span class="badge conf-high">证据 high</span>
      </div>
      <p>${esc(M.meta.target)}。这页把 <b>接入 -> 编排 -> llama core -> ggml -> 硬件</b> 串成一张可交互组件图, 并按场景播放数据流。</p>
      <p style="color:var(--dim);font-size:11px">${esc(M.meta.note)}</p>
      <div class="callout">
        <div class="t">怎么读</div>
        <div class="d">实线 = 当前代码路径; 虚线 = 实验 / 部分落地 / 规划; 移动的粒子 = 当前步骤的数据流。点击节点看职责、数字与 sourceRefs。</div>
      </div>
      <h3>关键读数 (目标口径)</h3>
      <table class="metrics-table">${rows}</table>
      <h3>图例</h3>
      ${legendHtml()}
      <h3>阅读顺序</h3>
      <ul>
        <li><b>总览</b>: 先看层与主链。</li>
        <li><b>加载与规划</b>: resident / bank / KV 怎么被决定。</li>
        <li><b>Prefill / Decode</b>: 命中、miss 与 H2D 上传。</li>
        <li><b>长上下文 KV / 三层 store / 验证回路</b>: 容量边界与证据闭环。</li>
      </ul>
      <div class="callout warn">
        <div class="t">当前边界</div>
        <div class="d">store 2a-2c 已落地, M1 未验收, M2 (use_step) 与 SSD 冷层未开工; 图中这两段标为 planned。</div>
      </div>
      <p style="color:var(--dim);font-size:10.8px;margin-top:10px">证据基线: ${esc(M.meta.branch)} / ${esc(M.meta.date)}. 数字只读 bench/ 四份文档; 本页不产生新结论。</p>
    `;
  }

  function renderNodePanel(id) {
    const n = nodeById.get(id);
    if (!n) {
      renderDefaultPanel();
      return;
    }
    const conns = M.edges
      .filter(e => e.from === id || e.to === id)
      .sort((a, b) => (a.from === id ? 0 : 1) - (b.from === id ? 0 : 1));
    const details = n.details && n.details.length
      ? `<h3>关键机制</h3><ul>${n.details.map(d => `<li>${esc(d)}</li>`).join("")}</ul>`
      : "";
    const refs = n.refs && n.refs.length
      ? `<h3>证据</h3>${refsHtml(n.refs)}`
      : "";
    const connHtml = conns.map(e => {
      const outgoing = e.from === id;
      const other = nodeById.get(outgoing ? e.to : e.from);
      return `<button data-edge="${esc(e.id)}" title="${esc(e.label)}"><span class="arrow">${outgoing ? "->" : "<-"}</span>${esc(other ? other.label : "?")}</button>`;
    }).join("");
    panel.innerHTML = `
      <h2>${esc(n.label)}</h2>
      <div class="badges">
        <span class="badge type-${n.type}">${TYPE_LABEL[n.type] || n.type}</span>
        <span class="badge st-${n.status}">${STATUS_LABEL[n.status] || n.status}</span>
        <span class="badge conf-${n.confidence}">证据 ${esc(n.confidence)}</span>
      </div>
      <p>${esc(n.desc)}</p>
      ${details}
      ${refs}
      <h3>连接 (${conns.length})</h3>
      <div class="conn">${connHtml || "<span style='color:var(--dim);font-size:11px'>无</span>"}</div>
      <p style="color:var(--dim);font-size:10.8px;margin-top:12px">sourceRefs 是仓库相对路径; 点击在新标签打开。连接按钮会打开该关系的说明。</p>
    `;
  }

  function renderEdgePanel(id) {
    const e = edgeById.get(id);
    const a = e && nodeById.get(e.from);
    const b = e && nodeById.get(e.to);
    if (!e || !a || !b) {
      renderDefaultPanel();
      return;
    }
    panel.innerHTML = `
      <h2>${esc(a.label)} <span style="color:var(--dim)">-></span> ${esc(b.label)}</h2>
      <div class="badges">
        <span class="badge">${EDGE_TYPE_LABEL[e.type] || e.type}</span>
        <span class="badge st-${e.status}">${STATUS_LABEL[e.status]}</span>
      </div>
      <p><b>${esc(e.label)}</b></p>
      <p>${EDGE_DESC[e.type] || ""}</p>
      <h3>端点</h3>
      <div class="conn">
        <button data-node="${esc(a.id)}"><span class="arrow">from</span>${esc(a.label)}</button>
        <button data-node="${esc(b.id)}"><span class="arrow">to</span>${esc(b.label)}</button>
      </div>
      <div class="callout">
        <div class="t">证据</div>
        <div class="d">关系本身没有独立 sourceRefs; 点击端点查看各自的代码 / 文档证据。</div>
      </div>
    `;
  }

  /* ---------- search / filter ---------- */

  function applySearch() {
    const q = searchInput.value.trim().toLowerCase();
    if (!q) {
      nodeEls.forEach(g => g.classList.remove("search-hit", "search-dim"));
      searchInput.style.borderColor = "";
      return [];
    }
    const hits = [];
    nodeEls.forEach((g, id) => {
      const n = nodeById.get(id);
      const hay = [n.id, n.label, n.sub, n.desc, (n.details || []).join(" "), (n.refs || []).join(" ")].join(" ").toLowerCase();
      const ok = hay.includes(q);
      g.classList.toggle("search-hit", ok);
      g.classList.toggle("search-dim", !ok);
      if (ok) hits.push(n);
    });
    searchInput.style.borderColor = hits.length ? "var(--cyan)" : "var(--red)";
    return hits;
  }

  function clearSearch() {
    searchInput.value = "";
    nodeEls.forEach(g => g.classList.remove("search-hit", "search-dim"));
    searchInput.style.borderColor = "";
  }

  function applyFilter() {
    showNonCurrent = showPlannedChk.checked;
    svg.classList.toggle("hide-noncurrent", !showNonCurrent);
    rebuildParticles();
    syncLabels();
  }

  /* ---------- tooltip ---------- */

  function showTip(html, ev) {
    tooltip.innerHTML = html;
    tooltip.hidden = false;
    moveTip(ev);
  }

  function moveTip(ev) {
    if (tooltip.hidden) return;
    const rect = canvasWrap.getBoundingClientRect();
    let x = ev.clientX - rect.left + 14;
    let y = ev.clientY - rect.top + 14;
    const w = tooltip.offsetWidth;
    const h = tooltip.offsetHeight;
    if (x + w > rect.width - 8) x = ev.clientX - rect.left - w - 14;
    if (y + h > rect.height - 8) y = ev.clientY - rect.top - h - 14;
    tooltip.style.left = Math.max(8, x) + "px";
    tooltip.style.top = Math.max(8, y) + "px";
  }

  function hideTip() {
    tooltip.hidden = true;
  }

  /* ---------- pan / zoom ---------- */

  function zoomAt(clientX, clientY, factor) {
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const p = pt.matrixTransform(ctm.inverse());
    const minW = 240;
    const maxW = 2600;
    const nw = Math.min(maxW, Math.max(minW, view.w / factor));
    const nh = nw * (view.h / view.w);
    const rx = (p.x - view.x) / view.w;
    const ry = (p.y - view.y) / view.h;
    view = { x: p.x - rx * nw, y: p.y - ry * nh, w: nw, h: nh };
    applyView();
  }

  /* ---------- bind ---------- */

  function bindUI() {
    chipsEl.addEventListener("click", ev => {
      const chip = ev.target.closest(".chip");
      if (!chip) return;
      setScenario(chip.dataset.id);
    });

    playBtn.addEventListener("click", () => {
      const sc = scenarioById.get(scenarioId);
      if (!sc || sc.steps.length <= 1) return;
      playing = !playing;
      lastStepAt = performance.now();
      updatePlayBtn();
    });
    prevBtn.addEventListener("click", () => manualStep(-1));
    nextBtn.addEventListener("click", () => manualStep(1));
    stepDots.addEventListener("click", ev => {
      const dot = ev.target.closest(".dot");
      if (!dot) return;
      stepIndex = parseInt(dot.dataset.step, 10) || 0;
      playing = false;
      lastStepAt = performance.now();
      updatePlayBtn();
      applyStep(true);
    });

    speedSel.addEventListener("change", () => {
      speed = parseFloat(speedSel.value) || 1;
      lastStepAt = performance.now();
    });

    followChk.addEventListener("change", () => {
      follow = followChk.checked;
      if (follow) {
        if (focusNodes.size) fitToNodes(Array.from(focusNodes), 90, true);
        else fitAll(true);
      }
    });

    showPlannedChk.addEventListener("change", applyFilter);

    searchInput.addEventListener("input", applySearch);
    searchInput.addEventListener("keydown", ev => {
      if (ev.key === "Escape") {
        clearSearch();
        searchInput.blur();
        return;
      }
      if (ev.key === "Enter") {
        ev.preventDefault();
        const hits = applySearch();
        if (hits.length) {
          selectNode(hits[0].id);
          fitToNodes([hits[0].id], 80, true);
        }
      }
    });

    $("#zoomIn").addEventListener("click", () => {
      stopViewAnim();
      breakFollow();
      const rect = svg.getBoundingClientRect();
      zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 1.25);
    });
    $("#zoomOut").addEventListener("click", () => {
      stopViewAnim();
      breakFollow();
      const rect = svg.getBoundingClientRect();
      zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 0.8);
    });
    $("#fitBtn").addEventListener("click", () => {
      follow = true;
      followChk.checked = true;
      fitAll(true);
    });

    panel.addEventListener("click", ev => {
      const edgeBtn = ev.target.closest("button[data-edge]");
      if (edgeBtn) {
        selectEdge(edgeBtn.dataset.edge);
        return;
      }
      const nodeBtn = ev.target.closest("button[data-node]");
      if (nodeBtn) {
        selectNode(nodeBtn.dataset.node);
      }
    });

    svg.addEventListener("click", ev => {
      if (suppressClick) {
        suppressClick = false;
        return;
      }
      if (!ev.target.closest(".node") && !ev.target.closest(".edge")) {
        selectNothing();
      }
    });

    svg.addEventListener("pointerdown", ev => {
      if (ev.button !== 0) return;
      if (ev.target.closest(".node") || ev.target.closest(".edge")) return;
      dragging = true;
      suppressClick = false;
      moved = 0;
      lastX = ev.clientX;
      lastY = ev.clientY;
      stopViewAnim();
      breakFollow();
      svg.classList.add("dragging");
      if (svg.setPointerCapture) svg.setPointerCapture(ev.pointerId);
    });

    svg.addEventListener("pointermove", ev => {
      if (!dragging) return;
      const dx = ev.clientX - lastX;
      const dy = ev.clientY - lastY;
      lastX = ev.clientX;
      lastY = ev.clientY;
      const sx = view.w / (svg.clientWidth || 1);
      const sy = view.h / (svg.clientHeight || 1);
      view.x -= dx * sx;
      view.y -= dy * sy;
      applyView();
      moved += Math.abs(dx) + Math.abs(dy);
      if (moved > 4) suppressClick = true;
    });

    const endDrag = () => {
      dragging = false;
      svg.classList.remove("dragging");
    };
    svg.addEventListener("pointerup", endDrag);
    svg.addEventListener("pointercancel", endDrag);

    svg.addEventListener("wheel", ev => {
      ev.preventDefault();
      stopViewAnim();
      breakFollow();
      const factor = Math.exp(-ev.deltaY * 0.0015);
      zoomAt(ev.clientX, ev.clientY, Math.min(2, Math.max(0.5, factor)));
    }, { passive: false });

    window.addEventListener("keydown", ev => {
      const tag = (ev.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "select" || tag === "textarea") return;
      if (ev.key === " ") {
        ev.preventDefault();
        playBtn.click();
        return;
      }
      if (ev.key === "ArrowRight") {
        ev.preventDefault();
        manualStep(1);
        return;
      }
      if (ev.key === "ArrowLeft") {
        ev.preventDefault();
        manualStep(-1);
        return;
      }
      if (ev.key === "Escape") {
        selectNothing();
        return;
      }
      const n = parseInt(ev.key, 10);
      if (n >= 1 && n <= M.scenarios.length) {
        setScenario(M.scenarios[n - 1].id);
      }
    });

    let resizeTimer = 0;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (follow) {
          if (focusNodes.size) fitToNodes(Array.from(focusNodes), 90, false);
          else fitAll(false);
        }
      }, 140);
    });

    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) lastStepAt = performance.now();
    });
  }

  /* ---------- loop ---------- */

  let lastFrame = performance.now();
  function tick(now) {
    if (playing) {
      const sc = scenarioById.get(scenarioId);
      if (sc && sc.steps.length > 1 && now - lastStepAt > 6200 / speed) {
        lastStepAt = now;
        stepIndex = (stepIndex + 1) % sc.steps.length;
        applyStep(true);
      }
    }
    updateParticles(now);
    requestAnimationFrame(tick);
  }

  /* ---------- init ---------- */

  function init() {
    renderBands();
    renderNotes();
    renderEdges();
    renderNodes();
    renderMetrics();

    const hash = decodeURIComponent((location.hash || "").replace(/^#/, ""));
    if (scenarioById.has(hash)) scenarioId = hash;
    stepIndex = 0;
    renderChips();
    bindUI();
    applyStep(false);
    updatePlayBtn();
    renderDefaultPanel();

    requestAnimationFrame(() => {
      if (follow && focusNodes.size) {
        fitToNodes(Array.from(focusNodes), 90, false);
      } else {
        fitAll(false);
      }
      requestAnimationFrame(tick);
    });
  }

  init();
})();

(function () {
  "use strict";

  const M = window.DDD_MODEL;
  if (!M) {
    document.body.insertAdjacentHTML("afterbegin", "<p style='padding:20px;color:#ff6b6b'>ddd-model.js 未加载</p>");
    return;
  }

  const NS = "http://www.w3.org/2000/svg";
  const SANS = '"Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, sans-serif';
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const TYPE_COLOR = {
    core: "#b388ff",
    supporting: "#6ea8fe",
    generic: "#ffb86b",
    aggregate: "#c9a7ff",
    value: "#7ee787",
    service: "#6ea8fe",
    usecase: "#4cc9f0",
    port: "#f4a261",
    adapter: "#ffb86b",
    event: "#f778ba",
    invariant: "#ff6b6b",
    shared: "#9fb0c2",
    repository: "#c9a7ff"
  };
  const TYPE_LABEL = {
    core: "核心域",
    supporting: "支撑域",
    generic: "通用子域",
    aggregate: "聚合根",
    value: "值对象",
    service: "领域服务",
    usecase: "应用服务",
    port: "端口",
    adapter: "适配器",
    event: "领域事件",
    invariant: "不变量",
    shared: "共享内核",
    repository: "仓储"
  };
  const STATUS_LABEL = {
    current: "当前代码",
    partial: "部分对应",
    proposed: "DDD 提议",
    planned: "未开工",
    experimental: "实验项"
  };
  const STATUS_COLOR = {
    current: "#7ee787",
    partial: "#f4a261",
    proposed: "#6ea8fe",
    planned: "#9fb0c2",
    experimental: "#f4a261"
  };
  const EDGE_TYPE_LABEL = {
    cs: "Customer-Supplier",
    upstream: "Upstream / Downstream",
    acl: "Anti-Corruption Layer",
    observer: "Observer",
    published: "Published Language",
    shared: "Shared Kernel",
    event: "Domain Event",
    contract: "Contract / Invariant"
  };
  const EDGE_DESC = {
    cs: "上游向下游提供能力, 下游依赖上游的契约; 双方可以独立演进。",
    upstream: "上游提供事实/模型, 下游消费; 上游不关心下游的执行细节。",
    acl: "防腐层: 把外部模型/设备概念翻译成本上下文的语言, 隔离变化。",
    observer: "观察者: 只读事件与指标, 不反向驱动运行时状态。",
    published: "发布语言: 双方共享的稳定契约, 例如 GGUF / ModelArtifact。",
    shared: "共享内核: 两个上下文共享的最小代码; 变更需要双方同意。",
    event: "领域事件: 对已发生事实的发布与订阅。",
    contract: "聚合/服务/端口之间的显式契约; 不是每 token 的调用序列。"
  };
  const CONTEXT_MAP_TYPES = new Set(["cs", "upstream", "acl", "observer", "published", "shared"]);

  const $ = (sel, root) => (root || document).querySelector(sel);
  const svg = $("#diagram");
  const gBands = $("#bands");
  const gEdges = $("#edges");
  const gParticles = $("#particles");
  const gNodes = $("#nodes");
  const gEdgeLabels = $("#edgeLabels");
  const gContexts = svgEl("g", { id: "contexts" }, null);
  gEdges.parentNode.insertBefore(gContexts, gEdges);
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

  let scenarioId = "contextmap";
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
  let view = { x: 0, y: 0, w: 1800, h: 1250 };
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

  function anchorSide(n, side, offset) {
    const d = offset || 0;
    if (side === "top") return { x: n.x + n.w / 2 + d, y: n.y, n: "t" };
    if (side === "bottom") return { x: n.x + n.w / 2 + d, y: n.y + n.h, n: "b" };
    if (side === "left") return { x: n.x, y: n.y + n.h / 2 + d, n: "l" };
    if (side === "right") return { x: n.x + n.w, y: n.y + n.h / 2 + d, n: "r" };
    return null;
  }

  function pathD(a, b, edge) {
    if (edge && edge.route) {
      const p = anchorSide(a, edge.fromSide, edge.fromOffset) || { x: a.x + a.w / 2, y: a.y + a.h / 2 };
      const q = anchorSide(b, edge.toSide, edge.toOffset) || { x: b.x + b.w / 2, y: b.y + b.h / 2 };
      const pts = [p, ...edge.route.map(pt => ({ x: pt[0], y: pt[1] })), q];
      return pts.map((pt, i) => `${i === 0 ? "M" : "L"} ${pt.x} ${pt.y}`).join(" ");
    }
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
      const path = svgEl("path", { class: "path", d: pathD(a, b, edge) }, g);
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
    for (const edge of M.edges) {
      const rt = edgeRuntime.get(edge.id);
      if (!rt) continue;
      const hidden = !showNonCurrent && (edge.status === "planned" || edge.status === "experimental");
      const isSelected = selected && selected.type === "edge" && selected.id === edge.id;
      const showActive = activeEdges.has(edge.id) && !CONTEXT_MAP_TYPES.has(edge.type);
      const show = !hidden && (isSelected || hoveredEdge === edge.id || showActive);
      rt.labelEl.classList.toggle("show", !!show);
    }
  }

  function addKindBadge(g, n, text, cls) {
    const t = addText(g, n.x + n.w - 10, n.y + n.h - 8, text, cls || "kind-badge");
    t.setAttribute("text-anchor", "end");
    return t;
  }

  function drawContext(n, g) {
    svgEl("rect", { class: "card", x: n.x, y: n.y, width: n.w, height: n.h, rx: 14 }, g);
    svgEl("rect", { class: "ctx-header", x: n.x + 1, y: n.y + 1, width: n.w - 2, height: 52, rx: 13 }, g);
    addText(g, n.x + 16, n.y + 24, n.label, "ctx-label");
    addText(g, n.x + 16, n.y + 42, n.sub, "ctx-sub");
    const badge = n.type === "core" ? "CORE DOMAIN" : n.type === "supporting" ? "SUPPORTING" : "GENERIC";
    const t = addText(g, n.x + n.w - 14, n.y + 22, badge, "ctx-badge");
    t.setAttribute("text-anchor", "end");
    if (n.status !== "current") {
      svgEl("circle", {
        cx: n.x + n.w - 15, cy: n.y + 40, r: 3,
        fill: STATUS_COLOR[n.status] || "#9fb0c2", opacity: 0.9
      }, g);
    }
  }

  function drawValue(n, g) {
    svgEl("rect", { class: "card", x: n.x, y: n.y, width: n.w, height: n.h, rx: n.h / 2 }, g);
    const t = addText(g, n.x + n.w / 2, n.y + n.h / 2 + 4, n.label, "value-label");
    t.setAttribute("text-anchor", "middle");
  }

  function drawEvent(n, g) {
    svgEl("rect", { class: "card", x: n.x, y: n.y, width: n.w, height: n.h, rx: 8 }, g);
    addText(g, n.x + 12, n.y + 19, n.label, "event-label");
    addText(g, n.x + 12, n.y + 34, n.sub, "event-sub");
    addKindBadge(g, n, "EVT");
  }

  function drawInvariant(n, g) {
    svgEl("rect", { class: "card", x: n.x, y: n.y, width: n.w, height: n.h, rx: 8 }, g);
    svgEl("rect", {
      x: n.x + 1, y: n.y + 7, width: 3, height: Math.max(6, n.h - 14), rx: 1.5,
      fill: "#ff6b6b", opacity: 0.9
    }, g);
    addText(g, n.x + 14, n.y + 20, n.label, "inv-label");
    addText(g, n.x + 14, n.y + 36, n.sub, "inv-sub");
    addKindBadge(g, n, "INV");
  }

  function drawStandard(n, g) {
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

    if (n.type === "aggregate") {
      const tag = n.invariants && n.invariants.length ? `AGG / INV ${n.invariants.length}` : "AGG";
      addKindBadge(g, n, tag, "agg-badge");
    } else if (n.type === "shared") {
      addKindBadge(g, n, "SHARED KERNEL");
    } else if (n.type === "repository") {
      addKindBadge(g, n, "REPOSITORY");
    }
  }

  function renderNodes() {
    const ordered = M.nodes.slice().sort((a, b) => (a.kind === "context" ? 0 : 1) - (b.kind === "context" ? 0 : 1));
    for (const n of ordered) {
      const parent = n.kind === "context" ? gContexts : gNodes;
      const g = svgEl("g", {
        class: "node",
        "data-id": n.id,
        "data-type": n.type,
        "data-kind": n.kind || "rect",
        "data-status": n.status,
        tabindex: "0",
        role: "button",
        "aria-label": n.label
      }, parent);

      if (n.kind === "context") drawContext(n, g);
      else if (n.kind === "value") drawValue(n, g);
      else if (n.kind === "event") drawEvent(n, g);
      else if (n.kind === "invariant") drawInvariant(n, g);
      else drawStandard(n, g);

      nodeEls.set(n.id, g);

      const tip = () => `<b>${esc(n.label)}</b>${esc(n.sub)}<br><span style="color:var(--dim)">${TYPE_LABEL[n.type] || n.type} / ${STATUS_LABEL[n.status] || n.status} - 点击查看 DDD 角色</span>`;
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
      x: 1020, y: 640, w: 400, h: 50,
      title: "热路径原则",
      lines: ["不引入每 token 抽象; 端口保持窄"]
    },
    {
      x: 1500, y: 555, w: 240, h: 100,
      title: "Adapter 规则",
      lines: ["Vulkan / CPU / SSD 是 adapter", "ACL 隐藏 prefetch 细节"]
    },
    {
      x: 1230, y: 1095, w: 500, h: 90,
      title: "Observer 规则",
      lines: ["events -> metrics -> ledger", "观察者不反向驱动运行时", "结论只进 bench/ 四份文档"]
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
    if (!duration || document.hidden) {
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
    fitBox(0, 0, 1800, 1250, 8, animate);
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
    if (x1 - x0 > 1650 || y1 - y0 > 1100) {
      fitAll(animate);
      return;
    }
    fitBox(x0, y0, x1, y1, pad == null ? 60 : pad, animate);
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
    scenarioId = scenarioById.has(id) ? id : "contextmap";
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
    focusNodes = scenarioId === "contextmap"
      ? new Set(M.nodes.map(n => n.id))
      : new Set(step.f || []);
    for (const id of Array.from(focusNodes)) {
      const n = nodeById.get(id);
      if (n && n.context) focusNodes.add(n.context);
    }
    activeEdges = new Set(step.e || []);
    trailEdges = new Set();
    for (let i = 0; i < stepIndex; i++) {
      for (const id of (sc.steps[i].e || [])) trailEdges.add(id);
    }

    const scenarioMode = scenarioId !== "contextmap";
    nodeEls.forEach((g, id) => {
      g.classList.toggle("focus", scenarioMode && focusNodes.has(id));
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
      if (scenarioId === "contextmap") {
        fitAll(animate);
      } else if (focusNodes.size) {
        fitToNodes(Array.from(focusNodes), 60, animate);
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
        <div class="item"><span class="swatch" style="background:var(--violet)"></span>核心域</div>
        <div class="item"><span class="swatch" style="background:var(--common)"></span>支撑域</div>
        <div class="item"><span class="swatch" style="background:var(--ggml)"></span>通用子域</div>
        <div class="item"><span class="swatch" style="background:#c9a7ff"></span>聚合根</div>
        <div class="item"><span class="swatch" style="background:var(--green)"></span>值对象</div>
        <div class="item"><span class="swatch" style="background:var(--blue)"></span>领域 / 应用服务</div>
        <div class="item"><span class="swatch" style="background:var(--amber)"></span>端口</div>
        <div class="item"><span class="swatch" style="background:var(--orange)"></span>适配器</div>
        <div class="item"><span class="swatch" style="background:var(--pink)"></span>领域事件</div>
        <div class="item"><span class="swatch" style="background:var(--red)"></span>不变量</div>
        <div class="item"><span class="line-sample" style="border-top-color:var(--blue)"></span>Customer-Supplier</div>
        <div class="item"><span class="line-sample" style="border-top-style:dashed;border-top-color:var(--amber)"></span>ACL / upstream</div>
        <div class="item"><span class="line-sample" style="border-top-style:dotted;border-top-color:var(--violet)"></span>Observer</div>
        <div class="item"><span class="line-sample" style="border-top-color:var(--cyan)"></span>Published language</div>
        <div class="item"><span class="line-sample" style="border-top-style:dashed;border-top-color:#9fb0c2"></span>Shared kernel</div>
        <div class="item"><span class="line-sample" style="border-top-style:dotted;border-top-color:var(--pink)"></span>Domain event</div>
      </div>`;
  }

  function renderDefaultPanel() {
    const rows = M.metrics.map(m =>
      `<tr><td>${esc(m.k)}</td><td class="v">${esc(m.v)}${m.u ? " " + esc(m.u) : ""}</td></tr>`
    ).join("");
    const contexts = M.nodes.filter(n => n.kind === "context").length;
    panel.innerHTML = `
      <h2>DDD 重画: 上下文地图</h2>
      <div class="badges">
        <span class="badge">${contexts} 上下文</span>
        <span class="badge">${M.nodes.length} 节点</span>
        <span class="badge">${M.edges.length} 条关系</span>
        <span class="badge">${M.scenarios.length} 个场景</span>
      </div>
      <p>这是 <b>proposed DDD 视图</b>, 不是当前 API 契约。它把容量、驻留、KV、执行与证据重画成限界上下文、聚合、策略与事件, 并保留到当前代码的映射。</p>
      <p style="color:var(--dim);font-size:11px">${esc(M.meta.note)}</p>
      <div class="callout">
        <div class="t">怎么读状态</div>
        <div class="d"><b>current</b> = 当前代码已有明确实现; <b>partial</b> = 已有雏形但没有显式边界/契约; <b>proposed</b> = DDD 提议的新概念; <b>planned</b> = 未开工 (例如 SSD 冷层)。</div>
      </div>
      <h3>关键计数</h3>
      <table class="metrics-table">${rows}</table>
      <h3>图例</h3>
      ${legendHtml()}
      <h3>上下文关系</h3>
      <ul>
        <li>Model Acquisition -> Inference Execution: Published Language</li>
        <li>Model Acquisition -> Residency & Capacity: upstream geometry / quant</li>
        <li>Conversation & Delivery -> Inference Execution: Customer-Supplier</li>
        <li>Inference Execution -> Residency & Capacity: ResidencyPort</li>
        <li>Residency & Capacity -> Backend Runtime: Customer-Supplier + ACL</li>
        <li>Experiment & Evidence -> core contexts: observer</li>
      </ul>
      <h3>阅读顺序</h3>
      <ul>
        <li><b>上下文地图</b>: 六个上下文 + 共享内核。</li>
        <li><b>LoadAndPlan</b>: ModelArtifact -> ResidencyPlan -> 专家/KV 配额。</li>
        <li><b>Prefill / DecodeStep</b>: 用例 -> 执行 -> 驻留 -> 后端。</li>
        <li><b>Degrade / SwitchTier</b>: 预算、降级与 SSD 冷层。</li>
        <li><b>Evidence / 边界原则</b>: 观察者回路与“不重写热路径”。</li>
      </ul>
      <div class="callout warn">
        <div class="t">设计边界</div>
        <div class="d">只把容量、驻留、降级规则提升为聚合/策略; 每 token 的图构建、调度与 kernel 保持过程式。proposed 概念要成为契约, 还需要接口与 ADR, 本页不替代它们。</div>
      </div>
      <p style="font-size:11px">对照当前技术视图: <a href="fastllama-architecture.html">fastllama-architecture.html</a></p>
      <p style="font-size:11px">改造计划与收益矩阵: <a href="ddd-refactor-plan.md">ddd-refactor-plan.md</a></p>
      <p style="color:var(--dim);font-size:10.8px;margin-top:10px">证据基线: ${esc(M.meta.branch)} / ${esc(M.meta.date)}. 所有映射指向当前代码或 bench/ 文档。</p>
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
    const parent = n.context ? nodeById.get(n.context) : null;
    const details = n.details && n.details.length
      ? `<h3>DDD 角色 / 机制</h3><ul>${n.details.map(d => `<li>${esc(d)}</li>`).join("")}</ul>`
      : "";
    const inv = n.invariants && n.invariants.length
      ? `<h3>不变量</h3><ul>${n.invariants.map(d => `<li>${esc(d)}</li>`).join("")}</ul>`
      : "";
    const refs = n.refs && n.refs.length
      ? `<h3>当前代码映射</h3>${refsHtml(n.refs)}`
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
        ${parent ? `<span class="badge">in ${esc(parent.label)}</span>` : ""}
      </div>
      <p>${esc(n.desc)}</p>
      ${inv}
      ${details}
      ${refs}
      <h3>连接 (${conns.length})</h3>
      <div class="conn">${connHtml || "<span style='color:var(--dim);font-size:11px'>无</span>"}</div>
      <p style="color:var(--dim);font-size:10.8px;margin-top:12px">proposed 是 DDD 提议, 不是当前 API; current / partial 节点给出当前代码映射。连接按钮打开关系说明。</p>
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
        <div class="t">关系语义</div>
        <div class="d">上下文关系是 proposed 契约; 当前代码映射在两端节点。箭头不代表每 token 调用序列。</div>
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
      const hay = [n.id, n.label, n.sub, n.desc, (n.details || []).join(" "), (n.invariants || []).join(" "), (n.refs || []).join(" "), n.context || ""].join(" ").toLowerCase();
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
      if (follow && focusNodes.size && scenarioId !== "contextmap") {
        fitToNodes(Array.from(focusNodes), 60, false);
      } else {
        fitAll(false);
      }
      requestAnimationFrame(tick);
    });
  }

  init();
})();

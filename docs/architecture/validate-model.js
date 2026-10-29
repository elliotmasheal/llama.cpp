#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

const modelFile = process.argv[2] || "architecture-model.js";
const modelPath = path.isAbsolute(modelFile) ? modelFile : path.join(__dirname, modelFile);
global.window = {};
require(modelPath);
const M = global.window.ARCH_MODEL || global.window.DDD_MODEL;
const ROOT = path.resolve(__dirname, "..", "..");
const errors = [];
const NODE_STATUS = ["current", "experimental", "planned", "partial", "proposed"];
const EDGE_STATUS = ["current", "experimental", "planned", "partial", "proposed"];

if (!M) {
  console.error(`FAIL: ${modelFile} did not define ARCH_MODEL or DDD_MODEL`);
  process.exit(1);
}

const nodeIds = new Set();
for (const n of M.nodes) {
  if (nodeIds.has(n.id)) errors.push(`duplicate node id: ${n.id}`);
  nodeIds.add(n.id);
  if (!n.desc || !n.refs || !n.refs.length) errors.push(`node without desc/refs: ${n.id}`);
  if (!NODE_STATUS.includes(n.status)) errors.push(`bad node status: ${n.id}`);
  if (!["high", "medium", "low", "unknown"].includes(n.confidence)) errors.push(`bad confidence: ${n.id}`);
}

const edgeIds = new Set();
for (const e of M.edges) {
  if (edgeIds.has(e.id)) errors.push(`duplicate edge id: ${e.id}`);
  edgeIds.add(e.id);
  if (!nodeIds.has(e.from)) errors.push(`edge from missing node: ${e.id} -> ${e.from}`);
  if (!nodeIds.has(e.to)) errors.push(`edge to missing node: ${e.id} -> ${e.to}`);
  if (!EDGE_STATUS.includes(e.status)) errors.push(`bad edge status: ${e.id}`);
}

const refs = new Set();
for (const n of M.nodes) {
  for (const ref of n.refs) {
    const file = ref.replace(/:\d+(?:-\d+)?$/, "");
    refs.add(file);
    if (!fs.existsSync(path.join(ROOT, file))) errors.push(`missing ref: ${n.id} -> ${ref}`);
  }
}

for (const s of M.scenarios) {
  if (!s.steps || !s.steps.length) errors.push(`scenario without steps: ${s.id}`);
  for (const step of s.steps || []) {
    if (!step.t) errors.push(`scenario step without text: ${s.id}`);
    for (const id of step.f || []) if (!nodeIds.has(id)) errors.push(`step focus missing node: ${s.id} -> ${id}`);
    for (const id of step.e || []) if (!edgeIds.has(id)) errors.push(`step edge missing edge: ${s.id} -> ${id}`);
  }
}

if (errors.length) {
  console.error(`FAIL: ${errors.length} problem(s)`);
  for (const e of errors) console.error("  " + e);
  process.exit(1);
}

console.log(`${path.basename(modelFile)}: OK: ${M.nodes.length} nodes, ${M.edges.length} edges, ${M.scenarios.length} scenarios, ${refs.size} evidence paths`);

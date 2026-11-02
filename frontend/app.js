"use strict";

const $ = (id) => document.getElementById(id);
const state = { events: [], busy: false, tab: "trace", runId: null };

const SUGGESTIONS = [
  "Find a plumber for my kitchen tap this week, mornings only, under ₹1,500.",
  "Move my AC servicing to Saturday.",
  "Book the same electrician as last time, but not before 10 a.m.",
  "Cancel my AC servicing",
  "What do you remember about me?",
];

// events hidden in "key events only" mode
const NOISY = new Set(["request_updated", "options_built", "turn_end", "memory_infer_skipped", "approval_checked"]);

// ---------------------------------------------------------------- api

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

// ---------------------------------------------------------------- chat

function addMessage(role, text, extraClass = "") {
  const empty = $("messages").querySelector(".empty");
  if (empty) empty.remove();
  const el = document.createElement("div");
  el.className = `msg ${role} ${extraClass}`.trim();
  el.textContent = text;
  $("messages").appendChild(el);
  $("messages").scrollTop = $("messages").scrollHeight;
  return el;
}

function replyClass(text) {
  if (text.startsWith("Please confirm") || text.includes("\nPlease confirm") || text.includes("(yes/no)")) return "confirm";
  if (/verified/i.test(text)) return "success";
  if (/^I had to stop|Something went wrong|couldn't book|conflicts with/i.test(text)) return "stop";
  return "";
}

function showEmpty() {
  $("messages").innerHTML = "";
  const el = document.createElement("div");
  el.className = "empty";
  el.innerHTML = "<p><b>Try a request below.</b></p><p>Every booking, cancel or reschedule needs your explicit “yes”. " +
    "Watch the trace on the right as the agent works.</p>";
  $("messages").appendChild(el);
}

async function send(text) {
  if (state.busy || !text.trim()) return;
  state.busy = true;
  $("send").disabled = true;
  $("input").value = "";
  addMessage("user", text);
  const typing = document.createElement("div");
  typing.className = "typing";
  typing.textContent = "agent is working";
  $("messages").appendChild(typing);
  $("messages").scrollTop = $("messages").scrollHeight;
  try {
    const data = await api("/api/chat", { message: text });
    typing.remove();
    addMessage("agent", data.reply, replyClass(data.reply));
    setPhase(data.phase);
  } catch (e) {
    typing.remove();
    addMessage("system", "Server error: " + e.message, "stop");
  } finally {
    state.busy = false;
    $("send").disabled = false;
    $("input").focus();
    await pollEvents();
    refreshSideTab();
  }
}

// ---------------------------------------------------------------- trace

const fmtArgs = (o) => Object.entries(o || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
const money = (n) => (n === null || n === undefined ? "" : `₹${n}`);

function summarize(e) {
  switch (e.kind) {
    case "turn_start": return `USER › ${e.user_message}`;
    case "turn_end": return `agent replied · ${e.model_calls} model / ${e.tool_calls} tool calls`;
    case "run_start": return `run ${e.run_id} · user ${e.user_id} · resumed phase ${e.resumed_phase} · guard ${e.guard}`;
    case "run_end": return e.reason || "run ended";
    case "state_transition": return `${e.from_phase} → ${e.to_phase}  (${e.reason})`;
    case "model_call": {
      const it = Object.entries(e.interpretation || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" ");
      const cost = e.cost_usd !== null && e.cost_usd !== undefined ? ` · $${e.cost_usd}` : "";
      return `${e.model} [${e.tier}: ${e.route_reason}] · ${e.latency_ms}ms · ${e.input_tokens ?? "?"}/${e.output_tokens ?? "?"} tok${cost} → ${it}`;
    }
    case "tool_call": {
      const head = `${e.tool}(${fmtArgs(e.args)}) · ${e.latency_ms}ms → `;
      if (!e.ok) return [head, { cls: "err-tag", text: `ERR ${e.error}` }];
      const r = e.result;
      if (e.tool === "search_providers" && Array.isArray(r)) return head + `${r.length} providers`;
      if (e.tool === "list_bookings" && Array.isArray(r)) return head + `${r.length} bookings`;
      if (e.tool === "get_provider") return head + `${r.name} · ${(r.reviews || []).length} reviews`;
      return [head, { cls: "ok-tag", text: JSON.stringify(r) }];
    }
    case "approval_proposed":
    case "approval_granted":
      return `${e.approval_id} · ${e.action} ${e.provider_id} @ ${e.slot} · ${money(e.price)} · fp ${e.fingerprint}`;
    case "approval_checked": return `${e.approval_id} · fp ${e.fingerprint} ✓ gateway re-check`;
    case "approval_voided": return `${e.approval_id} · ${e.reason}` + (e.new_price ? ` (₹${e.approved_price} → ₹${e.new_price})` : "");
    case "approval_rejected": return `${e.approval_id} · ${e.reason}`;
    case "booking_status_checked": return `after ${e.after}${e.attempt ? " (attempt " + e.attempt + ")" : ""} → ${e.status}`;
    case "booking_verified": return `${e.booking_id} confirmed by get_booking_status`;
    case "guard_check":
      return `${e.provider_id} · ${e.reviews} reviews · flagged ${JSON.stringify(e.flagged)} · ${e.guard}` +
        (e.scores && e.scores.some((x) => x !== null) ? ` · scores ${JSON.stringify(e.scores)}` : "");
    case "ask": return `asking for: ${e.fields.join(", ")}`;
    case "memory_write": return `${e.key} = ${e.value} (${e.source})`;
    case "limit_exceeded": return `${e.limit} cap ${e.cap}` + (e.elapsed_s ? ` (elapsed ${e.elapsed_s}s)` : "");
    case "error": return `${e.error_type}: ${e.error}`;
    default: {
      const rest = Object.fromEntries(Object.entries(e).filter(([k]) => !["run_id", "seq", "ts", "kind"].includes(k)));
      return JSON.stringify(rest);
    }
  }
}

function kindClass(k) {
  if (k === "turn_start") return "turn";
  if (k === "tool_call") return "c-tool";
  if (k === "model_call") return "c-model";
  if (k === "state_transition") return "c-state";
  if (/verified$|^approval_granted$|^approval_checked$/.test(k)) return "c-ok";
  if (/^error$|limit_exceeded|approval_voided|approval_rejected/.test(k)) return "c-err";
  if (k === "guard_check") return "c-guard";
  if (k.startsWith("approval")) return "c-approval";
  if (k === "turn_end" || k === "run_end") return "c-dim";
  return "";
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function eventRow(e, flash) {
  const row = document.createElement("div");
  row.className = `ev ${kindClass(e.kind)}${flash ? " flash" : ""}`;
  // summarize() returns plain text or [text | {cls, text}] parts; everything is escaped here
  const parts = [].concat(summarize(e));
  const body = parts.map((p) => typeof p === "string" ? escapeHtml(p)
    : `<span class="${p.cls}">${escapeHtml(p.text)}</span>`).join("");
  row.innerHTML = `<span class="t">${escapeHtml(e.ts.slice(11, 19))}</span><span class="k">${escapeHtml(e.kind)}</span>` +
    `<span class="b">${body}</span>`;
  row.addEventListener("click", () => {
    const pre = row.querySelector("pre");
    if (pre) return pre.remove();
    const p = document.createElement("pre");
    p.textContent = JSON.stringify(e, null, 2);
    row.appendChild(p);
  });
  return row;
}

function visible(e) {
  return !$("compact").checked || !NOISY.has(e.kind);
}

function renderTrace() {
  const box = $("tab-trace");
  box.innerHTML = "";
  state.events.filter(visible).forEach((e) => box.appendChild(eventRow(e, false)));
  box.scrollTop = box.scrollHeight;
  $("trace-count").textContent = state.events.length;
}

async function pollEvents() {
  try {
    const data = await api(`/api/events?after=${state.events.length}`);
    if (data.total < state.events.length) {  // server was reset
      state.events = [];
      return renderTrace();
    }
    if (!data.events.length) return;
    const box = $("tab-trace");
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
    for (const e of data.events) {
      state.events.push(e);
      if (visible(e)) box.appendChild(eventRow(e, true));
    }
    $("trace-count").textContent = state.events.length;
    if (atBottom) box.scrollTop = box.scrollHeight;
  } catch (_) { /* server restarting; next poll retries */ }
}

// ---------------------------------------------------------------- side tabs

function table(headers, rows) {
  const t = document.createElement("table");
  t.innerHTML = "<thead><tr>" + headers.map((h) => `<th>${h}</th>`).join("") + "</tr></thead>";
  const tb = document.createElement("tbody");
  rows.forEach((cells) => {
    const tr = document.createElement("tr");
    tr.innerHTML = cells.join("");
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  return t;
}

async function renderBookings() {
  const box = $("tab-bookings");
  const rows = await api("/api/bookings");
  box.innerHTML = '<div class="note">Read straight from the mock provider’s database: the source of truth for “no duplicates”.</div>';
  box.appendChild(table(["Booking", "Service", "Provider", "Slot", "Price", "Status"], rows.map((b) => [
    `<td class="mono">${escapeHtml(b.id)}</td>`, `<td>${escapeHtml(b.service.replace("_", " "))}</td>`,
    `<td class="mono">${escapeHtml(b.provider)}</td>`, `<td class="mono">${escapeHtml(b.slot.replace("T", " "))}</td>`,
    `<td>₹${b.price}</td>`, `<td><span class="pill ${escapeHtml(b.status)}">${escapeHtml(b.status)}</span></td>`])));
}

async function renderMemory() {
  const box = $("tab-memory");
  const facts = await api("/api/memory");
  box.innerHTML = '<div class="note">Inferred facts are never used until confirmed. Forgotten facts stay as tombstones so they can’t be re-inferred.</div>';
  if (!facts.length) {
    box.insertAdjacentHTML("beforeend", '<div class="note">No saved preferences yet.</div>');
    return;
  }
  box.appendChild(table(["Key", "Value", "Source", "Confirmed", "Status"], facts.map((f) => [
    `<td class="mono">${escapeHtml(f.key)}</td>`, `<td class="mono">${escapeHtml(f.value || "—")}</td>`,
    `<td><span class="pill ${f.source}">${f.source}</span></td>`, `<td>${f.confirmed ? "yes" : "no"}</td>`,
    `<td><span class="pill ${f.status}">${f.status}</span></td>`])));
}

function refreshSideTab() {
  if (state.tab === "bookings") renderBookings().catch(() => {});
  if (state.tab === "memory") renderMemory().catch(() => {});
}

function switchTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  ["trace", "bookings", "memory"].forEach((t) => $(`tab-${t}`).classList.toggle("hidden", t !== tab));
  $("filter-wrap").style.visibility = tab === "trace" ? "visible" : "hidden";
  refreshSideTab();
}

// ---------------------------------------------------------------- status & controls

function setPhase(phase) {
  $("phase").textContent = phase;
  $("phase-chip").className = "chip " + phase;
}

function applyStatus(s) {
  setPhase(s.phase);
  $("guard").textContent = s.guard;
  $("user-label").textContent = `user ${s.user_id}`;
  $("trace-file").textContent = `trace file: ${s.trace_file}`;
  if (s.scenarios) {
    $("scenario").innerHTML = s.scenarios.map((x) => `<option value="${x}">${x.replaceAll("_", " ")}</option>`).join("");
  }
  $("scenario").value = s.scenario;
  state.runId = s.run_id;
}

async function restart() {
  const s = await api("/api/restart", {});
  applyStatus(s);
  addMessage("system", `Agent restarted on the same saved state (new run ${s.run_id}). Pending work is reconciled on the next message.`);
  await pollEvents();
}

async function reset() {
  const s = await api("/api/reset", { scenario: $("scenario").value });
  applyStatus(s);
  state.events = [];
  renderTrace();
  showEmpty();
  addMessage("system", `New world · scenario “${s.scenario.replaceAll("_", " ")}”`);
  await pollEvents();
  refreshSideTab();
}

async function init() {
  SUGGESTIONS.forEach((text) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    b.addEventListener("click", () => { $("input").value = text; $("input").focus(); });
    $("suggestions").appendChild(b);
  });
  $("composer").addEventListener("submit", (ev) => { ev.preventDefault(); send($("input").value); });
  $("restart").addEventListener("click", () => restart().catch((e) => addMessage("system", e.message, "stop")));
  $("reset").addEventListener("click", () => reset().catch((e) => addMessage("system", e.message, "stop")));
  $("scenario").addEventListener("change", () => addMessage("system", "Scenario takes effect when you click “New world”."));
  $("compact").addEventListener("change", renderTrace);
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));

  showEmpty();
  applyStatus(await api("/api/status"));
  for (const m of await api("/api/transcript")) {
    addMessage(m.role, m.text, m.role === "agent" ? replyClass(m.text) : "");
  }
  await pollEvents();
  setInterval(pollEvents, 400);
  $("input").focus();
}

init();

"use strict";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const TYPE_NAMES = { K: "Knowledge & skills", A: "Ability & aptitude", P: "Personality & behaviour", B: "Situational judgement", S: "Simulation", C: "Competencies", D: "Development & 360", E: "Assessment exercises" };

// ---------- Views ----------
function show(view) {
  $("#view-find").hidden = view !== "find";
  $("#view-accuracy").hidden = view !== "accuracy";
  document.querySelectorAll(".nav a[data-view]").forEach((a) => a.toggleAttribute("aria-current", a.dataset.view === view));
  if (view === "accuracy") loadAccuracy();
}
document.querySelectorAll(".nav a[data-view]").forEach((a) => a.addEventListener("click", (e) => {
  e.preventDefault();
  history.replaceState(null, "", a.dataset.view === "find" ? "#" : `#${a.dataset.view}`);
  show(a.dataset.view);
}));

// ---------- Recommend ----------
let lastQuery = "";
async function recommend(query, overrides = null) {
  const btn = $("#go");
  btn.disabled = true;
  btn.setAttribute("aria-busy", "true");
  $("#rerun").disabled = true;
  $("#error").hidden = true;
  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(overrides ? { query, overrides } : { query }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(Array.isArray(body.detail) ? "Write at least a few words about the role." : body.detail || `Request failed (${res.status}).`);
    lastQuery = query;
    render(body);
  } catch (err) {
    $("#error").textContent = err.message || "Couldn't reach the server. Try again.";
    $("#error").hidden = false;
  } finally {
    btn.disabled = false;
    btn.removeAttribute("aria-busy");
    $("#rerun").disabled = false;
  }
}

function render(data) {
  const req = data.requirements;
  $("#results").hidden = false;
  $("#summary").textContent = req.summary;
  $("#skills").innerHTML = (req.skills.length ? req.skills : ["taken from your description"]).map((s) => `<span class="chip">${esc(s)}</span>`).join("");
  $("#max-minutes").value = req.max_minutes ?? "";
  $("#job-level").value = req.job_level ?? "";
  $("#remote").checked = !!req.remote_testing;
  const f = $("#fetched");
  f.hidden = !data.fetched_from;
  if (data.fetched_from) f.innerHTML = `Read from <a href="${esc(data.fetched_from)}" rel="noopener" target="_blank">${esc(new URL(data.fetched_from).hostname)}</a>.`;

  const batteryIds = new Set(data.battery.ids);
  renderTimeline(data.results.filter((r) => batteryIds.has(r.id)), data.battery);

  $("#list").innerHTML = data.results.map((r) => {
    const inBattery = batteryIds.has(r.id);
    const types = r.test_types.map((t) => `<span class="type"><i style="background:var(--t-${t})"></i>${esc(TYPE_NAMES[t] || t)}</span>`).join("");
    const extras = [r.remote ? "Remote testing" : null, r.adaptive ? "Adaptive" : null, r.job_levels.length ? r.job_levels.slice(0, 3).join(", ") : null].filter(Boolean).map(esc).join(", ");
    const matched = r.matched_queries.length ? `<div class="matched">Matched: ${r.matched_queries.map(esc).join(", ")}</div>` : "";
    return `<li class="item${inBattery ? " in-battery" : ""}">
      <span class="rank">${r.rank}</span>
      <div>
        <h3><a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.name)}</a>${inBattery ? '<span class="battery-tag">In battery</span>' : ""}</h3>
        ${r.reason ? `<p class="reason">${esc(r.reason)}</p>` : `<p class="reason">${esc(r.description.slice(0, 160))}${r.description.length > 160 ? "…" : ""}</p>`}
        <div class="meta">${types}${extras ? `<span>${extras}</span>` : ""}</div>
        ${matched}
      </div>
      <div class="duration"><strong>${r.minutes ? `${r.minutes} min` : "—"}</strong><small>${esc(r.minutes ? (r.duration_label.includes("untimed") ? "untimed" : "") : r.duration_label)}</small></div>
    </li>`;
  }).join("") || `<li class="item"><span></span><p>No assessments match these constraints. Try a longer time limit or turn off remote testing.</p></li>`;

  const p = data.pipeline;
  const s = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
  $("#steps").innerHTML = [
    `Your request was turned into requirements by ${p.parsed_by === "llm" ? "a language model" : "pattern rules (no language model configured)"} in ${s(data.timings.parse_ms)}.`,
    `${data.search_queries.length} searches ran, one for the whole request and one per skill: ${data.search_queries.slice(1).map((q) => `<code>${esc(q)}</code>`).join(" ") || "none"}. Each combines keyword (BM25) and meaning-based (embedding) matching.`,
    `${p.filtered_out} catalog entries were excluded by your constraints or because they are reports rather than tests. ${p.candidates} candidates remained.`,
    p.reranked_by === "llm" ? `A language model picked and ordered the final ten so every skill is covered, and wrote the reasons (${s(data.timings.rerank_ms)}).` : `The final ten take the best match for each skill in turn, so no single skill crowds out the others.`,
  ].map((t) => `<li>${t}</li>`).join("");
  $("#results").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
}

function renderTimeline(battery, info) {
  const budget = info.budget_minutes;
  const total = info.total_minutes;
  $("#battery-total").textContent = battery.length ? `${total} of ${budget} min` : "";
  const tl = $("#timeline");
  if (!battery.length) {
    tl.innerHTML = "";
    tl.hidden = true;
    $("#timeline-scale").innerHTML = `<p class="empty-battery">None of these tests has a fixed duration that fits ${budget} minutes.</p>`;
    return;
  }
  tl.hidden = false;
  tl.innerHTML = battery.map((r) => {
    const t = r.test_types[0];
    return `<div class="block" style="flex:0 0 ${(r.minutes / budget) * 100}%;background:var(--t-${t})" title="${esc(r.name)}, ${r.minutes} min"><b>${esc(r.name.replace(/\s*\(New\)/, ""))}</b><span>${r.minutes} min</span></div>`;
  }).join("");
  tl.setAttribute("aria-label", `Battery of ${battery.length} tests, ${total} of ${budget} minutes: ${battery.map((r) => `${r.name} ${r.minutes} minutes`).join(", ")}`);
  const marks = [0, Math.round(budget / 2), budget];
  $("#timeline-scale").innerHTML = marks.map((m) => `<span>${m} min</span>`).join("");
}

$("#form").addEventListener("submit", (e) => { e.preventDefault(); recommend($("#query").value.trim()); });
$("#query").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("#form").requestSubmit(); });
$("#examples").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  $("#query").value = b.textContent;
  recommend(b.textContent);
});
$("#rerun").addEventListener("click", () => {
  if (!lastQuery) return;
  const minutes = parseInt($("#max-minutes").value, 10);
  recommend(lastQuery, {
    max_minutes: Number.isFinite(minutes) && minutes > 0 ? minutes : null,
    job_level: $("#job-level").value || null,
    remote_testing: $("#remote").checked,
  });
});

// ---------- Accuracy ----------
let accuracyLoaded = false;
async function loadAccuracy() {
  if (accuracyLoaded) return;
  const el = $("#accuracy-body");
  try {
    const d = await (await fetch("/api/eval")).json();
    if (!d.available) { el.innerHTML = `<p class="muted">Results appear here after the first automated evaluation run.</p>`; return; }
    accuracyLoaded = true;
    const pct = (x) => `${Math.round(x * 100)}%`;
    const rows = d.stages.map((st) => `<tr><td>${esc(st.label)}</td><td><span class="meter"><i><b style="width:${st["recall@10"] * 100}%"></b></i>${pct(st["recall@10"])}</span></td><td>${st["map@10"].toFixed(3)}</td></tr>`).join("");
    const final = d.stages[d.stages.length - 1];
    const queries = [...final.queries].sort((a, b) => a["recall@10"] - b["recall@10"]);
    el.innerHTML = `<p class="run-meta">Run on ${new Date(d.run_at).toLocaleDateString(undefined, { dateStyle: "medium" })}, commit ${esc(d.commit)}. Embeddings: ${esc(d.embedder)}. Language model: ${esc(d.llm || "none")}.</p>
      <table class="stage-table"><thead><tr><th>Pipeline</th><th>Mean Recall@10</th><th>MAP@10</th></tr></thead><tbody>${rows}</tbody></table>
      <h2>Every test request, final pipeline</h2>
      <p class="muted">Sorted from weakest to strongest, with the expected assessments it missed.</p>
      <ul class="q-list">${queries.map((q) => `<li><span>${esc(q.query)}</span><strong>${pct(q["recall@10"])}</strong>${q.missed.length ? `<span class="missed">Missed: ${q.missed.map(esc).join(", ")}</span>` : ""}</li>`).join("")}</ul>`;
  } catch (_) {
    el.innerHTML = `<p class="muted">Couldn't load the results. Reload to try again.</p>`;
  }
}

show(location.hash === "#accuracy" ? "accuracy" : "find");

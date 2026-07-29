/* Study app — vanilla JS, no dependencies, no CDN, works offline.
   The force-directed graph is ~60 lines at the bottom rather than a d3 import,
   so the whole app keeps working with no network. */

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const api = async (path, opts) => {
  const r = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...opts,
  });
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
};

/* Minimal inline markdown → HTML for card text. Escapes first, so note content
   can never inject markup into the page. */
const md = (s) =>
  (s || "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<![*\w])\*([^*]+)\*(?!\w)/g, "<em>$1</em>");

let state = {
  queue: [], idx: 0, revealed: false,
  dash: null, graph: null, note: null, challenge: null,
};

/* ── navigation ──────────────────────────────────────────────────── */

$$("#nav button").forEach((b) =>
  b.addEventListener("click", () => show(b.dataset.view)),
);

function show(view) {
  $$("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  if (view === "review" && !state.queue.length) loadQueue();
  if (view === "read") loadLibrary();
  if (view === "code") loadChallenges();
  if (view === "exercises") loadExercises();
  if (view === "graph") loadGraph();
}

/* ── dashboard ───────────────────────────────────────────────────── */

async function loadDashboard() {
  const d = await api("/api/dashboard");
  state.dash = d;

  $("#subtitle").textContent =
    `${d.chapters_total} chapters · ${d.cards_total} cards · ${d.exercises_total} exercises · offline`;

  const badge = $("#due-badge");
  badge.textContent = d.cards_due + d.cards_new;
  badge.dataset.zero = d.cards_due + d.cards_new === 0 ? "1" : "0";

  const pct = d.chapters_total ? Math.round((d.chapters_done / d.chapters_total) * 100) : 0;
  $("#stats").innerHTML = [
    stat(`${d.chapters_done}/${d.chapters_total}`, "chapters done", `${pct}% of the roadmap`),
    stat(d.cards_due, "cards due today", `${d.cards_new} never seen`),
    stat(d.streak, d.streak === 1 ? "day streak" : "day streak", `${d.reviews_today} reviewed today`),
    stat(`${d.exercises_done}/${d.exercises_total}`, "exercises done", ""),
    stat(`${d.remaining_hours}h`, "estimated remaining", "from est_hours in the notes"),
    stat(d.totals.reviews, "total reviews", `${d.totals.cards_seen} cards seen`),
  ].join("");

  const max = Math.max(1, ...d.history.map((h) => h.count));
  $("#heatmap").innerHTML = d.history
    .map((h) => {
      const lvl = h.count === 0 ? 0 : Math.min(4, Math.ceil((h.count / max) * 4));
      return `<i data-l="${lvl}" title="${h.date}: ${h.count} review${h.count === 1 ? "" : "s"}"></i>`;
    })
    .join("");

  $("#phases").innerHTML = d.phases.map(phaseHTML).join("");
  $$(".phase-head").forEach((h) =>
    h.addEventListener("click", () => h.parentElement.classList.toggle("open")),
  );
  $$(".task input").forEach((cb) => cb.addEventListener("change", onToggle));
}

const stat = (n, l, s) =>
  `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`;

const phaseHTML = (p) => `
  <div class="phase">
    <div class="phase-head">
      <span class="chev">▶</span>
      <span class="phase-name">${p.name}</span>
      <span class="bar"><i style="width:${p.pct}%"></i></span>
      <span class="phase-count">${p.done}/${p.total}</span>
    </div>
    <div class="tasks">
      ${p.tasks
        .map(
          (t) => `<div class="task ${t.done ? "done" : ""}">
            <input type="checkbox" id="t-${t.code}" data-code="${t.code}" ${t.done ? "checked" : ""}>
            <code>${t.code}</code>
            <label for="t-${t.code}">${md(t.title)}</label>
          </div>`,
        )
        .join("")}
    </div>
  </div>`;

async function onToggle(e) {
  const cb = e.target;
  cb.disabled = true;
  try {
    await api("/api/roadmap/toggle", {
      method: "POST",
      body: JSON.stringify({ code: cb.dataset.code, done: cb.checked }),
    });
    const open = $$(".phase.open").map((p) => p.querySelector(".phase-name").textContent);
    await loadDashboard();
    $$(".phase").forEach((p) => {
      if (open.includes(p.querySelector(".phase-name").textContent)) p.classList.add("open");
    });
  } catch (err) {
    cb.checked = !cb.checked;
    alert(`Could not update ROADMAP.md: ${err.message}`);
  } finally {
    cb.disabled = false;
  }
}

/* ── review ──────────────────────────────────────────────────────── */

async function loadQueue() {
  const q = await api("/api/review/queue?limit=40");
  state.queue = q.cards;
  state.idx = 0;
  renderCard();
}

function renderCard() {
  const card = state.queue[state.idx];
  if (!card) {
    $("#review-card").classList.add("hidden");
    $("#review-empty").classList.remove("hidden");
    return;
  }
  $("#review-empty").classList.add("hidden");
  $("#review-card").classList.remove("hidden");
  state.revealed = false;

  $("#card-ref").textContent = `${card.ref} — ${card.chapter_title}`;
  $("#card-new").classList.toggle("hidden", !card.is_new);
  const lap = $("#card-lapses");
  lap.classList.toggle("hidden", card.lapses === 0);
  lap.textContent = `${card.lapses} lapse${card.lapses === 1 ? "" : "s"}`;

  $("#card-question").innerHTML = md(card.question);
  $("#card-answer").innerHTML = md(card.answer);
  $("#card-answer-wrap").classList.add("hidden");
  $("#grade-buttons").classList.add("hidden");
  $("#show-answer").classList.remove("hidden");
  $("#review-remaining").textContent = `${state.queue.length - state.idx} left`;

  for (const g of ["again", "hard", "good", "easy"]) {
    $(`#p-${g}`).textContent = card.preview[g] || "";
  }
}

function reveal() {
  if (state.revealed || !state.queue[state.idx]) return;
  state.revealed = true;
  $("#card-answer-wrap").classList.remove("hidden");
  $("#show-answer").classList.add("hidden");
  $("#grade-buttons").classList.remove("hidden");
}

async function grade(g) {
  if (!state.revealed) return;
  const card = state.queue[state.idx];
  if (!card) return;
  await api(`/api/review/${card.id}`, { method: "POST", body: JSON.stringify({ grade: g }) });
  state.idx += 1;
  if (state.idx >= state.queue.length) {
    await loadQueue();
    loadDashboard();
  } else {
    renderCard();
    loadDashboard();
  }
}

$("#show-answer").addEventListener("click", reveal);
$$("#grade-buttons button").forEach((b) =>
  b.addEventListener("click", () => grade(Number(b.dataset.grade))),
);

document.addEventListener("keydown", (e) => {
  if (!$("#view-review").classList.contains("active")) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space" || e.key === "Enter") {
    e.preventDefault();
    state.revealed ? grade(4) : reveal();
  }
  const map = { 1: 0, 2: 3, 3: 4, 4: 5 };
  if (e.key in map) grade(map[e.key]);
});

/* ── exercises ───────────────────────────────────────────────────── */

async function loadExercises() {
  const book = $("#ex-book").value;
  const kind = $("#ex-kind").value;
  const d = await api(`/api/exercises?book=${book}&kind=${kind}`);

  if ($("#ex-book").options.length === 1) {
    const books = [...new Map(d.exercises.map((e) => [e.book, e.book_label])).entries()];
    books.forEach(([slug, label]) => $("#ex-book").add(new Option(label, slug)));
  }

  const groups = {};
  for (const e of d.exercises) {
    const k = `${e.book_label} ch.${e.chapter} — ${e.chapter_title}`;
    (groups[k] ||= []).push(e);
  }

  $("#exercise-list").innerHTML =
    Object.entries(groups)
      .map(
        ([title, items]) => `<div class="ex-group"><h3>${title}</h3>${items
          .map(
            (e) => `<div class="ex ${e.done ? "done" : ""}">
              <input type="checkbox" data-id="${e.id}" ${e.done ? "checked" : ""}>
              <span class="kind kind-${e.kind}">${e.kind}</span>
              <span>${md(e.text)}</span>
            </div>`,
          )
          .join("")}</div>`,
      )
      .join("") || `<p class="hint">No exercises match.</p>`;

  $$("#exercise-list input").forEach((cb) =>
    cb.addEventListener("change", async () => {
      await api(`/api/exercises/${cb.dataset.id}`, {
        method: "POST",
        body: JSON.stringify({ done: cb.checked }),
      });
      cb.closest(".ex").classList.toggle("done", cb.checked);
      loadDashboard();
    }),
  );
}

$("#ex-book").addEventListener("change", loadExercises);
$("#ex-kind").addEventListener("change", loadExercises);

$("#run-tests").addEventListener("click", async () => {
  const btn = $("#run-tests");
  const out = $("#test-output");
  btn.disabled = true;
  btn.textContent = "Running…";
  out.classList.remove("hidden");
  out.textContent = "pytest -q …";
  try {
    const r = await api("/api/tests/run", { method: "POST" });
    out.textContent = `${r.passed ? "PASSED" : "FAILED"}\n\n${r.output}`;
  } catch (err) {
    out.textContent = `Could not run tests: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run test suite";
  }
});

/* ── knowledge graph — a small force simulation, no d3 ───────────── */

const BOOK_COLOR = {
  "01-hands-on-ml-geron": "#3b82f6",
  "02-hands-on-llms-alammar": "#a855f7",
  "03-build-llm-from-scratch-raschka": "#f43f5e",
  "04-ai-engineering-huyen": "#14b8a6",
  "05-ai-agents-in-action-lanham": "#f59e0b",
};

async function loadGraph() {
  if (state.graph) return;
  const g = await api("/api/graph");
  state.graph = g;
  $("#graph-stats").textContent = `${g.nodes.length} chapters · ${g.edges.length} links`;
  simulate(g);
}

function simulate(g) {
  const svg = $("#graph-svg");
  const W = svg.clientWidth || 900;
  const H = 560;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  // Deterministic start: same layout every time you open the tab.
  const nodes = g.nodes.map((n, i) => ({
    ...n,
    x: W / 2 + Math.cos((i / g.nodes.length) * 2 * Math.PI) * 200,
    y: H / 2 + Math.sin((i / g.nodes.length) * 2 * Math.PI) * 200,
    vx: 0,
    vy: 0,
  }));
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const edges = g.edges.filter((e) => byId[e.source] && byId[e.target]);

  for (let step = 0; step < 320; step++) {
    // repulsion
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        let dx = b.x - a.x, dy = b.y - a.y;
        let d2 = dx * dx + dy * dy || 0.01;
        const f = 2400 / d2;
        const d = Math.sqrt(d2);
        const fx = (dx / d) * f, fy = (dy / d) * f;
        a.vx -= fx; a.vy -= fy; b.vx += fx; b.vy += fy;
      }
    }
    // springs
    for (const e of edges) {
      const a = byId[e.source], b = byId[e.target];
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = (d - 90) * 0.012;
      const fx = (dx / d) * f, fy = (dy / d) * f;
      a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
    }
    // centring + damping + integrate
    for (const n of nodes) {
      n.vx += (W / 2 - n.x) * 0.0016;
      n.vy += (H / 2 - n.y) * 0.0016;
      n.vx *= 0.86; n.vy *= 0.86;
      n.x = Math.max(24, Math.min(W - 24, n.x + n.vx));
      n.y = Math.max(24, Math.min(H - 24, n.y + n.vy));
    }
  }

  const showLabels = $("#graph-labels").checked;
  const r = (n) => 4 + Math.min(9, n.degree * 0.7);

  svg.innerHTML =
    edges
      .map((e) => {
        const a = byId[e.source], b = byId[e.target];
        return `<line data-s="${e.source}" data-t="${e.target}" x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${b.x.toFixed(1)}" y2="${b.y.toFixed(1)}"/>`;
      })
      .join("") +
    nodes
      .map(
        (n) =>
          `<circle data-id="${n.id}" cx="${n.x.toFixed(1)}" cy="${n.y.toFixed(1)}" r="${r(n)}"
             fill="${BOOK_COLOR[n.book] || "#888"}" opacity="${n.done ? 1 : 0.45}"><title>${n.label} — ${n.title}</title></circle>`,
      )
      .join("") +
    (showLabels
      ? nodes
          .map(
            (n) =>
              `<text x="${n.x.toFixed(1)}" y="${(n.y - r(n) - 4).toFixed(1)}" text-anchor="middle">${n.code || n.label}</text>`,
          )
          .join("")
      : "");

  svg.querySelectorAll("circle").forEach((c) =>
    c.addEventListener("click", () => selectNode(c.dataset.id, byId, svg)),
  );
}

function selectNode(id, byId, svg) {
  const n = byId[id];
  const neighbours = new Set();
  svg.querySelectorAll("line").forEach((l) => {
    const hit = l.dataset.s === id || l.dataset.t === id;
    l.classList.toggle("hl", hit);
    if (hit) neighbours.add(l.dataset.s === id ? l.dataset.t : l.dataset.s);
  });

  const list = [...neighbours]
    .map((k) => byId[k])
    .filter(Boolean)
    .sort((a, b) => a.label.localeCompare(b.label))
    .map((m) => `<span class="tag">${m.code || m.label} ${m.title}</span>`)
    .join(" ");

  const detail = $("#graph-detail");
  detail.classList.remove("hidden");
  detail.innerHTML = `
    <h3>${n.code ? n.code + " · " : ""}${n.title}</h3>
    <span class="hint">${n.label} · difficulty ${n.difficulty} · ${n.cards} cards · ${n.exercises} exercises · ${n.done ? "done" : "not started"}</span>
    <div>${list || '<span class="hint">No cross-links.</span>'}</div>`;
}

$("#graph-labels").addEventListener("change", () => {
  if (state.graph) simulate(state.graph);
});

/* ── boot ────────────────────────────────────────────────────────── */

loadDashboard()
  .then(() => typeof route === "function" && route())
  .catch((e) => {
    $("#subtitle").textContent = `could not load: ${e.message}`;
  });

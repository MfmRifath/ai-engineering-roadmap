/* Read + Code views — the two surfaces where you actually study, as opposed to
   only tracking that you studied. Loaded after app.js and reuses its helpers. */

/* ═══════════════════════════ READ ═══════════════════════════════════ */

async function loadLibrary() {
  if ($("#library").dataset.loaded) return;
  const d = await api("/api/library");
  $("#library").innerHTML = d.books
    .map(
      (b) => `<div class="ex-group">
        <h3>${b.label} — ${b.title}</h3>
        ${b.chapters
          .map(
            (c) => `<a class="chapter-row ${c.done ? "done" : ""}" href="#/read/${b.slug}/${c.chapter}">
              <code>${c.code || "ch" + c.chapter}</code>
              <span class="ch-title">${c.title}</span>
              <span class="hint dots">${"●".repeat(c.difficulty)}${"○".repeat(5 - c.difficulty)}</span>
              <span class="hint">${c.est_hours}h</span>
              <span class="hint">${c.cards} cards</span>
              <span class="tick">${c.done ? "✓" : ""}</span>
            </a>`,
          )
          .join("")}
      </div>`,
    )
    .join("");
  $("#library").dataset.loaded = "1";
}

async function openNote(book, chapter) {
  const n = await api(`/api/read/${book}/${chapter}`);
  state.note = n;

  $("#read-index").classList.add("hidden");
  $("#read-note").classList.remove("hidden");

  $("#note-body").innerHTML =
    `<div class="note-head">
       <span class="tag">${n.book_label} ch.${n.chapter}</span>
       <span class="tag">difficulty ${n.difficulty}/5</span>
       <span class="tag">~${n.est_hours}h</span>
       <span class="tag">${n.cards} cards</span>
     </div>
     <h1 class="note-title">${n.title}</h1>` + n.html;

  $("#note-toc").innerHTML =
    `<div class="toc-title">On this page</div>` +
    n.toc.map((t) => `<a class="toc-l${t.level}" href="#${t.slug}">${t.title}</a>`).join("");

  $("#read-done").checked = n.done;
  $("#read-done").dataset.code = n.code;

  const prev = $("#read-prev");
  const next = $("#read-next");
  prev.classList.toggle("hidden", !n.prev);
  next.classList.toggle("hidden", !n.next);
  if (n.prev) {
    prev.textContent = `← ${n.prev.title}`;
    prev.dataset.to = `${n.prev.book}/${n.prev.chapter}`;
  }
  if (n.next) {
    next.textContent = `${n.next.title} →`;
    next.dataset.to = `${n.next.book}/${n.next.chapter}`;
  }

  window.scrollTo({ top: 0 });
}

$("#read-back").addEventListener("click", () => {
  history.replaceState(null, "", " ");
  $("#read-note").classList.add("hidden");
  $("#read-index").classList.remove("hidden");
  $("#library").dataset.loaded = "";
  loadLibrary();
});

["#read-prev", "#read-next"].forEach((id) =>
  $(id).addEventListener("click", (e) => {
    const to = e.currentTarget.dataset.to;
    if (to) location.hash = `#/read/${to}`;
  }),
);

$("#read-done").addEventListener("change", async (e) => {
  const code = e.target.dataset.code;
  if (!code) return;
  try {
    await api("/api/roadmap/toggle", {
      method: "POST",
      body: JSON.stringify({ code, done: e.target.checked }),
    });
    $("#library").dataset.loaded = "";
    loadDashboard();
  } catch (err) {
    e.target.checked = !e.target.checked;
    alert(`Could not update ROADMAP.md: ${err.message}`);
  }
});

/* Hash routing, so a cross-link inside a note opens that chapter here
   instead of dropping you into a raw markdown file. */
function route() {
  const m = location.hash.match(/^#\/read\/([^/]+)\/(\d+)/);
  if (m) {
    show("read");
    openNote(m[1], Number(m[2])).catch((e) => alert(e.message));
  }
}
window.addEventListener("hashchange", route);

/* ═══════════════════════════ CODE ═══════════════════════════════════ */

const PHASE_NAME = {
  1: "Phase 1 · ML Foundations",
  5: "Phase 5 · Transformers From Scratch",
  6: "Phase 6 · Fine-Tuning",
  7: "Phase 7 · Production AI Engineering",
  8: "Phase 8 · Agents",
};

async function loadChallenges() {
  const d = await api("/api/challenges");
  const groups = {};
  for (const c of d.challenges) (groups[c.phase] ||= []).push(c);

  $("#challenge-list").innerHTML = Object.keys(groups)
    .sort((a, b) => a - b)
    .map(
      (phase) => `<div class="ex-group">
        <h3>${PHASE_NAME[phase] || "Phase " + phase}</h3>
        ${groups[phase]
          .map(
            (c) => `<a class="chapter-row ${c.done ? "done" : ""}" data-cid="${c.id}">
              <code>${c.book}</code>
              <span class="ch-title">${c.title}</span>
              <span class="hint dots">${"●".repeat(c.difficulty)}${"○".repeat(5 - c.difficulty)}</span>
              <span class="hint">${c.tests} checks</span>
              <span class="tick">${c.done ? "✓" : ""}</span>
            </a>`,
          )
          .join("")}
      </div>`,
    )
    .join("");

  $$("#challenge-list [data-cid]").forEach((a) =>
    a.addEventListener("click", () => openChallenge(a.dataset.cid)),
  );
}

async function openChallenge(id) {
  const c = await api(`/api/challenges/${id}`);
  state.challenge = c;

  $("#code-index").classList.add("hidden");
  $("#code-work").classList.remove("hidden");
  $("#code-title").textContent = c.title;
  $("#code-meta").textContent = `${c.book} · difficulty ${c.difficulty}/5`;

  $("#code-prompt").innerHTML = c.prompt
    .split("\n\n")
    .map((p) => `<p>${md(p).replace(/\n/g, "<br>")}</p>`)
    .join("");

  $("#code-tests").innerHTML =
    `<h3 class="mini-h">Checks you can see</h3>` +
    c.tests.map((t) => `<div class="check-row"><span class="dot"></span><span>${md(t.name)}</span></div>`).join("");

  $("#code-hidden-note").textContent = c.hidden_count
    ? `+ ${c.hidden_count} hidden check${c.hidden_count === 1 ? "" : "s"} — so returning the expected value instead of implementing the function will not pass.`
    : "";

  $("#code-hints-body").innerHTML = c.hints.map((h) => `<p>${md(h)}</p>`).join("");
  $("#code-hints").classList.toggle("hidden", !c.hints.length);
  $("#code-hints").open = false;
  $("#code-solution-wrap").open = false;
  $("#code-solution").textContent = "";

  const saved = localStorage.getItem(`aieng:code:${id}`);
  $("#editor").value = saved || c.starter;
  $("#run-results").innerHTML = "";
  window.scrollTo({ top: 0 });
}

$("#code-back").addEventListener("click", () => {
  $("#code-work").classList.add("hidden");
  $("#code-index").classList.remove("hidden");
  loadChallenges();
});

$("#code-reset").addEventListener("click", () => {
  if (!state.challenge) return;
  if (confirm("Discard your code and restore the starter?")) {
    $("#editor").value = state.challenge.starter;
    localStorage.removeItem(`aieng:code:${state.challenge.id}`);
    $("#run-results").innerHTML = "";
  }
});

/* Drafts survive a reload — losing work to a stray refresh is the fastest way
   to stop trusting an editor. */
$("#editor").addEventListener("input", () => {
  if (state.challenge) {
    localStorage.setItem(`aieng:code:${state.challenge.id}`, $("#editor").value);
  }
});

$("#editor").addEventListener("keydown", (e) => {
  if (e.key === "Tab") {
    e.preventDefault();
    const el = e.target;
    const s = el.selectionStart;
    const t = el.selectionEnd;
    el.value = el.value.slice(0, s) + "    " + el.value.slice(t);
    el.selectionStart = el.selectionEnd = s + 4;
  }
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    runCode();
  }
});

$("#code-run").addEventListener("click", runCode);

async function runCode() {
  const c = state.challenge;
  if (!c) return;
  const btn = $("#code-run");
  btn.disabled = true;
  btn.textContent = "Running…";
  $("#run-results").innerHTML = `<div class="hint">running…</div>`;

  try {
    const r = await api(`/api/challenges/${c.id}/run`, {
      method: "POST",
      body: JSON.stringify({ code: $("#editor").value }),
    });
    renderRun(r);
    if (r.ok) loadDashboard();
  } catch (err) {
    $("#run-results").innerHTML = `<div class="run-error">Could not run: ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Run <kbd>ctrl+enter</kbd>";
  }
}

function renderRun(r) {
  let out = "";
  if (r.error) {
    out += `<div class="run-error">${md(r.error).replace(/\n/g, "<br>")}</div>`;
  }
  if (r.stdout) {
    out += `<pre class="test-output">${r.stdout.replace(/</g, "&lt;")}</pre>`;
  }
  if (r.total) {
    out += `<div class="run-summary ${r.ok ? "pass" : "fail"}">
        <span>${r.ok ? "All checks passed" : `${r.passed} of ${r.total} checks passed`}</span>
        <span class="hint">${r.duration_ms} ms</span>
      </div>`;
    out += r.results
      .map(
        (t) => `<div class="check-row ${t.passed ? "ok" : "bad"}">
          <span class="dot"></span>
          <span>${md(t.name)}${t.hidden ? ' <span class="tag tag-new">hidden</span>' : ""}
            ${t.error ? `<div class="check-err">${md(t.error)}</div>` : ""}</span>
        </div>`,
      )
      .join("");
  }
  if (r.ok) {
    out += `<div class="hint" style="margin-top:10px">Marked done — it will show on the dashboard.</div>`;
  }
  $("#run-results").innerHTML = out;
}

$("#code-solution-wrap").addEventListener("toggle", async (e) => {
  if (!e.target.open || !state.challenge || $("#code-solution").textContent) return;
  const r = await api(`/api/challenges/${state.challenge.id}/solution`);
  $("#code-solution").textContent = r.solution;
});

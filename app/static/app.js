/* Smart Healthcare Disease Prediction - front end (no dependencies).
   The form for each module is generated from /api/diseases, so adding a disease needs no UI change. */
(() => {
  "use strict";

  const state = { modules: [], current: null };
  const $ = (id) => document.getElementById(id);

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === false || v === null || v === undefined) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;   // conditional children
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return node;
  }

  // replaceChildren() would print the text "null" for a null argument, so drop empty entries first
  const setChildren = (node, items) => node.replaceChildren(...items.filter((x) => x !== null && x !== undefined && x !== false));

  const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
  const fmtNum = (x) => (Math.abs(x) >= 100 ? x.toFixed(0) : Number(x.toPrecision(3)).toString());

  async function api(path, options) {
    const res = await fetch(path, options);
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON error page */ }
    return { ok: res.ok, status: res.status, body };
  }

  // ------------------------------------------------------------------ modules
  async function init() {
    const { ok, body } = await api("/api/diseases");
    if (!ok || !body.length) {
      $("result").replaceChildren(el("div", { class: "error-box", text: "No trained models were found. Run `python -m src.train` first." }));
      return;
    }
    state.modules = body;
    renderModuleList();
    selectModule(body[0].key);
    loadHistory();
  }

  function renderModuleList() {
    const ul = $("module-list");
    ul.replaceChildren(...state.modules.map((m) => {
      const b = el("button", { type: "button", "data-key": m.key },
        el("strong", { text: m.name }), el("span", { text: m.selected_model }));
      b.addEventListener("click", () => selectModule(m.key));
      return el("li", {}, b);
    }));
  }

  function selectModule(key) {
    state.current = state.modules.find((m) => m.key === key);
    document.querySelectorAll("#module-list button").forEach((b) =>
      b.setAttribute("aria-current", b.dataset.key === key ? "true" : "false"));
    renderIntro();
    renderForm();
    renderEmptyResult();
  }

  function renderIntro() {
    const m = state.current;
    const t = m.metrics.test, ci = m.metrics.test_ci95;
    const range = (k) => `95% CI ${ci[k][0].toFixed(2)} to ${ci[k][1].toFixed(2)}`;
    $("module-intro").replaceChildren(
      el("h2", { id: "module-title", text: m.name }),
      el("p", { class: "desc", text: m.description }),
      el("dl", { class: "perf" },
        el("div", {}, el("dt", { text: "Accuracy" }), el("dd", { text: pct(t.accuracy, 1) }, el("small", { text: range("accuracy") }))),
        el("div", {}, el("dt", { text: "Recall (sensitivity)" }), el("dd", { text: pct(t.recall, 1) }, el("small", { text: range("recall") }))),
        el("div", {}, el("dt", { text: "ROC-AUC" }), el("dd", { text: t.roc_auc.toFixed(3) }, el("small", { text: range("roc_auc") })))),
      el("div", { class: "meta" },
        el("p", { text: `Model: ${m.selected_model}, trained on ${m.n_train} records; the metrics above are measured on ${m.n_test} held-out records.` }),
        el("p", { text: `Data: ${m.source}.` }),
        el("p", { text: `Limitation: ${m.population_note}` })));
  }

  // ------------------------------------------------------------------ form
  function renderForm() {
    const fields = state.current.features.map((f) => {
      const id = `f-${f.name}`;
      const hintParts = [];
      if (f.kind === "numeric" || f.kind === "integer") hintParts.push(`${f.min} to ${f.max}${f.unit ? " " + f.unit : ""}`);
      else if (f.unit) hintParts.push(f.unit);
      if (f.allow_missing) hintParts.push("optional, leave blank if unknown");
      let control;
      if (f.choices) {
        control = el("select", { id, name: f.name, "aria-describedby": `${id}-err` },
          f.allow_missing ? el("option", { value: "", text: "Not known" }) : null,
          ...f.choices.map((c) => el("option", { value: c.value, text: c.label })));
        if (!f.allow_missing) control.selectedIndex = 0;
      } else {
        control = el("input", { id, name: f.name, type: "number", inputmode: "decimal",
          step: f.kind === "integer" ? "1" : "any", min: f.min, max: f.max,
          required: !f.allow_missing, "aria-describedby": `${id}-err` });
      }
      return el("div", { class: "field", "data-name": f.name },
        el("label", { for: id }, f.label, el("span", { class: "hint", text: hintParts.join(", ") }),
          f.help ? el("span", { class: "hint", text: f.help }) : null),
        control, el("span", { class: "err", id: `${id}-err` }));
    });
    $("fields").replaceChildren(...fields);
  }

  function fillForm(values) {
    for (const f of state.current.features) {
      const node = $(`f-${f.name}`);
      const v = values[f.name];
      node.value = v === null || v === undefined ? "" : String(v);
    }
    clearErrors();
  }

  function clearErrors() {
    document.querySelectorAll(".field").forEach((d) => d.classList.remove("invalid"));
    document.querySelectorAll(".field .err").forEach((s) => (s.textContent = ""));
  }

  function showFieldErrors(fields) {
    clearErrors();
    let first = null;
    for (const [name, msg] of Object.entries(fields)) {
      const wrap = document.querySelector(`.field[data-name="${CSS.escape(name)}"]`);
      if (!wrap) continue;
      wrap.classList.add("invalid");
      wrap.querySelector(".err").textContent = msg;
      first = first || wrap.querySelector("input, select");
    }
    if (first) first.focus();
  }

  function collect() {
    const payload = {};
    for (const f of state.current.features) {
      const raw = $(`f-${f.name}`).value.trim();
      if (raw === "") continue;                       // blank -> omitted -> server treats as missing/required
      payload[f.name] = Number(raw);
    }
    return payload;
  }

  // ------------------------------------------------------------------ result rendering
  function renderEmptyResult() {
    $("result").replaceChildren(
      el("h2", { text: "Risk estimate" }),
      el("p", { class: "empty", text: "Enter the measurements or load a typical case, then choose Predict risk." }));
  }

  function renderResult(r) {
    const high = r.prediction === 1;
    const thr = r.screening_threshold;
    const ticks = [
      el("div", { class: "tick below", style: `left:${thr * 100}%` }, el("span", { text: `Screening line ${pct(thr)}` })),
      el("div", { class: "tick above", style: `left:${r.threshold * 100}%` }, el("span", { text: `Decision line ${pct(r.threshold)}` })),
    ];
    const ruler = el("div", { class: "ruler", role: "img", "aria-label": `Risk estimate ${pct(r.probability)} on a scale from 0 to 100 percent` },
      el("div", { class: `fill${high ? " high" : ""}`, style: `width:${r.probability * 100}%` }),
      ...ticks, el("div", { class: "pin", style: `left:${r.probability * 100}%` }));

    const flagText = r.screening_flag
      ? `Above the screening line. This profile would be flagged for follow-up, even if it is below the ${pct(r.threshold)} decision line.`
      : "Below the screening line. This profile would not be flagged for follow-up.";

    const maxEffect = Math.max(...(r.top_factors || []).map((f) => Math.abs(f.effect_pp)), 1);
    const factors = (r.top_factors || []).map((f) => {
      const w = (Math.abs(f.effect_pp) / maxEffect) * 50;
      const up = f.effect_pp > 0;
      return el("div", { class: "factor" },
        el("div", { class: "row" }, el("span", { text: `${f.label}: ${f.display_value}` }),
          el("b", { text: `${up ? "+" : "−"}${Math.abs(f.effect_pp).toFixed(1)} points` })),
        el("div", { class: "bar", role: "img", "aria-label": `${f.direction}` }, el("i", { class: up ? "up" : "down", style: `width:${w}%` })));
    });

    setChildren($("result"), [
      el("h2", { text: `${r.disease_name} risk estimate` }),
      el("div", { class: `risk-number ${high ? "high" : "low"}`, text: pct(r.probability) }),
      el("p", { class: "risk-label", text: r.label }),
      ruler,
      el("p", { class: `flag${r.screening_flag ? " on" : ""}`, text: flagText }),
      factors.length ? el("div", { class: "factors" },
        el("h3", { text: "What moved this estimate" }),
        el("p", { class: "cap", text: "Change in the estimate if this value were replaced by the typical training value. A what-if view, not a cause." }),
        ...factors) : null,
      r.warnings.length ? el("ul", { class: "warn" }, ...r.warnings.map((w) => el("li", { text: w }))) : null,
      el("p", { class: "foot", text: `${r.model.name}, version ${r.model.version}. Computed in ${r.latency_ms.toFixed(1)} ms.` })]);
  }

  // ------------------------------------------------------------------ history
  async function loadHistory() {
    const { ok, body } = await api("/api/history?limit=10");
    if (!ok) return;
    const rows = body.items.map((h) => el("tr", {},
      el("td", { text: h.created_at.replace("T", " ").replace("+00:00", "") }),
      el("td", { text: (state.modules.find((m) => m.key === h.disease) || { name: h.disease }).name }),
      el("td", { class: "num", text: pct(h.probability, 1) }),
      el("td", { text: h.screening_flag ? "Flagged" : "Not flagged" }),
      el("td", { text: h.model_name }),
      el("td", { class: "num", text: `${h.latency_ms.toFixed(1)} ms` })));
    $("history-body").replaceChildren(...rows);
    $("history-empty").hidden = rows.length > 0;
  }

  // ------------------------------------------------------------------ events
  $("predict-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = $("btn-predict");
    btn.disabled = true;
    clearErrors();
    try {
      const { ok, status, body } = await api(`/api/predict/${state.current.key}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(collect()),
      });
      if (ok) { renderResult(body); loadHistory(); return; }
      if (status === 422 && body && body.fields) {
        showFieldErrors(body.fields);
        $("result").replaceChildren(el("h2", { text: "Risk estimate" }),
          el("div", { class: "error-box", text: "Some values need fixing. See the highlighted fields." }));
      } else {
        $("result").replaceChildren(el("h2", { text: "Risk estimate" }),
          el("div", { class: "error-box", text: (body && body.error) || `Request failed with status ${status}.` }));
      }
    } catch (err) {
      $("result").replaceChildren(el("h2", { text: "Risk estimate" }),
        el("div", { class: "error-box", text: "The server could not be reached. Check that it is running and try again." }));
    } finally { btn.disabled = false; }
  });

  $("btn-pos").addEventListener("click", () => fillForm(state.current.examples.positive.values));
  $("btn-neg").addEventListener("click", () => fillForm(state.current.examples.negative.values));
  $("btn-clear").addEventListener("click", () => { fillForm({}); renderEmptyResult(); });
  $("btn-history").addEventListener("click", loadHistory);

  init();
})();

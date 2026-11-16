// Telco Churn retention console. Vanilla ES module; every number on screen comes from the API.

/* ------------------------------------------------------------------ DOM + format helpers */
const SVG = "http://www.w3.org/2000/svg";

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  setAttrs(el, attrs);
  append(el, kids);
  return el;
}
function s(tag, attrs, ...kids) {
  const el = document.createElementNS(SVG, tag);
  setAttrs(el, attrs);
  append(el, kids);
  return el;
}
function setAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "text") el.textContent = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
}
function append(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false) continue;
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
}

const nf = (d) => new Intl.NumberFormat("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmt = {
  int: (v) => nf(0).format(v),
  num: (v, d = 3) => (v == null ? "—" : nf(d).format(v)),
  pct: (v, d = 1) => (v == null ? "—" : `${nf(d).format(v * 100)}%`),
  money: (v) => (v == null ? "—" : `${v < 0 ? "−" : ""}$${nf(0).format(Math.abs(v))}`),
  money2: (v) => `${v < 0 ? "−" : ""}$${nf(2).format(Math.abs(v))}`,
  moneyK: (v) => {
    const a = Math.abs(v), sign = v < 0 ? "−" : "";
    if (a >= 1e6) return `${sign}$${nf(2).format(a / 1e6)}M`;
    if (a >= 1e4) return `${sign}$${nf(1).format(a / 1e3)}K`;
    return `${sign}$${nf(0).format(a)}`;
  },
  signed: (v, d = 3) => `${v > 0 ? "+" : v < 0 ? "−" : "±"}${nf(d).format(Math.abs(v))}`,
  signedMoney: (v) => `${v > 0 ? "+" : v < 0 ? "−" : "±"}$${nf(0).format(Math.abs(v))}`,
  ms: (v) => (v == null ? "—" : `${nf(v < 10 ? 2 : 1).format(v)} ms`),
  duration: (sec) => {
    sec = Math.round(sec);
    const d = Math.floor(sec / 86400), hr = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60);
    return d ? `${d}d ${hr}h` : hr ? `${hr}h ${m}m` : m ? `${m}m ${sec % 60}s` : `${sec}s`;
  },
  date: (iso) => new Date(iso).toLocaleString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" }),
};
const shown = (f, v) => (f === "SeniorCitizen" ? (Number(v) === 1 ? "Yes" : "No") : v);
const reasonText = (rs) => rs.map((x) => `${LABELS[x.feature] || x.feature} = ${shown(x.feature, x.value)}`).join(" · ") || "—";
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const titleCase = (k) => k.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

/* ------------------------------------------------------------------ API */
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (Array.isArray(j.detail)) {
        msg = j.detail.slice(0, 3).map((d) => `${(d.loc || []).filter((x) => x !== "body").join(" › ")}: ${d.msg}`).join("; ");
      } else if (j.detail) msg = String(j.detail);
    } catch { /* keep status text */ }
    throw new Error(msg);
  }
  return res.json();
}
let evidencePromise;
const evidence = () => (evidencePromise ??= api("/evidence"));

/* ------------------------------------------------------------------ tooltip */
const tip = document.getElementById("tip");
function showTip(x, y, nodes) {
  tip.replaceChildren(...nodes);
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > innerHeight - 8) top = y - r.height - 14;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}
const hideTip = () => { tip.hidden = true; };
function tipRow(value, label, color) {
  return h("div", { class: "row" }, color ? h("span", { class: "key", style: { background: color } }) : null,
    h("span", { class: "tv", text: value }), label ? h("span", { class: "tl", text: label }) : null);
}
/** Attach a hover + keyboard-focus tooltip to an element. `content()` returns nodes. */
function hover(el, content) {
  el.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, content()));
  el.addEventListener("pointerleave", hideTip);
  el.addEventListener("focus", () => { const r = el.getBoundingClientRect(); showTip(r.right, r.top, content()); });
  el.addEventListener("blur", hideTip);
  return el;
}

/* ------------------------------------------------------------------ chart primitives */
let clipSeq = 0;
function niceTicks(min, max, count = 5) {
  if (!isFinite(min) || !isFinite(max)) return [0, 1];
  if (min === max) { min -= 1; max += 1; }
  const step0 = (max - min) / count, mag = 10 ** Math.floor(Math.log10(step0)), err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.floor(min / step) * step; v <= Math.ceil(max / step) * step + step / 2; v += step) out.push(+v.toFixed(10));
  return out;
}

/** Render `draw(width)` into a responsive holder that redraws when its width changes. */
function chart(draw, cls = "chart") {
  const holder = h("div", { class: cls });
  let w = 0;
  new ResizeObserver((entries) => {
    const nw = Math.floor(entries[0].contentRect.width);
    if (nw > 0 && Math.abs(nw - w) > 3) { w = nw; holder.replaceChildren(draw(w)); }
  }).observe(holder);
  return holder;
}

function legend(items) {
  return h("div", { class: "legend" }, items.map(([name, color, kind = "line"]) =>
    h("span", {}, h("i", { class: kind === "sq" ? "sq" : kind === "dot" ? "dt" : "", style: { background: color } }), name)));
}

/**
 * Line / dot chart with one y-axis, hairline grid and a crosshair tooltip listing every series.
 * series: {name, color, points:[[x,y]], kind:"line"|"dots", band:[[x,lo,hi]]}
 */
function lineChart(w, o) {
  const H = o.height ?? 240, m = { t: 16, r: 16, b: o.xLabel ? 40 : 28, l: o.left ?? 58 };
  const pw = Math.max(40, w - m.l - m.r), ph = H - m.t - m.b;
  const xs = o.series.flatMap((sr) => sr.points.map((p) => p[0]));
  const ys = o.series.flatMap((sr) => [...sr.points.map((p) => p[1]), ...(sr.band || []).flatMap((b) => [b[1], b[2]])]);
  if (o.yZero) ys.push(0);
  for (const r of o.refs || []) if (r.y != null) ys.push(r.y);
  const [x0, x1] = o.xDomain || [Math.min(...xs), Math.max(...xs)];
  const yt = o.yTicks || (o.yDomain ? niceTicks(o.yDomain[0], o.yDomain[1], 4) : niceTicks(Math.min(...ys), Math.max(...ys), 4));
  const [y0, y1] = o.yDomain || [yt[0], yt[yt.length - 1]];
  const sx = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * pw;
  const sy = (v) => m.t + ph - ((v - y0) / (y1 - y0 || 1)) * ph;
  const fx = o.xFmt || ((v) => fmt.num(v, 0)), fy = o.yFmt || ((v) => fmt.num(v, 2));

  const svg = s("svg", { viewBox: `0 0 ${w} ${H}`, width: w, height: H, role: "img", "aria-label": o.label || "" });
  const clipId = `clip${++clipSeq}`;
  svg.append(s("defs", {}, s("clipPath", { id: clipId }, s("rect", { x: m.l - 6, y: m.t - 6, width: pw + 12, height: ph + 12 }))));
  const plot = s("g", { "clip-path": `url(#${clipId})` });
  const grid = s("g", { class: "grid" });
  for (const t of yt.filter((t) => t >= y0 - 1e-9 && t <= y1 + 1e-9)) {
    grid.append(s("line", { x1: m.l, x2: m.l + pw, y1: sy(t), y2: sy(t) }));
    svg.append(s("text", { x: m.l - 8, y: sy(t) + 3.5, "text-anchor": "end", text: fy(t) }));
  }
  svg.prepend(grid);
  if (y0 < 0 && y1 > 0) svg.append(s("line", { x1: m.l, x2: m.l + pw, y1: sy(0), y2: sy(0), stroke: css("--muted"), "stroke-width": 1 }));
  const xt = (o.xTicks || niceTicks(x0, x1, w < 520 ? 4 : 6)).filter((t) => t >= x0 - 1e-9 && t <= x1 + 1e-9);
  svg.append(s("line", { class: "axis", x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph, stroke: css("--line-strong") }));
  for (const t of xt) svg.append(s("text", { x: sx(t), y: m.t + ph + 16, "text-anchor": "middle", text: fx(t) }));
  if (o.xLabel) svg.append(s("text", { x: m.l + pw / 2, y: H - 4, "text-anchor": "middle", class: "ink", text: o.xLabel }));

  for (const r of o.refs || []) {
    if (r.x != null && r.x >= x0 && r.x <= x1) {
      svg.append(s("line", { x1: sx(r.x), x2: sx(r.x), y1: m.t, y2: m.t + ph, stroke: r.color || css("--ink-2"), "stroke-width": 1 }));
      if (r.label) svg.append(s("text", { x: sx(r.x) + (r.anchor === "end" ? -6 : 6), y: m.t + 10, "text-anchor": r.anchor || "start", class: "ink", text: r.label }));
    }
    if (r.y != null) {
      svg.append(s("line", { x1: m.l, x2: m.l + pw, y1: sy(r.y), y2: sy(r.y), stroke: r.color || css("--muted"), "stroke-width": 1 }));
      if (r.label) svg.append(s("text", { x: m.l + pw - 4, y: sy(r.y) - 5, "text-anchor": "end", class: "ink", text: r.label }));
    }
  }
  svg.append(plot);
  for (const sr of o.series) {
    if (sr.band?.length) {
      const top = sr.band.map((b) => `${sx(b[0])},${sy(b[2])}`), bot = sr.band.map((b) => `${sx(b[0])},${sy(b[1])}`).reverse();
      plot.append(s("path", { d: `M${top.join("L")}L${bot.join("L")}Z`, fill: sr.color, opacity: 0.12 }));
    }
    if (sr.kind === "dots") {
      for (const p of sr.points) plot.append(s("circle", { cx: sx(p[0]), cy: sy(p[1]), r: sr.r ?? 2.5, fill: sr.color, opacity: sr.opacity ?? 0.5 }));
    } else if (sr.points.length) {
      plot.append(s("path", { d: "M" + sr.points.map((p) => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("L"),
        fill: "none", stroke: sr.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    }
  }
  for (const mk of o.markers || []) {
    svg.append(s("circle", { cx: sx(mk.x), cy: sy(mk.y), r: 4.5, fill: mk.color, stroke: css("--page"), "stroke-width": 2 }));
    if (mk.label) svg.append(s("text", { x: sx(mk.x) + (mk.anchor === "end" ? -9 : 9), y: sy(mk.y) - 8, "text-anchor": mk.anchor || "start", class: "ink", text: mk.label }));
  }

  // Crosshair: snaps to the nearest x of the first series; the tooltip lists every series at that x.
  const xhair = s("line", { class: "xhair", y1: m.t, y2: m.t + ph, visibility: "hidden" });
  const dots = o.series.map((sr) => s("circle", { r: 4, fill: sr.color, stroke: css("--page"), "stroke-width": 2, visibility: "hidden", "pointer-events": "none" }));
  svg.append(xhair, ...dots);
  const hit = s("rect", { class: "hit", x: m.l, y: m.t, width: pw, height: ph });
  const nearest = (pts, xv) => pts.reduce((best, p) => (Math.abs(p[0] - xv) < Math.abs(best[0] - xv) ? p : best), pts[0]);
  hit.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const xv = x0 + ((e.clientX - r.left - m.l) / pw) * (x1 - x0);
    const anchor = nearest(o.series[0].points, xv);
    xhair.setAttribute("x1", sx(anchor[0])); xhair.setAttribute("x2", sx(anchor[0])); xhair.setAttribute("visibility", "visible");
    const rows = [h("div", { class: "tl", text: (o.tipX || fx)(anchor[0]) })];
    o.series.forEach((sr, i) => {
      if (!sr.points.length) return;
      const p = nearest(sr.points, anchor[0]);
      if (sr.kind !== "dots") { dots[i].setAttribute("cx", sx(p[0])); dots[i].setAttribute("cy", sy(p[1])); dots[i].setAttribute("visibility", "visible"); }
      rows.push(tipRow((o.tipY || fy)(p[1]), sr.name, sr.color));
    });
    showTip(e.clientX, e.clientY, rows);
  });
  hit.addEventListener("pointerleave", () => { hideTip(); xhair.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); });
  svg.append(hit);
  return svg;
}

/** Horizontal bar list in HTML: rows of [name, value, {color, title, tip}] with optional threshold ticks. */
function barList(rows, o = {}) {
  const max = o.max ?? Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  return h("div", { class: "bars", role: "list" }, rows.map((r) => {
    const track = h("div", { class: "track" },
      h("i", { style: { width: `${Math.min(100, (Math.abs(r.value) / max) * 100)}%`, background: r.color || "var(--s1)" } }),
      (o.ticks || []).map((t) => h("s", { style: { left: `${(t / max) * 100}%` } })));
    const row = h("div", { class: "bar-row", role: "listitem", tabindex: r.tip ? 0 : null, style: o.wideNames ? { gridTemplateColumns: "minmax(120px, 48%) minmax(0, 1fr) auto" } : null },
      h("span", { class: "name", title: r.name, text: r.name }), track, h("span", { class: "val", text: (o.fmt || fmt.num)(r.value) }));
    if (r.tip) hover(row, r.tip);
    return row;
  }));
}

function statusTag(status, text) {
  return h("span", { class: `tag st-${status}` }, h("i"), text ?? titleCase(status));
}
const ICONS = {
  check: '<svg viewBox="0 0 18 18" class="icon" aria-hidden="true"><circle cx="9" cy="9" r="8" fill="var(--good)"/><path d="M5.5 9.2l2.3 2.3 4.7-5" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  cross: '<svg viewBox="0 0 18 18" class="icon" aria-hidden="true"><circle cx="9" cy="9" r="8" fill="var(--critical)"/><path d="M6.2 6.2l5.6 5.6M11.8 6.2l-5.6 5.6" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/></svg>',
  hold: '<svg viewBox="0 0 18 18" class="icon" aria-hidden="true"><circle cx="9" cy="9" r="7.25" fill="none" stroke="var(--muted)" stroke-width="1.5"/><path d="M5.5 9h7" stroke="var(--muted)" stroke-width="1.8" stroke-linecap="round"/></svg>',
};
const icon = (name) => { const t = document.createElement("template"); t.innerHTML = ICONS[name]; return t.content.firstChild; };

/** "MonthlyCharges + 7 related" → "Monthly bill + 7 related", using the form's readable names. */
function groupLabel(g) {
  return g.split(" + ").map((part) => LABELS[part] || part).join(" + ");
}
function block(title, desc, content, opts = {}) {
  return h("div", { class: `block${opts.wide ? " wide" : ""}`, id: opts.id },
    h("div", { class: "aside" }, opts.eyebrow ? h("div", { class: "eyebrow", text: opts.eyebrow, style: { marginBottom: "8px" } }) : null,
      h("h2", { text: title }), desc ? (Array.isArray(desc) ? desc : [h("p", { text: desc })]) : null),
    h("div", { class: "stack", style: { minWidth: 0 } }, content));
}
function pageHead(eyebrow, title, lede) {
  return h("header", { class: "page-head" }, h("div", { class: "eyebrow accent", text: eyebrow }), h("h1", { text: title }), h("p", { class: "lede", text: lede }));
}
function figures(items) {
  return h("div", { class: "figures" }, items.map(([label, value, sub]) =>
    h("div", { class: "figure" }, h("div", { class: "label", text: label }), h("div", { class: "value", text: value }), sub ? h("div", { class: "sub", text: sub }) : null)));
}
function table(cols, rows, o = {}) {
  return h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, cols.map((c) => h("th", { class: [c.r ? "r" : "", c.cls || ""].join(" ").trim() || null, text: c.label })))),
    h("tbody", {}, rows.map((row) => row instanceof Node ? row : h("tr", { class: o.rowClass?.(row) },
      cols.map((c) => { const v = c.cell(row); return h("td", { class: [c.r ? "r" : "", c.cls || ""].join(" ").trim() || null }, v); }))))));
}
const ciText = (ci, f) => (ci ? `${f(ci[0])} – ${f(ci[1])}` : "");
function valueCI(v, ci, f) {
  return [h("div", { class: "num", text: f(v) }), ci ? h("div", { class: "ci num", text: ciText(ci, f) }) : null];
}
function loading(view, text = "Loading…") { view.replaceChildren(h("div", { class: "skeleton", text })); }
function failed(view, err) {
  view.replaceChildren(pageHead("Error", "Could not load this section", ""), h("p", { class: "error", text: err.message }));
}

/* ------------------------------------------------------------------ 01 · Customer scoring */
const YN = ["Yes", "No"];
const ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"];
const FIELDS = [
  ["Account", [
    ["tenure", "Tenure (months)", { type: "int", min: 0, max: 120 }],
    ["Contract", "Contract", ["Month-to-month", "One year", "Two year"]],
    ["MonthlyCharges", "Monthly bill ($)", { type: "float", min: 0, max: 1000, step: 0.05 }],
    ["PaymentMethod", "Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]],
    ["PaperlessBilling", "Paperless billing", YN],
  ]],
  ["Services", [
    ["PhoneService", "Phone service", YN],
    ["MultipleLines", "Multiple lines", ["Yes", "No", "No phone service"]],
    ["InternetService", "Internet", ["Fiber optic", "DSL", "No"]],
    ["OnlineSecurity", "Online security", ["Yes", "No", "No internet service"]],
    ["OnlineBackup", "Online backup", ["Yes", "No", "No internet service"]],
    ["DeviceProtection", "Device protection", ["Yes", "No", "No internet service"]],
    ["TechSupport", "Tech support", ["Yes", "No", "No internet service"]],
    ["StreamingTV", "Streaming TV", ["Yes", "No", "No internet service"]],
    ["StreamingMovies", "Streaming movies", ["Yes", "No", "No internet service"]],
  ]],
  ["Household", [
    ["SeniorCitizen", "Senior citizen", { type: "senior" }],
    ["gender", "Gender", ["Female", "Male"]],
    ["Partner", "Partner", YN],
    ["Dependents", "Dependents", YN],
  ]],
];
const LABELS = Object.fromEntries(FIELDS.flatMap(([, fs]) => fs.map(([k, l]) => [k, l])));
const PRESETS = {
  "Flight risk": { SeniorCitizen: 1, tenure: 1, MonthlyCharges: 95, gender: "Female", Partner: "No", Dependents: "No", PhoneService: "Yes",
    MultipleLines: "No", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No",
    StreamingTV: "No", StreamingMovies: "No", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check" },
  "Loyal": { SeniorCitizen: 0, tenure: 72, MonthlyCharges: 95, gender: "Female", Partner: "Yes", Dependents: "Yes", PhoneService: "Yes",
    MultipleLines: "No", InternetService: "Fiber optic", OnlineSecurity: "Yes", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "Yes",
    StreamingTV: "No", StreamingMovies: "No", Contract: "Two year", PaperlessBilling: "Yes", PaymentMethod: "Credit card (automatic)" },
  "Budget plan": { SeniorCitizen: 0, tenure: 3, MonthlyCharges: 20, gender: "Male", Partner: "No", Dependents: "No", PhoneService: "Yes",
    MultipleLines: "No", InternetService: "No", OnlineSecurity: "No internet service", OnlineBackup: "No internet service",
    DeviceProtection: "No internet service", TechSupport: "No internet service", StreamingTV: "No internet service",
    StreamingMovies: "No internet service", Contract: "Month-to-month", PaperlessBilling: "No", PaymentMethod: "Mailed check" },
};

function renderScore(view) {
  const state = { ...PRESETS["Flight risk"] };
  const inputs = {};
  const result = h("div", { class: "result", "aria-live": "polite" });
  let timer, seq = 0, econ = null;

  const presetBar = h("div", { class: "seg", role: "group", "aria-label": "Presets" });
  const presetButtons = Object.keys(PRESETS).map((name) => h("button", {
    type: "button", "aria-pressed": name === "Flight risk" ? "true" : "false", text: name,
    onclick: () => { Object.assign(state, PRESETS[name]); presetButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.textContent === name))); sync(); score(); },
  }));
  presetBar.append(...presetButtons);

  const form = h("form", { class: "form", onsubmit: (e) => e.preventDefault() }, FIELDS.map(([group, fields]) =>
    h("fieldset", { class: "fieldset" }, h("legend", { class: "eyebrow", text: group }), fields.map(([key, label, spec]) => {
      let input;
      if (Array.isArray(spec)) input = h("select", { id: `f-${key}` }, spec.map((o) => h("option", { value: o, text: o })));
      else if (spec.type === "senior") input = h("select", { id: `f-${key}` }, h("option", { value: "1", text: "Yes" }), h("option", { value: "0", text: "No" }));
      else input = h("input", { id: `f-${key}`, type: "number", min: spec.min, max: spec.max, step: spec.step || 1, inputmode: spec.type === "int" ? "numeric" : "decimal" });
      input.addEventListener("input", () => {
        const v = input.value;
        state[key] = spec.type === "int" || spec.type === "senior" ? parseInt(v || "0", 10) : spec.type === "float" ? parseFloat(v || "0") : v;
        presetButtons.forEach((b) => b.setAttribute("aria-pressed", "false"));
        sync();
        clearTimeout(timer); timer = setTimeout(score, 220);
      });
      inputs[key] = input;
      return h("div", { class: "field" }, h("label", { for: `f-${key}`, text: label }), input);
    }))));

  // Poka-yoke: impossible combinations cannot be entered. No internet → add-ons are "No internet service".
  function sync() {
    const noNet = state.InternetService === "No", noPhone = state.PhoneService === "No";
    for (const a of ADDONS) {
      if (noNet) state[a] = "No internet service";
      else if (state[a] === "No internet service") state[a] = "No";
      inputs[a].disabled = noNet;
      [...inputs[a].options].forEach((o) => { o.hidden = !noNet && o.value === "No internet service"; });
    }
    if (noPhone) state.MultipleLines = "No phone service";
    else if (state.MultipleLines === "No phone service") state.MultipleLines = "No";
    inputs.MultipleLines.disabled = noPhone;
    [...inputs.MultipleLines.options].forEach((o) => { o.hidden = !noPhone && o.value === "No phone service"; });
    for (const [k, el] of Object.entries(inputs)) {
      const v = String(state[k]);
      if (el.value !== v && document.activeElement !== el) el.value = v;
    }
  }

  async function score() {
    const mine = ++seq;
    result.classList.add("loading");
    try {
      const [res, ev] = await Promise.all([api("/explain", state), econ ?? evidence()]);
      econ = ev;
      if (mine === seq) drawResult(res, ev.economics);
    } catch (e) {
      if (mine === seq) result.replaceChildren(h("p", { class: "error", text: `Could not score: ${e.message}` }));
    } finally {
      if (mine === seq) result.classList.remove("loading");
    }
  }

  function drawResult(r, e) {
    const m = state.MonthlyCharges;
    const worth = e.save_rate * e.horizon * m;                 // value of saving this customer
    const breakeven = worth > 0 ? (e.offer_cost + r.margin) / worth : Infinity; // risk above which an offer pays off
    const send = r.send_offer;
    const why = send
      ? `Expected value of ${fmt.money2(r.expected_value)} clears the ${fmt.money(r.margin)} margin: ${fmt.pct(r.churn_probability)} churn risk × ${fmt.pct(e.save_rate, 0)} save rate × ${fmt.money(worth / e.save_rate)} of revenue over ${e.horizon} months, less the ${fmt.money(e.offer_cost)} offer.`
      : breakeven > 1
        ? `No risk level makes an offer pay off here: saving this customer returns at most ${fmt.money(worth)}, below the ${fmt.money(e.offer_cost)} offer cost.`
        : `At ${fmt.pct(r.churn_probability)} risk the expected return is ${fmt.money2(r.expected_value)}. For a ${fmt.money(m)}/month customer an offer only pays off above ${fmt.pct(breakeven)} risk.`;

    const meter = h("div", { class: "meter", role: "meter", "aria-valuemin": 0, "aria-valuemax": 1, "aria-valuenow": r.churn_probability, "aria-label": "Churn probability" },
      h("i", { style: { width: `${r.churn_probability * 100}%` } }),
      breakeven <= 1 ? h("b", { style: { left: `calc(${breakeven * 100}% - 1px)` }, title: `Break-even risk ${fmt.pct(breakeven)}` }) : null);

    const contribs = r.contributions;
    const top = contribs.slice(0, 9);
    const rest = contribs.slice(9);
    const items = top.map((c) => ({ label: `${LABELS[c.feature] || c.feature}`, value: shown(c.feature, c.value), group: c.group !== c.feature ? groupLabel(c.group) : null, c: c.contribution }));
    if (rest.length) items.push({ label: `${rest.length} other features`, value: null, group: null, c: rest.reduce((a, c) => a + c.contribution, 0) });

    result.replaceChildren(
      h("div", {},
        h("div", { class: "eyebrow", text: "Calibrated churn probability" }),
        h("div", { class: "hero", style: { margin: "10px 0 14px" } }, fmt.num(r.churn_probability * 100, 1), h("small", { text: "%" })),
        meter,
        h("div", { class: "note", style: { marginTop: "8px", display: "flex", justifyContent: "space-between", gap: "8px" } },
          h("span", { text: `Raw model score ${fmt.pct(r.raw_probability)}` }),
          breakeven <= 1 ? h("span", { text: `Break-even risk ${fmt.pct(breakeven)}` }) : h("span", { text: "Offer cannot pay off" }))),
      h("div", { class: "decision" }, icon(send ? "check" : "hold"),
        h("div", {}, h("div", { class: "verdict", text: send ? "Send retention offer" : "Hold — no offer" }), h("p", { text: why }))),
      h("dl", { class: "kv" },
        h("dt", { text: "Expected value of an offer" }), h("dd", { text: fmt.money2(r.expected_value) }),
        h("dt", { text: "Decision margin" }), h("dd", { text: fmt.money2(r.margin) }),
        h("dt", { text: `Value if saved (${fmt.pct(e.save_rate, 0)} × ${e.horizon} mo × bill)` }), h("dd", { text: fmt.money2(worth) }),
        h("dt", { text: "Offer cost" }), h("dd", { text: fmt.money2(e.offer_cost) })),
      h("div", {},
        h("div", { class: "chart-title" }, h("h3", { text: "Why this score" }), h("span", { class: "note", text: "SHAP, log-odds" })),
        legend([["Raises risk", "var(--pos)", "sq"], ["Lowers risk", "var(--neg)", "sq"]]),
        chart((w) => waterfall(w, r.base_log_odds, items)),
        h("p", { class: "note", style: { marginTop: "10px" } },
          "Bars add up from the average customer to this customer's raw model score; tick labels show the equivalent probability. Calibration then maps the raw score to the probability above. Features tagged with a group are correlated — read their credit jointly, not one by one.")),
    );
  }

  view.replaceChildren(
    pageHead("01 · Customer scoring", "Is this customer worth saving?",
      "Score one customer: a calibrated churn probability, the expected value of a retention offer against its cost, and the features that drive the risk."),
    h("div", { class: "toolbar", style: { paddingTop: "20px" } }, h("span", { class: "eyebrow", text: "Start from" }), presetBar),
    h("div", { class: "score-grid" }, form, result),
  );
  sync();
  score();
}

/** Horizontal SHAP waterfall: base → contributions → raw model score, on a log-odds axis labelled in probability. */
function waterfall(w, base, items) {
  const narrow = w < 560;
  const rowH = 34, labelW = narrow ? Math.min(150, w * 0.42) : 230, valW = 52, top = 6, axisH = 26;
  const rows = [{ kind: "base", label: "Average customer", c: 0 }, ...items, { kind: "final", label: "This customer (raw score)", c: 0 }];
  let cum = base;
  const geo = rows.map((r) => {
    if (r.kind === "base") return { ...r, from: base, to: base };
    if (r.kind === "final") return { ...r, from: cum, to: cum };
    const g = { ...r, from: cum, to: cum + r.c };
    cum += r.c;
    return g;
  });
  const vals = geo.flatMap((g) => [g.from, g.to]);
  const ticks = niceTicks(Math.min(...vals, -0.5), Math.max(...vals, 0.5), narrow ? 3 : 5);
  const lo = ticks[0], hi = ticks[ticks.length - 1];
  const x0 = labelW + 8, pw = w - x0 - valW - 6;
  const sx = (v) => x0 + ((v - lo) / (hi - lo)) * pw;
  const H = top + rows.length * rowH + axisH;
  const svg = s("svg", { viewBox: `0 0 ${w} ${H}`, width: w, height: H, role: "img", "aria-label": "SHAP waterfall of this customer's churn score" });
  for (const t of ticks) {
    svg.append(s("line", { x1: sx(t), x2: sx(t), y1: top, y2: top + rows.length * rowH, stroke: css("--line") }));
    svg.append(s("text", { x: sx(t), y: H - 8, "text-anchor": "middle", text: fmt.pct(sigmoid(t), sigmoid(t) < 0.1 || sigmoid(t) > 0.9 ? 1 : 0) }));
  }
  geo.forEach((g, i) => {
    const y = top + i * rowH, cy = y + rowH / 2;
    const name = s("text", { x: 0, y: cy - (g.group && !narrow ? 2 : -4), class: "ink", text: g.label });
    svg.append(name);
    if (g.value != null && g.kind == null) {
      const shown = typeof g.value === "number" ? String(g.value) : g.value;
      name.textContent = narrow ? `${g.label}: ${shown}` : `${g.label} = ${shown}`;
    }
    if (g.group && !narrow) svg.append(s("text", { x: 0, y: cy + 11, style: "font-size:10.5px", text: `group · ${g.group}` }));
    if (g.kind) {
      const v = g.from;
      svg.append(s("circle", { cx: sx(v), cy, r: 5, fill: css("--ink"), stroke: css("--page"), "stroke-width": 2 }));
      svg.append(s("text", { x: w, y: cy + 4, "text-anchor": "end", class: "ink", text: fmt.pct(sigmoid(v)) }));
    } else {
      const a = Math.min(g.from, g.to), b = Math.max(g.from, g.to), pos = g.c >= 0;
      const width = Math.max(1.5, sx(b) - sx(a)), r = Math.min(4, width / 2);
      // 4px rounded data-end, square at the start of the push.
      const x = sx(a), yb = cy - 7, hb = 14;
      const d = pos
        ? `M${x},${yb}h${width - r}a${r},${r} 0 0 1 ${r},${r}v${hb - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(width - r)}z`
        : `M${x + width},${yb}h${-(width - r)}a${r},${r} 0 0 0 ${-r},${r}v${hb - 2 * r}a${r},${r} 0 0 0 ${r},${r}h${width - r}z`;
      svg.append(s("path", { d, fill: pos ? css("--pos") : css("--neg") }));
      svg.append(s("text", { x: w, y: cy + 4, "text-anchor": "end", class: "ink", text: fmt.signed(g.c, 2) }));
      if (i < geo.length - 1) svg.append(s("line", { x1: sx(g.to), x2: sx(g.to), y1: cy + 7, y2: cy + rowH - 7, stroke: css("--line-strong") }));
    }
    const hit = s("rect", { x: 0, y, width: w, height: rowH, fill: "transparent", tabindex: 0 });
    hover(hit, () => [
      h("div", { class: "tl", text: g.kind === "base" ? "Average customer" : g.kind === "final" ? "This customer, before calibration" : g.label }),
      g.kind ? tipRow(fmt.pct(sigmoid(g.from)), `log-odds ${fmt.num(g.from, 2)}`)
        : tipRow(fmt.signed(g.c, 3), "log-odds push"),
      g.kind ? null : tipRow(`${fmt.pct(sigmoid(g.from))} → ${fmt.pct(sigmoid(g.to))}`, "probability"),
      g.group ? h("div", { class: "tl", text: `Correlated group: ${g.group}` }) : null,
    ]);
    svg.append(hit);
  });
  return svg;
}

/* ------------------------------------------------------------------ 02 · Batch predictions */
function parseCSV(text) {
  const rows = [];
  let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += c;
    } else if (c === '"') q = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows;
  return { head: head.map((x) => x.trim()), body };
}

function renderBatch(view) {
  const input = h("input", { type: "file", accept: ".csv,text/csv", hidden: true });
  const status = h("div", { class: "note", id: "batch-status", "aria-live": "polite" });
  const out = h("div", { class: "stack" });
  const drop = h("label", { class: "drop", tabindex: 0 },
    h("svg", { width: 0, height: 0 }),
    h("b", { text: "Drop a customer CSV, or click to choose" }),
    h("span", { class: "note", text: "Same columns as the IBM Telco export; extra columns (Churn, TotalCharges) are ignored. Up to 10,000 rows." }),
    input);
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files[0]) load(e.dataTransfer.files[0].text(), e.dataTransfer.files[0].name); });
  input.addEventListener("change", () => input.files[0] && load(input.files[0].text(), input.files[0].name));
  const sampleBtn = h("button", { type: "button", class: "btn", text: "Use sample — 500 customers",
    onclick: () => load(fetch("/static/sample.csv").then((r) => r.text()), "sample.csv") });

  view.replaceChildren(
    pageHead("02 · Batch predictions", "Score a whole customer file",
      "Upload a customer extract to rank it for retention outreach, export the decisions, and check whether the file looks like the customers the model was trained on."),
    block("Input", "The file is parsed in your browser and sent to the scoring API in chunks of 2,000. Nothing is stored server-side.",
      [drop, h("div", { class: "toolbar" }, sampleBtn, status)]),
    out,
  );

  async function load(textPromise, name) {
    out.replaceChildren();
    status.className = "note";
    status.textContent = `Reading ${name}…`;
    try {
      const [text, ev] = await Promise.all([textPromise, evidence()]);
      const { head, body } = parseCSV(text);
      const need = ev.features;
      const missing = need.filter((f) => !head.includes(f));
      if (missing.length) throw new Error(`Missing column${missing.length > 1 ? "s" : ""}: ${missing.join(", ")}`);
      const ix = Object.fromEntries(head.map((k, i) => [k, i]));
      const records = [], skipped = [];
      body.forEach((r, n) => {
        const rec = {};
        for (const f of need) rec[f] = (r[ix[f]] ?? "").trim();
        rec.SeniorCitizen = Number(rec.SeniorCitizen);
        rec.tenure = Number(rec.tenure);
        rec.MonthlyCharges = Number(rec.MonthlyCharges);
        if (ix.customerID != null) rec.customerID = r[ix.customerID];
        if (!Number.isFinite(rec.tenure) || !Number.isFinite(rec.MonthlyCharges) || ![0, 1].includes(rec.SeniorCitizen) || rec.tenure < 0 || rec.MonthlyCharges < 0) skipped.push(n + 2);
        else records.push({ rec, churn: ix.Churn != null ? r[ix.Churn] : null });
      });
      if (!records.length) throw new Error("No valid rows found.");
      if (records.length > 10000) throw new Error(`The file has ${fmt.int(records.length)} rows; the limit is 10,000.`);
      const preds = [];
      for (let i = 0; i < records.length; i += 2000) {
        status.textContent = `Scoring ${fmt.int(Math.min(i + 2000, records.length))} of ${fmt.int(records.length)}…`;
        preds.push(...await api("/predict", records.slice(i, i + 2000).map((x) => x.rec)));
      }
      status.textContent = `${name}: ${fmt.int(records.length)} customers scored${skipped.length ? `, ${skipped.length} rows skipped (lines ${skipped.slice(0, 5).join(", ")}${skipped.length > 5 ? "…" : ""})` : ""}.`;
      const rows = preds.map((p, i) => ({ ...p, monthly: records[i].rec.MonthlyCharges, tenure: records[i].rec.tenure, contract: records[i].rec.Contract, churn: records[i].churn, id: p.customerID ?? `row ${i + 1}` }));
      drawResults(rows, ev, records.map((x) => x.rec));
    } catch (e) {
      status.className = "error";
      status.textContent = e.message;
    }
  }

  function drawResults(rows, ev, recs) {
    const H = ev.economics.horizon;
    const offers = rows.filter((r) => r.send_offer);
    const atRisk = rows.reduce((a, r) => a + r.churn_probability * H * r.monthly, 0);
    const expected = offers.reduce((a, r) => a + r.expected_value, 0);
    const labelled = rows.filter((r) => r.churn === "Yes" || r.churn === "No");
    const figs = [
      ["Customers scored", fmt.int(rows.length), `mean churn risk ${fmt.pct(rows.reduce((a, r) => a + r.churn_probability, 0) / rows.length)}`],
      ["Offers recommended", fmt.int(offers.length), `${fmt.pct(offers.length / rows.length)} of the file`],
      ["Expected retention profit", fmt.moneyK(expected), "Σ expected value of the offers sent"],
      [`Revenue at risk, ${H} months`, fmt.moneyK(atRisk), "Σ P(churn) × bill × horizon, if nobody is contacted"],
    ];
    const extra = [];
    if (labelled.length === rows.length) {
      const hit = offers.filter((r) => r.churn === "Yes").length, churners = rows.filter((r) => r.churn === "Yes").length;
      extra.push(h("p", { class: "callout" }, h("b", { text: "This file has outcomes. " }),
        `${fmt.pct(offers.length ? hit / offers.length : 0)} of recommended offers went to customers who really churned (precision); the offers reach ${fmt.pct(churners ? hit / churners : 0)} of all churners (recall). `,
        "Note: the sample is drawn from the same dataset the model was built from, so this is not an independent test."));
    }

    // Sortable, paginated table.
    const cols = [
      { key: "id", label: "Customer", get: (r) => r.id, cell: (r) => h("span", { class: "mono", style: { whiteSpace: "nowrap" }, text: r.id }) },
      { key: "p", label: "Churn risk", r: true, get: (r) => r.churn_probability, cell: (r) => h("span", {}, fmt.pct(r.churn_probability)) },
      { key: "m", label: "Bill", r: true, get: (r) => r.monthly, cell: (r) => fmt.money2(r.monthly) },
      { key: "ev", label: "Expected value", r: true, get: (r) => r.expected_value, cell: (r) => fmt.money2(r.expected_value) },
      { key: "d", label: "Decision", get: (r) => (r.send_offer ? 1 : 0), cell: (r) => r.send_offer ? h("span", { class: "tag accent", text: "Send offer" }) : h("span", { class: "muted", text: "Hold" }) },
      { key: "why", label: "Top reasons", get: (r) => r.top_reasons[0]?.feature ?? "", cell: (r) => h("span", { class: "ink2", text: reasonText(r.top_reasons) }) },
    ];
    let sortKey = "ev", dir = -1, page = 0;
    const per = 25;
    const tbody = h("tbody");
    const pager = h("div", { class: "pager" });
    const thead = h("thead", {}, h("tr", {}, cols.map((c) => {
      const th = h("th", { class: c.r ? "r" : null, scope: "col" });
      th.append(h("button", { type: "button", text: c.label, onclick: () => { dir = sortKey === c.key ? -dir : -1; sortKey = c.key; page = 0; draw(); } }));
      c.th = th;
      return th;
    })));
    function draw() {
      const col = cols.find((c) => c.key === sortKey);
      const sorted = [...rows].sort((a, b) => { const x = col.get(a), y = col.get(b); return (x < y ? -1 : x > y ? 1 : 0) * dir; });
      cols.forEach((c) => c.th.setAttribute("aria-sort", c.key === sortKey ? (dir < 0 ? "descending" : "ascending") : "none"));
      const pages = Math.ceil(sorted.length / per);
      tbody.replaceChildren(...sorted.slice(page * per, page * per + per).map((r) =>
        h("tr", {}, cols.map((c) => h("td", { class: c.r ? "r" : null }, c.cell(r))))));
      pager.replaceChildren(
        h("span", { text: `${fmt.int(page * per + 1)}–${fmt.int(Math.min(sorted.length, (page + 1) * per))} of ${fmt.int(sorted.length)}` }),
        h("span", { class: "toolbar" },
          h("button", { type: "button", class: "btn", text: "Previous", disabled: page === 0, onclick: () => { page--; draw(); } }),
          h("button", { type: "button", class: "btn", text: "Next", disabled: page >= pages - 1, onclick: () => { page++; draw(); } })));
    }
    draw();

    const download = h("button", { type: "button", class: "btn primary", text: "Download decisions (.csv)", onclick: () => {
      const esc = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
      const lines = [["customerID", "churn_probability", "expected_value", "send_offer", "reason_1", "reason_2", "reason_3"].join(",")];
      for (const r of rows) lines.push([r.id, r.churn_probability, r.expected_value, r.send_offer, ...[0, 1, 2].map((k) => r.top_reasons[k] ? `${r.top_reasons[k].feature}=${r.top_reasons[k].value}` : "")].map(esc).join(","));
      const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
      const a = h("a", { href: url, download: "retention_decisions.csv" });
      document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } });

    const driftBox = h("div", { class: "stack" }, h("div", { class: "note", text: recs.length < 30 ? "A drift check needs at least 30 customers." : "Comparing with the reference customers…" }));
    out.replaceChildren(
      block("Summary", [h("p", { text: "Revenue at risk is what churn would cost if nobody were contacted. Expected retention profit is what the recommended offers should return after their cost — a much smaller, actionable number." })],
        [figures(figs), ...extra]),
      block("Decisions", [h("p", { text: "Highlighted rows get an offer. Sort by any column; the export contains every row." })],
        [h("div", { class: "toolbar" }, download), h("div", { class: "table-wrap" }, h("table", {}, thead, tbody)), pager]),
      block("Drift check", [h("p", { text: "Population Stability Index of each feature and of the model score against the model's reference customers. Ticks mark 0.10 (watch) and 0.25 (act)." }),
        h("p", { class: "note", text: "A drifted file is still scored, but its probabilities deserve less trust until the model is checked on fresh outcomes." })], [driftBox]),
    );
    if (recs.length >= 30) {
      api("/drift", recs.slice(0, 10000)).then((d) => {
        driftBox.replaceChildren(
          h("div", { class: "toolbar" }, statusTag(d.status, `Overall: ${d.status}`), h("span", { class: "note", text: `${fmt.int(d.n)} customers vs reference` })),
          psiBars(d.features));
      }).catch((e) => driftBox.replaceChildren(h("p", { class: "error", text: e.message })));
    }
  }
}

function psiBars(features, scaleMax) {
  const color = { stable: "var(--good)", moderate: "var(--warning)", major: "var(--critical)" };
  const max = scaleMax ?? Math.max(0.3, ...features.map((f) => f.psi)) * 1.05;
  return barList(features.map((f) => ({
    name: f.feature === "model score" ? "Model score" : LABELS[f.feature] || f.feature, value: f.psi, color: color[f.status],
    tip: () => [h("div", { class: "tl", text: f.feature }), tipRow(fmt.num(f.psi, 3), "PSI"), tipRow(titleCase(f.status), "status"), tipRow(String(f.p_value), "p-value (KS / χ²)")],
  })), { max, ticks: [0.1, 0.25], fmt: (v) => fmt.num(v, 3) });
}

/* ------------------------------------------------------------------ 03 · Retention strategy */
function rng(seed) { // mulberry32
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

async function renderStrategy(view) {
  loading(view);
  const ev = await evidence();
  const P = ev.test_predictions, n = P.length;
  const B = 300, rand = rng(7);
  const idx = Array.from({ length: B }, () => { const a = new Uint16Array(n); for (let i = 0; i < n; i++) a[i] = Math.floor(rand() * n); return a; });
  const defaults = ev.economics;
  const st = { offer_cost: defaults.offer_cost, save_rate: defaults.save_rate, horizon: defaults.horizon, budget: n };

  const sliders = [
    ["offer_cost", "Offer cost", 0, 300, 5, (v) => fmt.money(v)],
    ["save_rate", "Save rate", 0.05, 0.8, 0.05, (v) => fmt.pct(v, 0)],
    ["horizon", "Value horizon", 3, 36, 1, (v) => `${v} months`],
    ["budget", "Contact budget", 0, n, 1, (v) => (v >= n ? "Unlimited" : `${fmt.int(v)} customers`)],
  ].map(([key, label, min, max, step, f]) => {
    const out = h("output", { text: f(st[key]) });
    const range = h("input", { type: "range", min, max, step, value: st[key], "aria-label": label });
    range.addEventListener("input", () => { st[key] = Number(range.value); out.textContent = f(st[key]); update(); });
    return h("div", { class: "slider" }, h("div", { class: "top" }, h("span", { text: label }), out), range);
  });
  const reset = h("button", { type: "button", class: "link", text: "Reset to model assumptions", onclick: () => renderStrategy(view) });

  // The served policy's margin was tuned for the model's own economics; with any other economics the
  // Bayes-optimal margin for calibrated probabilities ($0) applies.
  const served = ev.threshold.margin;
  const isDefault = () => st.offer_cost === defaults.offer_cost && Math.abs(st.save_rate - defaults.save_rate) < 1e-9 && st.horizon === defaults.horizon;
  const marginFor = () => (isDefault() ? served : 0);
  const marginNote = h("span", { class: "note" });
  const figs = h("div");
  const policyTable = h("div");
  const curve = h("div");
  const ciTxt = (ci) => `${fmt.moneyK(ci[0])} – ${fmt.moneyK(ci[1])}`;

  function compute() {
    const { offer_cost: c, save_rate: sr, horizon: H, budget } = st;
    const margin = marginFor();
    const evs = P.map((x) => x.p * sr * H * x.monthly - c);
    const gainIf = P.map((x) => x.churned * sr * H * x.monthly - c); // realised $ if contacted
    const order = evs.map((_, i) => i).sort((a, b) => evs[b] - evs[a]);
    const evTarget = new Uint8Array(n);
    let k = 0;
    for (const i of order) { if (evs[i] <= margin || k >= budget) break; evTarget[i] = 1; k++; }
    const policies = [
      ["Expected-value targeting", evTarget, true],
      ["Contact if P(churn) > 0.5", Uint8Array.from(P, (x) => x.p > 0.5)],
      ["Contact everyone", new Uint8Array(n).fill(1)],
      ["Perfect foresight (ceiling)", Uint8Array.from(P, (x) => x.churned && sr * H * x.monthly > c)],
    ].map(([name, t, main]) => {
      let profit = 0, contacted = 0, tp = 0;
      for (let i = 0; i < n; i++) if (t[i]) { profit += gainIf[i]; contacted++; tp += P[i].churned; }
      const draws = idx.map((ix) => { let s2 = 0; for (let j = 0; j < n; j++) { const i = ix[j]; if (t[i]) s2 += gainIf[i]; } return s2; }).sort((a, b) => a - b);
      const churners = P.reduce((a, x) => a + x.churned, 0);
      return { name, main, profit, contacted, precision: contacted ? tp / contacted : null, recall: tp / churners, ci: [draws[Math.floor(B * 0.025)], draws[Math.floor(B * 0.975)]] };
    });
    const atRisk = P.reduce((a, x) => a + x.p * H * x.monthly, 0);
    const expected = order.reduce((a, i) => a + (evTarget[i] ? evs[i] : 0), 0);
    // Profit curve: contact the top-k by expected value.
    const pts = [[0, 0]], exp = [[0, 0]];
    let cr = 0, ce = 0;
    const stepN = Math.max(1, Math.floor(n / 140));
    order.forEach((i, j) => { cr += gainIf[i]; ce += evs[i]; if ((j + 1) % stepN === 0 || j === n - 1) { pts.push([j + 1, cr]); exp.push([j + 1, ce]); } });
    return { policies, atRisk, expected, pts, exp, k };
  }

  function update() {
    const r = compute();
    marginNote.textContent = isDefault()
      ? `Economics are assumptions, not data. At the model's own economics (${fmt.money(defaults.offer_cost)} offer, ${fmt.pct(defaults.save_rate, 0)} save rate, ${defaults.horizon}-month value) targeting uses the served decision margin, ${fmt.money(served)}: customers with expected value above it are contacted, highest first, up to the budget — the same policy the API applies.`
      : `Economics are assumptions, not data. These differ from the model's (${fmt.money(defaults.offer_cost)} / ${fmt.pct(defaults.save_rate, 0)} / ${defaults.horizon} mo), so the served margin of ${fmt.money(served)} no longer applies; targeting uses $0, the Bayes-optimal margin for calibrated probabilities.`;
    const main = r.policies[0];
    figs.replaceChildren(figures([
      [`Revenue at risk · ${st.horizon} mo`, fmt.moneyK(r.atRisk), "Σ P(churn) × bill × horizon — the problem's size, no action taken"],
      ["Expected retention profit", fmt.moneyK(r.expected), `what the model expects the ${fmt.int(main.contacted)} offers to return after cost`],
      ["Realised profit", fmt.moneyK(main.profit), `95% CI ${ciTxt(main.ci)} on actual outcomes`],
      ["Offers sent", fmt.int(main.contacted), `${fmt.pct(main.contacted / n)} of ${fmt.int(n)} customers`],
    ]));
    policyTable.replaceChildren(table([
      { label: "Policy", cell: (p) => p.main ? h("b", { text: p.name }) : p.name },
      { label: "Contacted", r: true, cell: (p) => fmt.int(p.contacted) },
      { label: "Precision", r: true, cls: "wide-only", cell: (p) => fmt.pct(p.precision) },
      { label: "Recall", r: true, cls: "wide-only", cell: (p) => fmt.pct(p.recall) },
      { label: "Realised profit", r: true, cell: (p) => valueCI(p.profit, p.ci, fmt.moneyK) },
    ], r.policies, { rowClass: (p) => (p.main ? "hl" : null) }));
    curve.replaceChildren(
      legend([["Realised profit (true outcomes)", "var(--s1)"], ["Expected profit (model)", "var(--s2)"]]),
      chart((w) => lineChart(w, {
        height: 280, label: "Profit as more customers are contacted", yZero: true, xLabel: "Customers contacted, highest expected value first",
        series: [{ name: "Realised", color: css("--s1"), points: r.pts }, { name: "Expected", color: css("--s2"), points: r.exp }],
        refs: [{ x: main.contacted, label: `Policy · ${fmt.int(main.contacted)}`, anchor: main.contacted > n * 0.6 ? "end" : "start" }],
        xFmt: (v) => fmt.int(v), yFmt: fmt.moneyK, tipX: (v) => `Top ${fmt.int(v)} contacted`,
      })));
  }

  const sens = ev.test.sensitivity;
  const rates = [...new Set(sens.map((x) => x.save_rate))].sort((a, b) => b - a);
  const costs = [...new Set(sens.map((x) => x.offer_cost))].sort((a, b) => a - b);
  const maxAbs = Math.max(...sens.map((x) => Math.abs(x.model)));
  const heat = h("div", { class: "heat", style: { gridTemplateColumns: `72px repeat(${costs.length}, minmax(0, 1fr))` } },
    h("div", { class: "hd row axis-title", text: "Save rate" }), h("div", { class: "hd axis-title", style: { gridColumn: `span ${costs.length}` }, text: "Offer cost per customer contacted" }),
    h("div", { class: "hd row", text: "" }), costs.map((c) => h("div", { class: "hd", text: fmt.money(c) })),
    rates.map((rate) => [h("div", { class: "hd row", text: fmt.pct(rate, 0) }), costs.map((c) => {
      const cell = sens.find((x) => x.save_rate === rate && x.offer_cost === c);
      const t = cell.model / maxAbs, pct = Math.round(Math.abs(t) * 85);
      const el = h("div", { class: "cell", tabindex: 0, text: fmt.moneyK(cell.model),
        style: { background: `color-mix(in oklab, ${t >= 0 ? "var(--neg)" : "var(--pos)"} ${pct}%, var(--mid))`, color: pct > 45 ? "#fff" : "var(--ink)" } });
      return hover(el, () => [h("div", { class: "tl", text: `Save rate ${fmt.pct(rate, 0)} · offer ${fmt.money(c)}` }),
        tipRow(fmt.money(cell.model), "EV targeting"), tipRow(ciText(cell.model_ci, fmt.moneyK), "95% CI"),
        tipRow(fmt.money(cell.everyone), "contact everyone"), tipRow(fmt.int(cell.contacted), "contacted")]);
    })]));

  view.replaceChildren(
    pageHead("03 · Retention strategy & simulation", "What should the campaign look like?",
      `Change the campaign economics and see how targeting holds up on the ${fmt.int(n)} held-out customers, whose real outcomes are known. Every profit figure carries a 95% bootstrap interval.`),
    h("div", { style: { paddingTop: "24px" } }, h("div", { class: "controls" }, sliders), h("div", { class: "toolbar", style: { justifyContent: "space-between", paddingTop: "10px" } },
      marginNote, reset)),
    block("Outcome", [h("p", { text: "Revenue at risk sizes the problem; it is not money a campaign can recover. Expected retention profit is the model's forecast for the offers it sends; realised profit is what those offers actually earned on known outcomes." })], [figs]),
    block("Policies compared", [h("p", { text: "Same customers, same economics, different targeting rules. Perfect foresight is the ceiling: contact exactly the churners whose bill makes saving them worth the offer." })], [policyTable]),
    block("Profit curve", [h("p", { text: "Contacting customers in order of expected value. Realised profit peaks and then falls: past that point each extra offer costs more than it saves." })], [curve]),
    block("Sensitivity", [h("p", { text: "Realised profit of re-optimised expected-value targeting across save-rate and offer-cost assumptions (12-month horizon). Blue is profit, red is loss; hover for the interval and the contact-everyone comparison." }),
      h("p", { class: "note", text: "Computed at training time from the held-out customers." })], [h("div", { class: "table-wrap" }, h("div", { style: { minWidth: "380px" } }, heat))]),
  );
  update();
}

/* ------------------------------------------------------------------ 04 · Model evidence */
const MODEL_NAMES = { logistic: "Logistic regression", xgboost_default: "XGBoost, hand-set", xgboost_tuned: "XGBoost, Optuna-tuned" };
const METRIC_INFO = {
  roc_auc: ["ROC-AUC", 1, (v) => fmt.num(v, 3)], pr_auc: ["PR-AUC", 1, (v) => fmt.num(v, 3)], log_loss: ["Log loss", -1, (v) => fmt.num(v, 4)],
  brier: ["Brier", -1, (v) => fmt.num(v, 4)], ece: ["ECE", -1, (v) => fmt.num(v, 3)], profit: ["Profit", 1, fmt.moneyK],
};

async function renderEvidence(view) {
  loading(view);
  const ev = await evidence();
  const chosen = ev.selection.chosen, ref = ev.test.reference_model, models = ev.test.models;
  const nm = (k) => MODEL_NAMES[k] || k;

  // Selection
  const cands = Object.entries(ev.selection.candidates);
  const best = Math.min(...cands.map(([, c]) => c.cv_log_loss));
  const selTable = table([
    { label: "Candidate", cell: ([k]) => [nm(k), k === chosen ? h("span", { class: "tag accent", style: { marginLeft: "8px" }, text: "selected" }) : null] },
    { label: "CV log loss", r: true, cell: ([, c]) => fmt.num(c.cv_log_loss, 4) },
    { label: "Δ vs best", r: true, cell: ([, c]) => (c.cv_log_loss === best ? "—" : fmt.signed(c.cv_log_loss - best, 4)) },
    { label: "Search", cell: ([k, c]) => { const o = ev.selection.optuna[c.family]; return k === "xgboost_default" ? h("span", { class: "muted", text: "fixed parameters" }) : h("span", { class: "ink2", text: `Optuna TPE · ${o.n_trials} trials · ${o.pruned} pruned` }); } },
  ], cands, { rowClass: ([k]) => (k === chosen ? "hl" : null) });
  const hist = ev.selection.optuna.xgboost.history;
  let run = Infinity;
  const bestLine = hist.map((t) => [t.trial, (run = Math.min(run, t.value))]);
  const trialChart = chart((w) => lineChart(w, {
    height: 180, label: "Optuna trial history", xLabel: "Trial",
    series: [{ name: "Best so far", color: css("--s1"), points: bestLine }, { name: "Trial", color: css("--muted"), kind: "dots", points: hist.map((t) => [t.trial, t.value]), r: 2.5 }],
    yFmt: (v) => fmt.num(v, 3), tipY: (v) => fmt.num(v, 4), tipX: (v) => `Trial ${v}`,
    refs: [{ y: ev.selection.candidates.logistic.cv_log_loss, label: "logistic regression", color: css("--s2") }],
    yDomain: (() => { const ys = hist.map((t) => t.value).sort((a, b) => a - b); return [ys[0] - 0.002, Math.min(ys[Math.floor(ys.length * 0.9)], ys[0] + 0.03)]; })(),
  }));
  const abl = ev.selection.ablation_cv_log_loss;

  // Test comparison
  const mkeys = Object.keys(METRIC_INFO);
  const perfTable = table([
    { label: "Model", cell: ([k]) => [nm(k), k === chosen ? h("span", { class: "tag accent", style: { marginLeft: "8px" }, text: "served" }) : null] },
    ...mkeys.map((m) => ({ label: METRIC_INFO[m][0], r: true, cell: ([, v]) => valueCI(v[m].value, v[m].ci, METRIC_INFO[m][2]) })),
  ], Object.entries(models), { rowClass: ([k]) => (k === chosen ? "hl" : null) });
  const diff = models[chosen][`vs_${ref}`];
  const verdicts = diff ? mkeys.map((m) => {
    const d = diff[m], [lo, hi] = d.ci, better = METRIC_INFO[m][1];
    const sig = lo > 0 || hi < 0;
    const good = sig && ((better > 0 && lo > 0) || (better < 0 && hi < 0));
    return { m, d, sig, good };
  }) : [];
  const nSig = verdicts.filter((v) => v.sig).length;
  const diffTable = diff ? table([
    { label: "Metric", cell: (v) => METRIC_INFO[v.m][0] },
    { label: `${nm(chosen)} − ${nm(ref)}`, r: true, cell: (v) => v.m === "profit" ? fmt.signedMoney(v.d.diff) : fmt.signed(v.d.diff, 4) },
    { label: "95% CI", r: true, cell: (v) => h("span", { class: "ci", text: v.m === "profit" ? `${fmt.signedMoney(v.d.ci[0])} to ${fmt.signedMoney(v.d.ci[1])}` : `${fmt.signed(v.d.ci[0], 4)} to ${fmt.signed(v.d.ci[1], 4)}` }) },
    { label: "Verdict", cell: (v) => v.sig ? statusTag(v.good ? "pass" : "fail", v.good ? "Better" : "Worse") : h("span", { class: "muted", text: "No detectable difference" }) },
  ], verdicts) : null;

  // Calibration
  const cal = ev.calibration;
  const reli = chart((w) => reliability(w, cal.reliability_raw, cal.reliability));

  // Threshold
  const pbm = Object.entries(ev.threshold.profit_by_margin_on_calibration).map(([k, v]) => [Number(k), v]).sort((a, b) => a[0] - b[0]);
  const thr = chart((w) => lineChart(w, {
    height: 220, label: "Calibration-split profit by decision margin", xLabel: "Decision margin: contact when expected value exceeds this ($)",
    series: [{ name: "Profit on calibration split", color: css("--s1"), points: pbm }],
    refs: [{ x: ev.threshold.margin, label: `tuned margin ${fmt.money(ev.threshold.margin)}` }],
    xFmt: (v) => fmt.money(v), yFmt: fmt.moneyK, tipX: (v) => `Margin ${fmt.money(v)}`,
  }));

  // Subgroups
  const groupsOf = [...new Set(ev.subgroups.map((s2) => s2.column))];
  const subRows = groupsOf.flatMap((col) => [
    h("tr", { class: "group" }, h("td", { colspan: 7, text: col === "tenure_band" ? "Tenure" : LABELS[col] || col })),
    ...ev.subgroups.filter((s2) => s2.column === col).map((s2) => {
      const width = s2.roc_auc_ci ? s2.roc_auc_ci[1] - s2.roc_auc_ci[0] : null;
      const flags = [];
      if (Math.abs(s2.calibration_gap) > 0.05) flags.push(statusTag("moderate", `gap ${fmt.signed(s2.calibration_gap * 100, 1)} pts`));
      if (width != null && width > 0.15) flags.push(h("span", { class: "tag", text: "wide CI" }));
      if (s2.n < 150) flags.push(h("span", { class: "tag", text: "small n" }));
      return h("tr", {},
        h("td", { text: s2.level }), h("td", { class: "r", text: fmt.int(s2.n) }), h("td", { class: "r", text: fmt.pct(s2.churn_rate) }),
        h("td", { class: "r", text: fmt.pct(s2.mean_predicted) }),
        h("td", { class: "r" }, s2.roc_auc != null ? valueCI(s2.roc_auc, s2.roc_auc_ci, (v) => fmt.num(v, 3)) : "—"),
        h("td", { class: "r", text: fmt.pct(s2.recall) }),
        h("td", {}, h("span", { class: "toolbar", style: { gap: "4px" } }, flags)));
    }),
  ]);
  const subTable = h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, ["Segment", "n", "Churn rate", "Predicted", "ROC-AUC", "Recall", "Flags"].map((t, i) => h("th", { class: i && i < 6 ? "r" : null, text: t })))),
    h("tbody", {}, subRows)));

  // Errors
  const er = ev.errors;
  const errSegs = table([
    { label: "Segment", cell: (s2) => s2.segment.replace("tenure_band=", "Tenure ").replace("=", " = ") },
    { label: "n", r: true, cell: (s2) => fmt.int(s2.n) },
    { label: "Wasted-offer rate", r: true, cell: (s2) => fmt.pct(s2.false_offer_rate) },
    { label: "Missed-churner rate", r: true, cell: (s2) => fmt.pct(s2.missed_churner_rate) },
    { label: "Cost of errors", r: true, cell: (s2) => fmt.money(s2.cost) },
  ], er.segments.slice(0, 8));
  const costliest = table([
    { label: "Error", cell: (c) => c.kind === "wasted offer" ? "Wasted offer" : "Missed churner" },
    { label: "Risk", r: true, cell: (c) => fmt.pct(c.probability) },
    { label: "Bill", r: true, cell: (c) => fmt.money2(c.monthly) },
    { label: "Tenure", r: true, cell: (c) => `${c.tenure} mo` },
    { label: "Contract", cell: (c) => c.contract },
    { label: "Cost", r: true, cell: (c) => fmt.money(c.cost) },
    { label: "Model's top reasons", cell: (c) => h("span", { class: "ink2", text: reasonText(c.reasons) }) },
  ], er.costliest.slice(0, 6));

  // Explanations
  const ex = ev.explain;
  const groupBars = barList(Object.entries(ex.importance_group).map(([k, v]) => ({ name: groupLabel(k), value: v,
    tip: () => [h("div", { class: "tl", text: k }), tipRow(fmt.num(v, 3), "mean |SHAP|, log-odds"), h("div", { class: "tl", text: `Members: ${Object.entries(ex.groups).filter(([, g]) => g === k).map(([f]) => LABELS[f] || f).join(", ")}` })] })), { fmt: (v) => fmt.num(v, 3), wideNames: true });
  const featBars = barList(Object.entries(ex.importance_feature).map(([k, v]) => ({ name: LABELS[k] || k, value: v, color: "var(--s1)",
    tip: () => [h("div", { class: "tl", text: k }), tipRow(fmt.num(v, 3), "mean |SHAP|, log-odds"), h("div", { class: "tl", text: `Group: ${groupLabel(ex.groups[k])}` })] })), { fmt: (v) => fmt.num(v, 3) });
  const topFeats = Object.keys(ex.importance_feature).slice(0, 8);
  const swarm = chart((w) => beeswarm(w, ex.beeswarm, topFeats));
  const assoc = h("div", { class: "table-wrap" }, associationHeat(ex.association, ex.groups));

  view.replaceChildren(
    pageHead("04 · Model evidence & explainability", "How far can these predictions be trusted?",
      `Every number here comes from the ${fmt.int(ev.splits.test.n)} test customers, scored once after model selection, calibration and threshold tuning were finished on separate customers.`),
    block("Strict separation", [h("p", { text: "Customers are assigned to a split by a hash of their ID, so the assignment never changes as data grows and no stage sees another stage's customers." })], [
      table([{ label: "Split", cell: ([k]) => h("b", { text: titleCase(k) }) }, { label: "Customers", r: true, cell: ([, v]) => fmt.int(v.n) },
        { label: "Churn rate", r: true, cell: ([, v]) => fmt.pct(v.churn_rate) },
        { label: "Used for", cell: ([k]) => ({ train: "Model selection — Optuna, 5-fold CV", calibration: "Calibrator choice & fit · decision threshold", test: "Final evaluation, touched once" })[k] }],
      Object.entries(ev.splits))]),
    block("Model selection", [h("p", { text: `Chosen by 5-fold cross-validated log loss on the train split — log loss, not AUC, because the profit rule multiplies by the probabilities.` }),
      h("p", { class: "note", text: `Dropping TotalCharges (≈ tenure × bill): CV log loss ${fmt.num(abl.without_total_charges, 4)} without vs ${fmt.num(abl.with_total_charges, 4)} with. Kept out so explanations don't split credit between two copies of tenure.` })],
      [selTable, h("div", {}, h("div", { class: "chart-title" }, h("h3", { text: "XGBoost search, CV log loss per trial" })),
        legend([["Best so far", "var(--s1)"], ["Trial", "var(--muted)", "dot"], ["Logistic regression", "var(--s2)"]]), trialChart)]),
    block("Held-out performance", [h("p", { text: "Point estimates with 95% bootstrap intervals (2,000 resamples). Each model got the same calibration and threshold treatment." })], [perfTable,
      diff ? h("div", { class: "stack" },
        h("p", { class: "callout" }, h("b", { text: nSig === 0 ? `The served model is statistically indistinguishable from ${nm(ref).toLowerCase()}. ` : `${nSig} of ${verdicts.length} metrics differ significantly from ${nm(ref).toLowerCase()}. ` }),
          "Paired bootstrap on the same resampled customers. An interval spanning zero means this test set cannot tell the models apart — the simpler baseline is a legitimate alternative."),
        diffTable) : null]),
    block("Calibration", [h("p", { text: `Calibrator chosen by cross-validated log loss on the calibration split: ${cal.method}. Expected calibration error on test: ${fmt.num(cal.ece_raw, 3)} raw → ${fmt.num(models[chosen].ece.value, 3)} calibrated.` }),
      h("p", { class: "note", text: Object.entries(cal.cv_log_loss).map(([k, v]) => `${k} ${fmt.num(v, 4)}`).join(" · ") })],
      [legend([["Calibrated (served)", "var(--s1)", "dot"], ["Raw model score", "var(--s2)", "dot"], ["Perfect calibration", "var(--muted)"]]), reli,
        h("p", { class: "note", text: "Each point is a tenth of the test customers, binned by predicted risk. Vertical lines are Wilson 95% intervals on the observed churn rate." })]),
    block("Decision threshold", [h("p", { text: `The expected-value margin was tuned on cross-fitted calibrated probabilities of the calibration split. It landed on ${fmt.money(ev.threshold.margin)}; ${fmt.money(ev.threshold.bayes_optimal_margin)} is the theoretical optimum for well-calibrated probabilities.` }),
      h("p", { class: "note", text: `Test profit at the tuned margin ${fmt.money(ev.threshold.test_profit_at_tuned_margin)}; at $0 ${fmt.money(ev.threshold.test_profit_at_zero_margin)}.` })], [thr]),
    block("Subgroups", [h("p", { text: "Discrimination and calibration per segment. Small segments carry wide intervals — read them as weak evidence, not as fairness verdicts." }),
      h("p", { class: "note", text: "Flags: calibration gap over 5 points, AUC interval wider than 0.15, or fewer than 150 customers." })], [subTable]),
    block("Error analysis", [h("p", { text: "Where the policy's mistakes cost money. A wasted offer costs the offer; a missed churner costs the profit an offer would have made." })], [
      figures([["Wasted offers", fmt.int(er.wasted_offers.count), `${fmt.money(er.wasted_offers.cost)} spent on customers who stayed`],
        ["Missed profitable churners", fmt.int(er.missed_profitable_churners.count), `${fmt.money(er.missed_profitable_churners.forgone_profit)} forgone`],
        ["Skipped by design", fmt.int(er.missed_unprofitable_churners), "churners on plans too cheap to be worth an offer"]]),
      h("h3", { text: "Costliest segments" }), errSegs, h("h3", { text: "Costliest individual errors" }), costliest]),
    block("Global explanations", [h("p", { text: "Mean absolute SHAP value per correlated group and per feature, on the test customers (log-odds)." }),
      h("p", { class: "note", text: "SHAP shares credit among correlated features somewhat arbitrarily, so the group view is the trustworthy one. SHAP explains the model, not the customer — it is not causal." })],
      [h("div", { class: "split" }, h("div", {}, h("h3", { text: "By correlated group" }), groupBars), h("div", {}, h("h3", { text: "By feature" }), featBars))]),
    block("Feature effects", [h("p", { text: "Each dot is one test customer: how far the feature pushed their risk. Numeric features are shaded by value; categorical features are grey and labelled by level. Hover any dot for its value." })],
      [rampKey(), swarm]),
    block("Correlated features", [h("p", { text: `Association between features: |Spearman| for numeric pairs, Cramér's V for categorical pairs, correlation ratio for mixed. Features above ${ex.group_threshold} (average linkage) are explained as one group.` }),
      h("p", { class: "note", text: "Ordered by group. Internet service and its six add-ons are tightly linked through the \"No internet service\" level; tenure and contract move together." })], [assoc]),
  );
}

function reliability(w, raw, calb) {
  const H = Math.min(340, Math.max(240, w * 0.55)), m = { t: 12, r: 16, b: 40, l: 48 };
  const pw = w - m.l - m.r, ph = H - m.t - m.b;
  const mx = Math.min(1, Math.ceil(Math.max(...[...raw, ...calb].flatMap((r) => [r.predicted, r.ci[1]])) * 10 + 0.5) / 10);
  const sx = (v) => m.l + (v / mx) * pw, sy = (v) => m.t + ph - (v / mx) * ph;
  const svg = s("svg", { viewBox: `0 0 ${w} ${H}`, width: w, height: H, role: "img", "aria-label": "Reliability diagram" });
  const ticks = niceTicks(0, mx, 4).filter((t) => t <= mx + 1e-9);
  for (const t of ticks) {
    svg.append(s("line", { x1: m.l, x2: m.l + pw, y1: sy(t), y2: sy(t), stroke: css("--line") }));
    svg.append(s("text", { x: m.l - 8, y: sy(t) + 3.5, "text-anchor": "end", text: fmt.pct(t, 0) }));
    svg.append(s("text", { x: sx(t), y: m.t + ph + 16, "text-anchor": "middle", text: fmt.pct(t, 0) }));
  }
  svg.append(s("text", { x: m.l + pw / 2, y: H - 4, "text-anchor": "middle", class: "ink", text: "Predicted churn probability" }));
  svg.append(s("line", { x1: sx(0), y1: sy(0), x2: sx(mx), y2: sy(mx), stroke: css("--muted"), "stroke-width": 1 }));
  const series = [[raw, css("--s2"), "Raw model score", 3], [calb, css("--s1"), "Calibrated", -3]];
  for (const [pts, color, name, off] of series) {
    for (const p of pts) {
      const x = sx(p.predicted) + off;
      svg.append(s("line", { x1: x, x2: x, y1: sy(p.ci[0]), y2: sy(p.ci[1]), stroke: color, "stroke-width": 1.5, opacity: 0.55 }));
      svg.append(s("circle", { cx: x, cy: sy(p.observed), r: 4.5, fill: color, stroke: css("--page"), "stroke-width": 2 }));
      const hit = s("circle", { cx: x, cy: sy(p.observed), r: 12, fill: "transparent", tabindex: 0 });
      hover(hit, () => [h("div", { class: "tl", text: name }), tipRow(fmt.pct(p.predicted), "predicted"), tipRow(fmt.pct(p.observed), "observed"),
        tipRow(ciText(p.ci, (v) => fmt.pct(v)), "95% CI"), tipRow(fmt.int(p.n), "customers")]);
      svg.append(hit);
    }
  }
  return svg;
}

// Sequential blue, low → high. On a dark surface low values recede toward the surface, so the ramp runs the other way.
function beeRamp() {
  const ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
  return getComputedStyle(document.documentElement).colorScheme === "dark" ? [...ramp].reverse() : ramp;
}
function rampKey() {
  return h("div", { class: "legend", style: { alignItems: "center" } }, h("span", { text: "Numeric value: low" }),
    h("span", { style: { display: "inline-flex", gap: "2px" } }, beeRamp().map((c) => h("i", { class: "sq", style: { background: c, width: "14px", height: "8px" } }))),
    h("span", { text: "high" }), h("span", { class: "muted", text: "· categorical features in grey, labelled by level" }));
}
function beeswarm(w, data, feats) {
  const narrow = w < 560;
  const labelW = narrow ? 110 : 150, rowH = 44, top = 4, axisH = 30;
  const all = feats.flatMap((f) => data[f].shap);
  const ext = Math.max(...all.map(Math.abs));
  const ticks = niceTicks(-ext, ext, narrow ? 4 : 6);
  const lo = ticks[0], hi = ticks[ticks.length - 1];
  const x0 = labelW, pw = w - x0 - 8;
  const sx = (v) => x0 + ((v - lo) / (hi - lo)) * pw;
  const H = top + feats.length * rowH + axisH;
  const svg = s("svg", { viewBox: `0 0 ${w} ${H}`, width: w, height: H, role: "img", "aria-label": "SHAP beeswarm" });
  for (const t of ticks) {
    svg.append(s("line", { x1: sx(t), x2: sx(t), y1: top, y2: top + feats.length * rowH, stroke: t === 0 ? css("--muted") : css("--line") }));
    svg.append(s("text", { x: sx(t), y: H - 14, "text-anchor": "middle", text: fmt.signed(t, 1).replace("±", "") }));
  }
  svg.append(s("text", { x: x0 + pw / 2, y: H, "text-anchor": "middle", class: "ink", text: "SHAP value (log-odds): ← lowers risk · raises risk →" }));
  const seq = beeRamp();
  const rand = rng(3);
  feats.forEach((f, r) => {
    const cy = top + r * rowH + rowH / 2;
    svg.append(s("text", { x: 0, y: cy + 4, class: "ink", text: LABELS[f] || f }));
    const { values, shap } = data[f];
    const numeric = values.every((v) => typeof v === "number") && new Set(values).size > 2;
    const sorted = numeric ? [...values].sort((a, b) => a - b) : null;
    const g = s("g");
    const pts = shap.map((v, i) => {
      let color = css("--muted");
      if (numeric) {
        const rank = sorted.indexOf(values[i]) / Math.max(1, sorted.length - 1);
        color = seq[Math.min(seq.length - 1, Math.floor(rank * seq.length))];
      }
      const p = { x: sx(v), y: cy + 5 + (rand() - 0.5) * (rowH - 22), v, val: shown(f, values[i]), color };
      g.append(s("circle", { cx: p.x, cy: p.y, r: 2.6, fill: color, opacity: numeric ? 0.9 : 0.55 }));
      return p;
    });
    svg.append(g);
    if (!numeric) {
      // Direct labels at each level's centre, so identity never rests on colour or hover alone.
      const levels = {};
      pts.forEach((q) => { (levels[q.val] ??= []).push(q.x); });
      // Levels whose clumps coincide share one label; a label that would overlap a placed one is left to the tooltip.
      const cents = Object.entries(levels).map(([lv, xsArr]) => ({ lv, n: xsArr.length, x: xsArr.reduce((a, b) => a + b, 0) / xsArr.length }))
        .sort((a, b) => a.x - b.x);
      const merged = [];
      for (const c of cents) {
        const last = merged[merged.length - 1];
        if (last && c.x - last.x < 24) { last.lv += ` / ${c.lv}`; last.n += c.n; } else merged.push({ ...c });
      }
      const placed = [];
      merged.sort((a, b) => b.n - a.n).forEach((c) => {
        const text = c.lv.length > 26 ? c.lv.slice(0, 25) + "…" : c.lv;
        const half = text.length * 2.9 + 6;
        if (placed.some(([l, r]) => c.x - half < r && c.x + half > l)) return;
        placed.push([c.x - half, c.x + half]);
        svg.append(s("text", { x: c.x, y: cy - rowH / 2 + 10, "text-anchor": "middle", style: "font-size:10px", text }));
      });
    }
    const hit = s("rect", { x: x0, y: cy - rowH / 2, width: pw, height: rowH, fill: "transparent" });
    hit.addEventListener("pointermove", (e) => {
      const b = svg.getBoundingClientRect(), mx = e.clientX - b.left, my = e.clientY - b.top;
      const p = pts.reduce((best, q) => ((q.x - mx) ** 2 + (q.y - my) ** 2 < (best.x - mx) ** 2 + (best.y - my) ** 2 ? q : best));
      showTip(e.clientX, e.clientY, [h("div", { class: "tl", text: LABELS[f] || f }), tipRow(String(p.val), "value"), tipRow(fmt.signed(p.v, 3), "SHAP, log-odds")]);
    });
    hit.addEventListener("pointerleave", hideTip);
    svg.append(hit);
  });
  return svg;
}

function associationHeat(a, groups) {
  const order = [...a.features].sort((x, y) => (groups[x] === groups[y] ? a.features.indexOf(x) - a.features.indexOf(y) : groups[x].localeCompare(groups[y])));
  const ix = Object.fromEntries(a.features.map((f, i) => [f, i]));
  const cell = 20, labelW = 128, n = order.length, W = labelW + n * cell + 4, H = labelW + n * cell + 4;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "Feature association matrix", style: "max-width:none" });
  order.forEach((f, i) => {
    svg.append(s("text", { x: labelW - 6, y: labelW + i * cell + 14, "text-anchor": "end", class: "ink", text: LABELS[f] || f }));
    svg.append(s("text", { transform: `translate(${labelW + i * cell + 14},${labelW - 6}) rotate(-55)`, class: "ink", text: LABELS[f] || f }));
    order.forEach((g, j) => {
      const v = a.matrix[ix[f]][ix[g]];
      const r = s("rect", { x: labelW + j * cell + 1, y: labelW + i * cell + 1, width: cell - 2, height: cell - 2, rx: 2,
        fill: `color-mix(in oklab, var(--s1) ${Math.round(v * 100)}%, var(--sunken))`, tabindex: i === j ? null : 0 });
      if (groups[f] === groups[g] && i !== j && groups[f].includes("+")) r.setAttribute("stroke", css("--ink-2"));
      hover(r, () => [h("div", { class: "tl", text: `${f} × ${g}` }), tipRow(fmt.num(v, 2), "association"),
        groups[f] === groups[g] && f !== g ? h("div", { class: "tl", text: `Same group: ${groupLabel(groups[f])}` }) : null]);
      svg.append(r);
    });
  });
  return h("div", { class: "chart", style: { width: `${W}px` } }, svg);
}

/* ------------------------------------------------------------------ 05 · MLOps & health */
let opsTimer;
async function renderOps(view) {
  loading(view);
  const [health, ops] = await Promise.all([api("/health"), api("/ops")]);
  const latency = h("div");
  const live = h("span", { class: "note" });
  function drawLatency(o) {
    const routes = Object.entries(o.routes);
    live.textContent = `Updated ${new Date().toLocaleTimeString()} · uptime ${fmt.duration(o.uptime_seconds)}`;
    latency.replaceChildren(routes.length ? table([
      { label: "Route", cell: ([k]) => h("span", { class: "mono", text: `POST ${k}` }) },
      { label: "Requests", r: true, cell: ([, v]) => fmt.int(v.requests) },
      { label: "Errors", r: true, cell: ([, v]) => fmt.int(v.errors) },
      { label: "p50", r: true, cell: ([, v]) => fmt.ms(v.p50_ms) },
      { label: "p95", r: true, cell: ([, v]) => fmt.ms(v.p95_ms) },
      { label: "p99", r: true, cell: ([, v]) => fmt.ms(v.p99_ms) },
    ], routes) : h("p", { class: "note", text: "No scoring traffic since this server started. Score a customer or a batch and this table fills in." }));
  }
  drawLatency(ops);
  clearInterval(opsTimer);
  opsTimer = setInterval(async () => {
    if (location.hash !== "#ops" || document.hidden) return;
    try { drawLatency(await api("/ops")); } catch { /* keep the last frame */ }
  }, 5000);

  const reg = ops.registry;
  const events = [...reg.events].reverse();
  const timeline = h("ol", { class: "timeline" }, events.map((e) => {
    const v = reg.versions[e.version];
    return h("li", { class: e.version === reg.champion && e.action !== "registered" ? "champ" : null },
      h("div", { class: "toolbar", style: { gap: "8px" } }, h("b", { text: titleCase(e.action) }), h("span", { class: "mono", text: e.version }),
        e.version === reg.champion && e.action !== "registered" ? h("span", { class: "tag accent", text: "champion" }) : null),
      h("div", { class: "note", text: [fmt.date(e.at), e.previous ? `replaced ${e.previous}` : e.action === "promoted" ? "first champion" : null].filter(Boolean).join(" · ") }),
      v && e.action !== "registered" ? h("div", { class: "note num", text: `${v.family} · AUC ${fmt.num(v.roc_auc, 3)} · ECE ${fmt.num(v.ece, 3)} · profit ${fmt.money(v.profit)} · data ${v.data_sha256}` }) : null);
  }));

  const gate = ops.gate;
  const gateList = h("div", {}, gate.checks.map((c) => h("div", { class: "check" }, icon(c.passed ? "check" : "cross"), h("span", { text: c.check }),
    h("span", { class: "v", text: typeof c.value === "object" ? `${fmt.signed(c.value.diff, 4)} [${fmt.signed(c.value.ci[0], 4)}, ${fmt.signed(c.value.ci[1], 4)}]` : typeof c.value === "number" ? (Math.abs(c.value) > 100 ? fmt.money(c.value) : fmt.num(c.value, 3)) : String(c.value) }))));

  const sim = ops.drift_simulation;
  const simMax = Math.max(0.3, ...sim.scenarios.flatMap((sc) => sc.features.map((f) => f.psi))) * 1.05;
  const simRows = sim.scenarios.map((sc) => h("div", { class: "block sim-row" },
    h("div", {},
      h("div", { class: "toolbar", style: { gap: "8px", marginBottom: "6px" } }, statusTag(sc.status)),
      h("div", { style: { fontWeight: 500 }, text: sc.scenario }),
      h("div", { class: "note", text: `${fmt.int(sc.n)} customers · mean score ${fmt.pct(sc.mean_score)}` }),
      h("div", { class: "note", text: sc.performance ? `Labels valid (real customers resampled): AUC ${fmt.num(sc.performance.roc_auc, 3)} · ECE ${fmt.num(sc.performance.ece, 3)}` : "Features edited, so true outcomes are unknown: performance not reported." })),
    psiBars(sc.features, simMax)));

  const bench = ops.benchmark;
  view.replaceChildren(
    pageHead("05 · MLOps & system health", "What is serving, and is it healthy?",
      "The model in production, the gate it passed, its lineage in the registry, live latency of this server, and the drift monitor."),
    block("Serving", [h("p", { text: "The API refuses to start unless the artifact on disk hashes to the SHA-256 recorded when it was registered, so the model that passed CI is byte-for-byte the model serving." })], [
      h("div", { class: "toolbar" }, h("span", { class: "dot ok" }), h("b", { text: health.status === "ok" ? "Healthy" : health.status }), live),
      figures([["Model version", health.model_version, `${health.family} + ${health.calibrator} calibration`],
        ["Artifact SHA-256", health.artifact_sha256.slice(0, 12), "verified at startup"],
        ["Uptime", fmt.duration(health.uptime_seconds), "since last restart"]])]),
    block("Promotion gate", [h("p", { text: gate.has_champion ? "The challenger was compared with the serving champion on the same held-out customers before promotion." : "First champion: absolute quality floors only. Later challengers must also be non-inferior to the champion on the same held-out customers." })], [
      h("div", { class: "toolbar" }, statusTag(gate.passed ? "pass" : "fail", gate.passed ? "All checks passed" : "Gate failed")), gateList]),
    block("Registry", [h("p", { text: "Versions live in git with their full evaluation reports. Rollback re-points the champion to the previous promoted version after re-verifying its artifact." }),
      h("p", { class: "note mono", text: "python -m churn.registry rollback" })], [timeline]),
    block("Live latency", [h("p", { text: "Measured inside this server over its most recent requests; refreshes every 5 seconds while this page is open." })], [latency]),
    block("Benchmark", [h("p", { text: "Load test of the full HTTP path, recorded by the benchmark script." })], [bench ? benchTable(bench) : h("p", { class: "note", text: "No benchmark report yet. Run the benchmark script to produce reports/benchmark.json." })]),
    h("div", { class: "block wide", style: { paddingBottom: 0 } }, h("div", { class: "stack" },
      h("h2", { text: "Drift monitor" }),
      h("div", { class: "sim-banner", role: "note" }, h("b", { text: "Simulation" }), h("span", { class: "ink2", text: sim.note })),
      h("p", { class: "note", text: `PSI per feature and on the model score against the reference customers. Stable below ${sim.thresholds.moderate}, watch to ${sim.thresholds.major}, act above. Ticks mark both thresholds.` }))),
    ...simRows,
  );
}

function benchTable(b) {
  if (b.by_workers) {
    const rows = Object.entries(b.by_workers).flatMap(([wk, rs]) => rs.map((r) => ({ workers: wk, ...r })));
    const env = b.environment || {};
    return h("div", { class: "stack" },
      h("p", { class: "note", text: [env.machine, env.cpus && `${env.cpus} CPUs`, env.python && `Python ${env.python}`, env.model_version, b.created_at && fmt.date(b.created_at)].filter(Boolean).join(" · ") }),
      b.in_process ? figures([["In-process scoring", `${fmt.int(b.in_process.predictions_per_s)}/s`, `${fmt.int(b.in_process.customers)} customers per call, no HTTP`],
        ["Best HTTP throughput", `${fmt.int(Math.max(...rows.map((r) => r.predictions_per_s)))}/s`, "predictions per second, batched"],
        ["Single-customer p50", fmt.ms(Math.min(...rows.filter((r) => r.batch === 1).map((r) => r.p50_ms))), "one request at a time"]]) : null,
      table([
        { label: "Workers", r: true, cell: (r) => r.workers },
        { label: "Batch", r: true, cell: (r) => fmt.int(r.batch) },
        { label: "Clients", r: true, cell: (r) => fmt.int(r.clients) },
        { label: "Predictions/s", r: true, cell: (r) => fmt.int(r.predictions_per_s) },
        { label: "Requests/s", r: true, cell: (r) => fmt.num(r.requests_per_s, 1) },
        { label: "p50", r: true, cell: (r) => fmt.ms(r.p50_ms) },
        { label: "p95", r: true, cell: (r) => fmt.ms(r.p95_ms) },
        { label: "p99", r: true, cell: (r) => fmt.ms(r.p99_ms) },
      ], rows));
  }
  const arr = Array.isArray(b) ? b : Object.values(b).find(Array.isArray);
  const meta = Array.isArray(b) ? [] : Object.entries(b).filter(([, v]) => v != null && typeof v !== "object");
  const metaObj = Array.isArray(b) ? [] : Object.entries(b).filter(([, v]) => v && typeof v === "object" && !Array.isArray(v));
  const out = [];
  if (meta.length || metaObj.length) {
    out.push(h("p", { class: "note", text: [...meta.map(([k, v]) => `${titleCase(k)}: ${v}`), ...metaObj.flatMap(([, o]) => Object.entries(o).filter(([, v]) => typeof v !== "object").map(([k, v]) => `${titleCase(k)}: ${v}`))].join(" · ") }));
  }
  if (arr?.length) {
    const keys = Object.keys(arr[0]).filter((k) => typeof arr[0][k] !== "object");
    out.push(table(keys.map((k) => ({ label: titleCase(k), r: typeof arr[0][k] === "number", cell: (r) => typeof r[k] === "number" ? (Number.isInteger(r[k]) ? fmt.int(r[k]) : fmt.num(r[k], r[k] < 10 ? 2 : 1)) : String(r[k]) })), arr));
  }
  return h("div", { class: "stack" }, out);
}

/* ------------------------------------------------------------------ routing, health, theme */
const RENDER = { score: renderScore, batch: renderBatch, strategy: renderStrategy, evidence: renderEvidence, ops: renderOps };
const rendered = new Set();
async function route() {
  const name = (location.hash || "#score").slice(1);
  const key = RENDER[name] ? name : "score";
  document.querySelectorAll("[data-view]").forEach((v) => { v.hidden = v.dataset.view !== key; });
  document.querySelectorAll(".nav a").forEach((a) => { if (a.dataset.route === key) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  hideTip();
  const view = document.getElementById(`view-${key}`);
  if (key === "ops" || !rendered.has(key)) {
    rendered.add(key);
    try { await RENDER[key](view); } catch (e) { rendered.delete(key); failed(view, e); }
  }
  document.getElementById("main").focus({ preventScroll: true });
  scrollTo({ top: 0 });
}
addEventListener("hashchange", route);

async function railStatus() {
  const el = document.getElementById("rail-status");
  try {
    const hl = await api("/health");
    el.replaceChildren(h("span", { class: "dot ok" }), h("span", { text: `${hl.model_version} · ${hl.artifact_sha256.slice(0, 7)}` }));
  } catch {
    el.replaceChildren(h("span", { class: "dot bad" }), h("span", { text: "API unreachable" }));
  }
}

const themeBtn = document.getElementById("theme");
const THEMES = ["system", "light", "dark"];
function applyTheme(t) {
  if (t === "system") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
  themeBtn.textContent = `Theme: ${t}`;
  try { t === "system" ? localStorage.removeItem("theme") : localStorage.setItem("theme", t); } catch { /* private mode */ }
}
themeBtn.addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme || "system";
  applyTheme(THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length]);
  // Charts read colours at draw time: redraw the visible view.
  rendered.clear();
  route();
});
applyTheme(document.documentElement.dataset.theme || "system");
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (!document.documentElement.dataset.theme) { rendered.clear(); route(); } });

railStatus();
route();

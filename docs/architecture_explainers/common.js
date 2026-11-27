// Shared header / navigation / theme toggle and small helpers for the architecture explainer pages.
const PAGES = [
  ["index.html", "Overview"],
  ["01_capture.html", "1–3 Capture"],
  ["02_map.html", "4 Map"],
  ["03_detection.html", "5 Detect + track"],
  ["04_embedding.html", "5b Crop + embed"],
  ["05_track_fusion.html", "6 Track fusion"],
  ["06_final_fusion.html", "6–6b Final fusion"],
  ["07_naming.html", "7 Naming"],
  ["08_relations.html", "9 Relations"],
  ["09_query_update.html", "Query + update"],
];

(function () {
  try {
    const t = localStorage.getItem("explainer-theme");
    if (t) document.documentElement.dataset.theme = t;
  } catch (e) { /* storage unavailable */ }
  const here = location.pathname.split("/").pop() || "index.html";
  const header = document.createElement("header");
  header.className = "top";
  header.innerHTML = `<div class="row"><a class="brand" href="index.html">Scene graph pipeline</a><nav>${
    PAGES.map(([f, n]) => `<a href="${f}" class="${f === here ? "on" : ""}">${n}</a>`).join("")
  }</nav><button class="theme" title="Toggle light / dark">◐ theme</button></div>`;
  document.body.prepend(header);
  header.querySelector("button.theme").onclick = () => {
    const dark = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() === "#161615";
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("explainer-theme", next); } catch (e) { /* ignore */ }
    window.dispatchEvent(new Event("themechange"));
  };
  const i = PAGES.findIndex(([f]) => f === here);
  const main = document.querySelector("main");
  if (main && i >= 0) {
    const nav = document.createElement("div");
    nav.className = "next";
    nav.innerHTML = (i > 0 ? `<a href="${PAGES[i - 1][0]}">← ${PAGES[i - 1][1]}</a>` : "<span></span>") +
      (i < PAGES.length - 1 ? `<a href="${PAGES[i + 1][0]}">${PAGES[i + 1][1]} →</a>` : "<span></span>");
    main.appendChild(nav);
  }
})();

// read a CSS variable (so canvases follow the theme)
function cssv(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

// bind a range input to a label and a callback: slider("id", v => ..., fmt)
function slider(id, onChange, fmt = (v) => v) {
  const el = document.getElementById(id);
  const out = document.querySelector(`[data-for="${id}"]`);
  const run = () => { const v = parseFloat(el.value); if (out) out.textContent = fmt(v); onChange(v); };
  el.addEventListener("input", run);
  return run;
}

// set up a canvas for crisp drawing on high-DPI screens; returns the 2D context in CSS pixels
function hidpi(canvas, w, h) {
  const r = window.devicePixelRatio || 1;
  canvas.width = w * r; canvas.height = h * r;
  canvas.style.width = w + "px"; canvas.style.aspectRatio = `${w} / ${h}`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(r, 0, 0, r, 0, 0);
  return ctx;
}

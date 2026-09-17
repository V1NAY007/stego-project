// ===========================================================
// Latent Channel — frontend logic
// ===========================================================
(() => {
  "use strict";

  /* ---------- side panel (hamburger toggle) ---------- */
  const sidePanel = document.getElementById("side-panel");
  const sideOverlay = document.getElementById("side-overlay");
  const menuToggle = document.getElementById("menu-toggle");
  const sideClose = document.getElementById("side-close");

  function openSide() { sidePanel.hidden = false; sideOverlay.hidden = false; }
  function closeSide() { sidePanel.hidden = true; sideOverlay.hidden = true; }
  menuToggle.addEventListener("click", openSide);
  sideClose.addEventListener("click", closeSide);
  sideOverlay.addEventListener("click", closeSide);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSide(); });

  /* ---------- mode switching (list items + side nav items both drive this) ---------- */
  const listItems = document.querySelectorAll(".list-item");
  const sideNavItems = document.querySelectorAll(".side-nav-item");
  const readingPanels = document.querySelectorAll(".reading-content");

  function setMode(mode) {
    listItems.forEach((el) => el.classList.toggle("is-active", el.dataset.mode === mode));
    sideNavItems.forEach((el) => el.classList.toggle("is-active", el.dataset.mode === mode));
    readingPanels.forEach((el) => { el.hidden = el.dataset.panel !== mode; });
  }
  listItems.forEach((el) => el.addEventListener("click", () => { setMode(el.dataset.mode); closeSide(); }));
  sideNavItems.forEach((el) => el.addEventListener("click", () => { setMode(el.dataset.mode); closeSide(); }));

  /* ---------- profile popover ---------- */
  const profileTrigger = document.getElementById("profile-trigger");
  const profilePopover = document.getElementById("profile-popover");
  profileTrigger.addEventListener("click", (e) => {
    e.stopPropagation();
    profilePopover.hidden = !profilePopover.hidden;
  });
  document.addEventListener("click", (e) => {
    if (!profilePopover.hidden && !profilePopover.contains(e.target) && e.target !== profileTrigger) {
      profilePopover.hidden = true;
    }
  });

  /* ---------- status pill + profile popover model info ---------- */
  const statusPill = document.getElementById("status-pill");
  const popoverModel = document.getElementById("popover-model");
  const popoverCapacity = document.getElementById("popover-capacity");
  let modelReady = false;
  let modelMsgLen = null;

  async function refreshStatus() {
    try {
      const r = await fetch("/api/status");
      const data = await r.json();
      if (data.ready) {
        modelReady = true;
        modelMsgLen = data.msg_len;
        statusPill.className = "status-pill status-pill--ready";
        statusPill.innerHTML = `<span class="status-dot"></span> model loaded`;
        popoverModel.textContent = data.model_path || "checkpoint";
        popoverCapacity.textContent = `${data.max_capacity_bytes_repeat1}B`;
      } else {
        modelReady = false;
        statusPill.className = "status-pill status-pill--down";
        statusPill.innerHTML = `<span class="status-dot"></span> no checkpoint`;
        popoverModel.textContent = "none loaded";
        popoverCapacity.textContent = "—";
      }
    } catch (e) {
      statusPill.className = "status-pill status-pill--down";
      statusPill.innerHTML = `<span class="status-dot"></span> server unreachable`;
    }
    updateByteCount();
  }
  refreshStatus();

  /* ---------- dropzone helper ---------- */
  function wireDropzone(zoneId, inputId, previewId) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    const empty = zone.querySelector(".dropzone-empty");

    function setFile(file) {
      if (!file) return;
      const url = URL.createObjectURL(file);
      preview.src = url;
      preview.hidden = false;
      empty.hidden = true;
    }

    zone.addEventListener("click", () => input.click());
    zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
    input.addEventListener("change", () => setFile(input.files[0]));

    ["dragenter", "dragover"].forEach((evt) =>
      zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add("is-dragover"); })
    );
    ["dragleave", "drop"].forEach((evt) =>
      zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.remove("is-dragover"); })
    );
    zone.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files[0];
      if (file) { input.files = e.dataTransfer.files; setFile(file); }
    });

    return { get file() { return input.files[0] || null; } };
  }

  const coverZone = wireDropzone("cover-dropzone", "cover-input", "cover-preview");
  const stegoZone = wireDropzone("stego-dropzone", "stego-input", "stego-preview");

  /* ---------- byte / capacity readout ---------- */
  const messageInput = document.getElementById("message-input");
  const byteCount = document.getElementById("byte-count");
  const capacityNote = document.getElementById("capacity-note");
  const sizeInput = document.getElementById("size-input");

  function updateByteCount() {
    const n = new TextEncoder().encode(messageInput.value).length;
    byteCount.textContent = n;
    if (modelReady && modelMsgLen) {
      const cap = Math.floor((modelMsgLen - 16) / 8);
      capacityNote.textContent = `${Math.max(cap, 0)}B`;
      capacityNote.style.color = n > cap ? "var(--alert)" : "var(--muted)";
    } else {
      capacityNote.textContent = "—";
    }
  }
  messageInput.addEventListener("input", updateByteCount);

  /* ---------- embed ---------- */
  const embedBtn = document.getElementById("embed-btn");
  const embedError = document.getElementById("embed-error");
  const embedResultShell = document.getElementById("embed-result-shell");
  const keyInput = document.getElementById("key-input");

  function showError(el, msg) { el.textContent = msg; el.hidden = false; }
  function hideError(el) { el.hidden = true; }

  embedBtn.addEventListener("click", async () => {
    hideError(embedError);
    const file = coverZone.file;
    if (!file) return showError(embedError, "attach a cover image first");
    if (!keyInput.value) return showError(embedError, "a key is required");
    if (!messageInput.value) return showError(embedError, "enter a message to hide");

    const fd = new FormData();
    fd.append("cover", file);
    fd.append("message", messageInput.value);
    fd.append("key", keyInput.value);
    fd.append("size", sizeInput.value || "0");   // 0 = native resolution, like embed.py

    embedBtn.disabled = true;
    embedBtn.classList.add("is-loading");
    try {
      const r = await fetch("/api/embed", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) {
        showError(embedError, data.message || "embed failed");
        return;
      }
      renderEmbedResult(data);
    } catch (e) {
      showError(embedError, "network error reaching the server");
    } finally {
      embedBtn.disabled = false;
      embedBtn.classList.remove("is-loading");
    }
  });

  function renderEmbedResult(data) {
    const src = `data:image/png;base64,${data.stego_png_b64}`;
    embedResultShell.innerHTML = `
      <img class="result-embed-img" src="${src}" alt="stego output">
      <div class="result-stats">
        <div><div class="stat-label">PSNR vs cover</div><div class="stat-value stat-value--signal">${data.psnr_db} dB</div></div>
        <div><div class="stat-label">embed time</div><div class="stat-value">${data.embed_ms} ms</div></div>
        <div><div class="stat-label">chi² p, cover</div><div class="stat-value">${data.chi_square_cover}</div></div>
        <div><div class="stat-label">chi² p, stego</div><div class="stat-value">${data.chi_square_stego}</div></div>
      </div>
      <div class="result-actions">
        <a class="download-link" href="${src}" download="stego.png">download stego.png</a>
      </div>
    `;
  }

  /* ---------- extract ---------- */
  const extractBtn = document.getElementById("extract-btn");
  const extractError = document.getElementById("extract-error");
  const extractResultShell = document.getElementById("extract-result-shell");
  const extractKeyInput = document.getElementById("extract-key-input");

  extractBtn.addEventListener("click", async () => {
    hideError(extractError);
    const file = stegoZone.file;
    if (!file) return showError(extractError, "attach a stego image first");
    if (!extractKeyInput.value) return showError(extractError, "a key is required");

    const fd = new FormData();
    fd.append("stego", file);
    fd.append("key", extractKeyInput.value);

    extractBtn.disabled = true;
    extractBtn.classList.add("is-loading");
    try {
      const r = await fetch("/api/extract", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) {
        showError(extractError, data.message || "extract failed");
        return;
      }
      renderExtractResult(data);
    } catch (e) {
      showError(extractError, "network error reaching the server");
    } finally {
      extractBtn.disabled = false;
      extractBtn.classList.remove("is-loading");
    }
  });

  function renderExtractResult(data) {
    if (data.decodable) {
      extractResultShell.innerHTML = `
        <div class="extract-result">
          <span class="extract-verdict extract-verdict--ok">recovered</span>
          <div class="extract-message">${escapeHtml(data.text)}</div>
          <div class="extract-meta">${data.byte_length} bytes decoded</div>
        </div>
      `;
    } else {
      extractResultShell.innerHTML = `
        <div class="extract-result">
          <span class="extract-verdict extract-verdict--noise">not valid UTF-8 — wrong key or corrupted</span>
          <div class="extract-message">${data.raw_hex || "(empty)"}</div>
          <div class="extract-meta">${data.byte_length} raw bytes, shown as hex</div>
        </div>
      `;
    }
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
})();

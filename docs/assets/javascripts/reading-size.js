/* Reader-controlled text size, scoped to this book and saved on this device. */
(() => {
  "use strict";

  const siteRoot = new URL("../../", document.currentScript.src);
  const storageKey = `book-text-size-px:${siteRoot.pathname}`;
  const sizes = [15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19];
  const defaultSize = 17;
  let size = defaultSize;

  function readSize() {
    try {
      const saved = Number(localStorage.getItem(storageKey));
      return sizes.includes(saved) ? saved : defaultSize;
    } catch {
      return size;
    }
  }

  function updateFont() {
    const root = document.documentElement;
    const style = getComputedStyle(root);
    const base = parseFloat(style.fontSize) * parseFloat(style.getPropertyValue("--book-body-base"));
    // The selected body size is in pixels. Scale related content proportionally
    // using the theme's current base, including after a responsive breakpoint.
    root.style.setProperty("--book-text-scale", size / base);
  }

  // Apply before MathJax starts so prose and formulas use the same preference.
  size = readSize();
  updateFont();

  const header = document.querySelector(".md-header__inner");
  if (!header) return;
  const controls = document.createElement("div");
  controls.className = "book-reading-size";
  controls.innerHTML = `
    <button type="button" class="book-reading-size__toggle" aria-label="调整字号"
      title="调整字号" aria-expanded="false" aria-haspopup="dialog" aria-controls="book-reading-size-panel">
      <span aria-hidden="true">Aa</span>
    </button>
    <div id="book-reading-size-panel" class="book-reading-size__panel" hidden
      role="dialog" aria-labelledby="book-reading-size-title">
      <div class="book-reading-size__heading">
        <span id="book-reading-size-title">字号</span>
        <button type="button" class="book-reading-size__reset">恢复默认</button>
      </div>
      <div class="book-reading-size__range">
        <input class="book-reading-size__slider" type="range" min="0" max="${sizes.length - 1}"
          step="1" value="${sizes.indexOf(defaultSize)}" aria-label="字号档位">
        <div class="book-reading-size__ticks" aria-hidden="true">
          ${sizes.map(value => `<span${value === defaultSize ? ' class="is-default"' : ''}></span>`).join("")}
        </div>
      </div>
      <div class="book-reading-size__labels" aria-hidden="true">
        <span class="book-reading-size__small">A</span><span>标准</span><span class="book-reading-size__large">A</span>
      </div>
    </div>
    <span class="book-reading-size__status" role="status" aria-live="polite" aria-atomic="true"></span>`;
  header.querySelector(".md-header__title").after(controls);

  const toggle = controls.querySelector(".book-reading-size__toggle");
  const panel = controls.querySelector(".book-reading-size__panel");
  const slider = controls.querySelector(".book-reading-size__slider");
  const reset = controls.querySelector(".book-reading-size__reset");
  const status = controls.querySelector(".book-reading-size__status");
  let alignmentFrame;

  function close() {
    controls.classList.remove("is-open");
    toggle.setAttribute("aria-expanded", "false");
    panel.hidden = true;
  }

  function updateControls(announce = false) {
    const index = sizes.indexOf(size);
    const offset = index - sizes.indexOf(defaultSize);
    const description = offset === 0 ? "标准字号" : `${offset < 0 ? "较小" : "较大"} ${Math.abs(offset)} 档`;
    slider.value = String(index);
    slider.setAttribute("aria-valuetext", description);
    slider.style.setProperty("--book-slider-fill", `${index / (sizes.length - 1) * 100}%`);
    toggle.classList.toggle("has-preference", size !== defaultSize);
    if (announce) status.textContent = `已调整为${description}`;
  }

  function readingPosition() {
    if (window.scrollY < 1) return null;
    const article = document.querySelector("article.md-typeset");
    if (!article) return null;
    const rect = article.getBoundingClientRect();
    const y = Math.max(document.querySelector(".md-header").getBoundingClientRect().bottom, rect.top) + 24;
    const x = Math.max(0, rect.left) + Math.min(24, rect.width / 4);
    if (y >= window.innerHeight || rect.bottom <= y) return null;
    let range;
    if (document.caretPositionFromPoint) {
      const caret = document.caretPositionFromPoint(x, y);
      if (caret) {
        range = document.createRange();
        range.setStart(caret.offsetNode, caret.offset);
        range.collapse(true);
      }
    } else if (document.caretRangeFromPoint) {
      range = document.caretRangeFromPoint(x, y);
    }
    if (range && article.contains(range.startContainer)) {
      const node = range.startContainer;
      if (node.nodeType === Node.TEXT_NODE && range.startOffset < node.length) {
        range.setEnd(node, range.startOffset + 1);
      }
      const bounds = range.getBoundingClientRect();
      if (bounds.height) return { target: range, top: bounds.top };
    }
    const block = [...article.querySelectorAll("h1,h2,h3,h4,p,pre,table,figure")]
      .find(element => element.getBoundingClientRect().bottom > y);
    return block ? { target: block, top: block.getBoundingClientRect().top } : null;
  }

  function fitInlineMath() {
    for (const wrapper of document.querySelectorAll("article span.arithmatex")) {
      const math = wrapper.querySelector("mjx-container");
      if (!math) continue;
      let block = wrapper.parentElement;
      while (block && ["inline", "contents"].includes(getComputedStyle(block).display)) {
        block = block.parentElement;
      }
      if (!block) continue;
      const style = getComputedStyle(block);
      const available = block.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
      const wide = available > 0 && math.getBoundingClientRect().width > available + 1;
      const wasWide = wrapper.classList.contains("book-inline-math-scroll");
      wrapper.classList.toggle("book-inline-math-scroll", wide);
      if (wide) {
        wrapper.setAttribute("tabindex", "0");
        wrapper.setAttribute("role", "region");
        wrapper.setAttribute("aria-label", "可横向滚动的公式");
      } else if (wasWide) {
        for (const attribute of ["tabindex", "role", "aria-label"]) wrapper.removeAttribute(attribute);
      }
    }
  }

  function apply(next, save = false) {
    if (!sizes.includes(next)) return;
    cancelAnimationFrame(alignmentFrame);
    const position = size === next ? null : readingPosition();
    // Native scroll anchoring and our text-position restoration must not both
    // compensate for the same reflow. Keep it disabled through the next paint.
    document.documentElement.style.overflowAnchor = "none";
    size = next;
    updateFont();
    updateControls(save);
    fitInlineMath();
    if (position) {
      const align = () => window.scrollBy({
        top: position.target.getBoundingClientRect().top - position.top, behavior: "instant",
      });
      align();
      alignmentFrame = requestAnimationFrame(() => {
        align();
        alignmentFrame = requestAnimationFrame(() => {
          align();
          document.documentElement.style.removeProperty("overflow-anchor");
        });
      });
    } else document.documentElement.style.removeProperty("overflow-anchor");
    if (save) {
      try { localStorage.setItem(storageKey, String(size)); } catch {
        // The controls still work when the browser blocks persistent storage.
      }
    }
  }

  slider.addEventListener("input", () => apply(sizes[Number(slider.value)], true));
  reset.addEventListener("click", () => apply(defaultSize, true));
  toggle.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    controls.classList.toggle("is-open", open);
    toggle.setAttribute("aria-expanded", String(open));
    panel.hidden = !open;
    if (open) slider.focus({ preventScroll: true });
  });
  document.addEventListener("pointerdown", event => {
    if (!controls.contains(event.target)) close();
  });
  controls.addEventListener("keydown", event => {
    if (event.key === "Escape" && !panel.hidden) {
      close();
      toggle.focus({ preventScroll: true });
    }
  });
  controls.addEventListener("focusout", event => {
    if (!controls.contains(event.relatedTarget)) close();
  });
  window.addEventListener("storage", event => {
    if (event.key === storageKey || event.key === null) apply(readSize());
  });
  window.addEventListener("pageshow", () => apply(readSize()));
  let resizeFrame;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
      updateFont();
      fitInlineMath();
    });
  });
  const afterLoad = () => Promise.allSettled([
    window.MathJax?.startup?.promise, document.fonts?.ready,
  ]).then(fitInlineMath);
  if (document.readyState === "complete") afterLoad();
  else window.addEventListener("load", afterLoad, { once: true });
  close();
  updateControls();
})();

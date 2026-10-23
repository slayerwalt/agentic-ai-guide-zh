/* Return from a reference jump without adding another browser history entry. */
(() => {
  "use strict";

  const root = new URL("../../", document.currentScript.src);
  const stateKey = "__bookReading";
  const storagePrefix = `book-reading:${root.pathname}:`;
  const memory = {};
  const maxAge = 120000;
  let storageUsable = true;
  let cancelRestore = () => {};

  function read(key) {
    if (!storageUsable) return memory[key] || null;
    try {
      return JSON.parse(sessionStorage.getItem(storagePrefix + key));
    } catch {
      storageUsable = false;
      return memory[key] || null;
    }
  }

  function write(key, value) {
    memory[key] = value;
    if (!storageUsable) return;
    try {
      if (value === null) sessionStorage.removeItem(storagePrefix + key);
      else sessionStorage.setItem(storagePrefix + key, JSON.stringify(value));
    } catch {
      storageUsable = false;
      // Same-page returns still work when browser storage is unavailable.
    }
  }

  function internal(url) {
    try {
      const target = new URL(url);
      return target.origin === root.origin && target.pathname.startsWith(root.pathname);
    } catch {
      return false;
    }
  }

  function fresh(record) {
    return record && Date.now() - record.time < maxAge;
  }

  function current() {
    const record = history.state?.[stateKey];
    return record?.url === location.href ? record : null;
  }

  function save(record) {
    history.replaceState({ ...history.state, [stateKey]: record }, "");
    return record;
  }

  function createEntry(back = null, push = false) {
    const record = {
      id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`,
      url: location.href,
      back,
    };
    if (push) history.pushState({ ...history.state, [stateKey]: record }, "");
    else save(record);
    return record;
  }

  const button = document.createElement("button");
  button.type = "button";
  button.className = "book-reading-back";
  button.hidden = true;
  button.title = "返回跳转前的阅读位置";
  button.innerHTML =
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
    '<path d="m9 14-5-5 5-5M4 9h10a6 6 0 0 1 0 12h-2"/></svg>' +
    "<span>返回引用位置</span>";
  document.body.append(button);

  function contentLinks() {
    return [...document.querySelectorAll(".md-content__inner a[href]")];
  }

  function position(link) {
    return {
      x: window.scrollX,
      y: window.scrollY,
      link: contentLinks().indexOf(link),
      href: link.href,
      top: link.getBoundingClientRect().top,
    };
  }

  function restore(saved) {
    cancelRestore();
    let cancelled = false;
    const interactions = ["wheel", "touchstart", "pointerdown", "keydown"];
    const stop = () => {
      cancelled = true;
      for (const type of interactions) window.removeEventListener(type, stop, true);
    };
    cancelRestore = stop;
    for (const type of interactions) window.addEventListener(type, stop, { capture: true, passive: true });

    const link = contentLinks()[saved.link];
    const source = link?.href === saved.href ? link : null;
    const align = () => {
      if (cancelled) return;
      const top = source
        ? window.scrollY + source.getBoundingClientRect().top - saved.top
        : saved.y;
      window.scrollTo({ left: saved.x, top, behavior: "instant" });
    };
    requestAnimationFrame(align);
    const loaded = document.readyState === "complete"
      ? Promise.resolve()
      : new Promise(resolve => window.addEventListener("load", resolve, { once: true }));
    Promise.allSettled([
      loaded,
      window.MathJax?.startup?.promise,
      document.fonts?.ready,
    ]).then(() => requestAnimationFrame(() => requestAnimationFrame(() => {
      if (!cancelled) {
        source?.focus({ preventScroll: true });
        align();
      }
      stop();
    })));
  }

  function fromPreviousDocument(pending) {
    if (!pending?.departed || history.length < 2 || !document.referrer) return false;
    const source = new URL(pending.from.url);
    source.hash = "";
    return source.href === document.referrer;
  }

  function synchronize(acceptPending, crossDocument = false) {
    const pending = read("pending");
    let record = current();
    if (acceptPending && fresh(pending) && pending.to === location.href &&
        internal(pending.from.url) && (!crossDocument || fromPreviousDocument(pending))) {
      // Native hash navigation can inherit the source state; give its new
      // history entry a distinct identity before handling back/forward.
      record = createEntry(pending.from);
    } else if (!record) {
      record = createEntry();
    }
    write("pending", null);

    button.disabled = false;
    button.hidden = !(record.back && internal(record.back.url));
    const returning = read("returning");
    if (fresh(returning) && returning.to === record.id) {
      write("returning", null);
      restore(returning.position);
    } else if (returning) {
      write("returning", null);
    }
  }

  document.addEventListener("click", event => {
    if (event.defaultPrevented || event.button !== 0 ||
        event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest?.(".md-content__inner a[href]");
    if (!link || link.hasAttribute("download") ||
        (link.target && link.target !== "_self") ||
        link.matches(".headerlink, .footnote-backref, .note-backlink, a[href^='#note-ref-']")) return;

    const target = new URL(link.href);
    if (!internal(target.href) || !target.hash ||
        !(/\/$|\.html$/.test(target.pathname))) return;

    cancelRestore();
    const source = current() || createEntry();
    const snapshot = { id: source.id, url: location.href, position: position(link) };
    if (target.href === location.href) {
      let anchor;
      try {
        anchor = document.getElementById(decodeURIComponent(target.hash.slice(1)));
      } catch {
        return;
      }
      if (!anchor) return;
      // Revisit the current fragment after scrolling away from it.
      event.preventDefault();
      createEntry(snapshot, true);
      synchronize(false);
      anchor.scrollIntoView({ block: "start", behavior: "instant" });
      return;
    }
    write("pending", { from: snapshot, to: target.href, time: Date.now() });
    // The link's normal navigation creates the next browser history entry.
  });

  button.addEventListener("click", () => {
    const back = current()?.back;
    if (!back || !internal(back.url)) return;
    write("returning", { to: back.id, position: back.position, time: Date.now() });
    button.disabled = true;
    history.back();
  });

  window.addEventListener("popstate", () => synchronize(true));
  window.addEventListener("hashchange", () => synchronize(true));
  window.addEventListener("pageshow", () => synchronize(false));
  window.addEventListener("pagehide", () => {
    const pending = read("pending");
    if (fresh(pending) && pending.from.id === current()?.id) {
      write("pending", { ...pending, departed: true });
    }
  });
  const navigation = performance.getEntriesByType("navigation")[0]?.type;
  synchronize(navigation !== "reload" && navigation !== "back_forward", true);
})();

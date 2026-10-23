/* MathJax 3.2.2 is staged locally by scripts/vendor_mathjax.py. */
window.MathJax = {
  // Text-mode escapes such as \text{logp\_chosen} need textmacros.
  loader: { load: ["[tex]/mathtools", "[tex]/boldsymbol", "[tex]/textmacros"] },
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    packages: { "[+]": ["mathtools", "boldsymbol", "textmacros"] },
    tags: "ams",
    macros: {
      argmax: "\\mathop{\\mathrm{arg\\,max}}\\limits",
      argmin: "\\mathop{\\mathrm{arg\\,min}}\\limits",
      softmax: "\\operatorname{softmax}",
      KL: "\\operatorname{KL}",
      Tr: "\\operatorname{Tr}",
      diag: "\\operatorname{diag}",
      rank: "\\operatorname{rank}",
      bm: ["\\boldsymbol{#1}", 1]
    }
  },
  options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" },
  chtml: { scale: 1, matchFontHeight: false }
};

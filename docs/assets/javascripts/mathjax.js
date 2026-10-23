/* MathJax 3.2.2 is staged locally by scripts/vendor_mathjax.py. */
window.MathJax = {
  loader: { load: ["[tex]/mathtools", "[tex]/boldsymbol"] },
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    packages: { "[+]": ["mathtools", "boldsymbol"] },
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

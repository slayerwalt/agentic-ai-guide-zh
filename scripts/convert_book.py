"""Convert the complete, existing Chinese TeX text into section reading pages.

Pandoc reads prose, lists and table cells. A small source pass preserves the
constructs its LaTeX reader cannot faithfully retain: box titles, tabularx,
explicit equation numbers, listings, and TikZ figures. This script never runs
the book's example programs. Generated Markdown is committed; normal site
builds need neither Pandoc nor BibTeX.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

from book_structure import ROOT, active_tex, braced, catalog

ESC = html.escape
WORK = ROOT / ".site-work/conversion"
BOXES = {"keybox": ("note", "要点"), "intuitionbox": ("tip", "直觉"),
         "examplebox": ("example", "示例"), "warningbox": ("warning", "注意"),
         "questionbox": ("question", "思考"), "tcolorbox": ("note", "要点")}
SPECIAL_ENVS = set(BOXES) | {"table", "tabular", "tabularx", "figure", "lstlisting",
                            "equation", "equation*", "align", "align*", "multline", "multline*"}
HEADING = re.compile(r"^\\(chapter|section|subsection|subsubsection|paragraph)(\*?)\s*(?=\{)", re.M)
TOKEN = re.compile(r"\\begin\{([^}]+)\}|(?<!\\)\\\[|\$\$")
LAYOUT = re.compile(r"\\(?:centering|noindent|small|footnotesize|scriptsize|normalsize|newpage|clearpage|smallskip|bigskip|medskip|nopagebreak)(?:\[[^\]]*\])?(?:\{\})?\s*$")


def argument(text, pos, left="{", right="}"):
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] != left:
        raise ValueError(f"Expected {left} near {text[pos:pos+100]!r}")
    depth = 0
    braces = 0
    for i in range(pos, len(text)):
        if text[i] == "\\":
            continue
        escaped = (len(text[:i]) - len(text[:i].rstrip("\\"))) % 2
        if escaped:
            continue
        if left == "[":
            braces += (text[i] == "{") - (text[i] == "}")
        if not braces:
            depth += (text[i] == left) - (text[i] == right)
        if depth == 0:
            return text[pos+1:i], i+1
    raise ValueError(f"Unbalanced argument {text[pos:pos+100]!r}")


def visible_titles(text):
    while match := re.search(r"\\texorpdfstring(?=\{)", text):
        visible, end = braced(text, match.end())
        _, finish = argument(text, end)
        text = text[:match.start()] + visible + text[finish:]
    return text


def environment_end(text, start, name):
    if name == "lstlisting":
        end = text.index(r"\end{lstlisting}", start)
        return end, end + len(r"\end{lstlisting}")
    depth = 1
    for match in re.finditer(r"\\(begin|end)\{" + re.escape(name) + r"\}", text[start:]):
        depth += 1 if match[1] == "begin" else -1
        if depth == 0:
            return start + match.start(), start + match.end()
    raise ValueError(f"Unclosed environment {name}")


def command_argument(text, command):
    match = re.search(r"\\" + command + r"\s*(?=\{)", text)
    if not match:
        return None, text
    value, end = braced(text, match.end())
    return value, text[:match.start()] + text[end:]


def options(text):
    pairs = {}
    for match in re.finditer(r"(?:^|,)\s*([\w-]+)\s*=\s*", text):
        start = match.end()
        if text[start:start+1] == "{":
            value, _ = braced(text, start)
        else:
            value = text[start:].split(",", 1)[0].strip()
        pairs[match[1]] = value
    return pairs


def split_math_rows(text):
    """Split align rows, ignoring row breaks inside cases/matrices/groups."""
    result, start, braces, envs = [], 0, 0, []
    for m in re.finditer(r"\\(?:begin|end)\{[^}]+\}|\\\\(?:\[[^\]]*\])?|(?<!\\)[{}]", text):
        token = m[0]
        if token.startswith(r"\begin{"):
            envs.append(token)
        elif token.startswith(r"\end{"):
            envs.pop()
        elif token == "{":
            braces += 1
        elif token == "}":
            braces -= 1
        elif not braces and not envs:
            result.append((text[start:m.start()], token))
            start = m.end()
    if text[start:].strip():
        result.append((text[start:], ""))
    return result


@dataclass
class Page:
    path: str
    chapter: int
    title: str
    level: int
    source: str
    number: str
    anchor: str
    children: list = field(default_factory=list)
    fragment: int | None = None


class Book:
    def __init__(self, pandoc):
        self.pandoc = pandoc
        self.structure = catalog()
        self.pages = []
        self.headers = {}
        self.targets = {}
        self.fragments = []
        self.asts = {}
        self.specials = []
        self.counts = defaultdict(Counter)
        self.counters = defaultdict(Counter)
        self.backlinks = defaultdict(list)
        self.images = set()
        self.source_notes = []
        self.note_counters = Counter()
        self.numbers = {}
        self.references = []

    def target(self, label, page, number):
        if label in self.targets and self.targets[label] != (page.path, number):
            raise ValueError(f"Duplicate source label: {label}")
        self.targets[label] = (page.path, number)

    def model(self):
        chapters = [c for p in self.structure["parts"] for c in p["chapters"]]
        n = 0
        for part in self.structure["parts"]:
            text = active_tex(ROOT / part["source"])
            starts = list(re.finditer(r"^\\chapter\{", text, re.M))
            for ci, start in enumerate(starts):
                chapter = chapters[n]
                n += 1
                body = text[start.start():starts[ci+1].start() if ci+1 < len(starts) else len(text)]
                headers = []
                section = subsection = 0
                inherited = str(n)
                for hi, match in enumerate(HEADING.finditer(body)):
                    raw_title, end = braced(body, match.end())
                    labels = []
                    while label := re.match(r"\s*\\label\{([^}]+)\}", body[end:]):
                        labels.append(label[1])
                        end += label.end()
                    kind = match[1]
                    level = ["chapter", "section", "subsection", "subsubsection", "paragraph"].index(kind) + 1
                    numbered = not match[2] and level <= 3
                    if kind == "section" and numbered:
                        section += 1
                        subsection = 0
                        inherited = f"{n}.{section}"
                    elif kind == "subsection" and numbered:
                        subsection += 1
                        inherited = f"{n}.{section}.{subsection}"
                    anchor = labels[0] if labels else f"heading-{n}-{hi+1}"
                    headers.append(dict(start=match.start(), end=end, title=visible_titles(raw_title),
                                        anchor=anchor, labels=labels or [anchor], level=level,
                                        number=inherited if numbered else "", ref=inherited,
                                        section=section, subsection=subsection))
                cuts = [h for h in headers if h["level"] <= 2 or (n == 2 and h["level"] == 3)]
                chapter_pages = []
                for i, cut in enumerate(cuts):
                    if cut["level"] == 1:
                        path = chapter["path"]
                    elif cut["level"] == 2:
                        path = str(Path(chapter["path"]).with_name(f"section-{cut['section']:02d}.md")).replace("\\", "/")
                    else:
                        path = str(Path(chapter["path"]).with_name(f"section-{cut['section']:02d}-{cut['subsection']:02d}.md")).replace("\\", "/")
                    end = cuts[i+1]["start"] if i+1 < len(cuts) else len(body)
                    contained = [h for h in headers if cut["start"] <= h["start"] < end]
                    chunk = body[cut["start"]:end]
                    # Give every source heading an explicit anchor, including
                    # unnumbered paragraphs, so links survive route splitting.
                    for h in reversed(contained):
                        htext = "\\" + ["chapter", "section", "subsection", "subsubsection", "paragraph"][h["level"]-1]
                        replacement = htext + "{" + h["title"] + "}\\label{" + h["anchor"] + "}"
                        a, b = h["start"]-cut["start"], h["end"]-cut["start"]
                        chunk = chunk[:a] + replacement + chunk[b:]
                    page = Page(path, n, cut["title"], cut["level"], chunk, cut["number"], cut["anchor"])
                    self.pages.append(page)
                    chapter_pages.append(page)
                    for h in contained:
                        self.headers[h["anchor"]] = h
                        for label in h["labels"]:
                            self.target(label, page, h["ref"])
                chapter_pages[0].children = [p for p in chapter_pages[1:] if p.level == 2]
                parent = None
                for page in chapter_pages[1:]:
                    if page.level == 2:
                        parent = page
                    elif page.level == 3:
                        parent.children.append(page)

    def bibliography(self):
        # A separate optional tool directory keeps conversion dependencies
        # out of the pinned website runtime.
        sys.path.insert(0, str(ROOT / ".site-work/python-tools"))
        from pybtex.bibtex import BibTeXEngine
        from pybtex.errors import capture
        source = active_tex(ROOT / "book.tex")
        source = re.sub(r"\\input\{([^}]+)\}", lambda m: active_tex(ROOT / (m[1]+".tex")), source)
        keys = list(dict.fromkeys(k.strip() for group in re.findall(r"\\cite\{([^}]+)\}",source) for k in group.split(",")))
        with capture() as warnings:
            bbl = BibTeXEngine().format_from_file(str(ROOT / "book.bib"),
                style=str(ROOT / "scripts/bibliography/plainnat"), citations=keys,
                bib_encoding="utf-8", bst_encoding="utf-8", output_encoding="utf-8")
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "book.bbl").write_text(bbl,encoding="utf-8")
        (WORK / "bibliography-warnings.txt").write_text("\n".join(str(w) for w in warnings),encoding="utf-8")
        matches = list(re.finditer(r"\\bibitem(?:\[.*?\])?\{([^}]+)\}",bbl,re.S))
        for i, m in enumerate(matches):
            end = matches[i+1].start() if i+1<len(matches) else bbl.index(r"\end{thebibliography}")
            self.numbers[m[1]] = i+1
            content = bbl[m.end():end].replace(r"\newblock", " ")
            content = re.sub(r"\\natexlab\{([^}]+)\}",r"\1",content)
            self.references.append(dict(key=m[1],number=i+1,fragment=self.fragment(content)))
        assert set(self.numbers) == set(keys)

    def write_references(self):
        for path in [*sorted((ROOT / "docs/front-matter").glob("*.md")),ROOT / "docs/reading-sample.md"]:
            source = path.read_text(encoding="utf-8")
            for m in re.finditer(r'<a class="book-citation"([^>]+)>',source):
                key = re.search(r'#ref-([^"\s]+)',m[1])[1]
                anchor = re.search(r'id="([^"]+)"',m[1])
                anchor = anchor[1] if anchor else "bpe-citation"
                title = re.search(r'^# ([^{\n]+)',source,re.M)[1].strip()
                self.backlinks[key].insert(0,(path.relative_to(ROOT / "docs").as_posix(),anchor,title))
        page = Page("references.md",0,"参考文献",1,"","", "references")
        renderer = Renderer(self,page)
        output = ["# 参考文献 {#references}",
                  f"本页收录全书引用的 {len(self.references)} 篇文献，沿用原书编号。"]
        for entry in self.references:
            key = entry["key"]
            text = renderer.fragment_inline(entry["fragment"])
            output.append(f'<div class="reference-entry" id="ref-{key}"><p><span class="reference-number">[{entry["number"]}]</span> {text}</p></div>')
        (ROOT / "docs/references.md").write_text("\n\n".join(output)+"\n",encoding="utf-8")

    def write_navigation(self):
        q = lambda s: json.dumps(s,ensure_ascii=False)
        lines = ['nav = [', '  { "首页" = "index.md" },', '  { "全书目录" = "book-outline.md" },',
                 '  { "阅读样例" = "reading-sample.md" },', '  { "卷首" = [']
        for slug,title in [('dedication','献辞'),('disclaimer','免责声明'),('author','关于作者'),('preface','前言'),('introduction','引言'),('glossary','缩略语表')]:
            lines.append(f'    {{ {q(title)} = "front-matter/{slug}.md" }},')
        lines.append('  ] },')
        roman = ['I','II','III','IV','V','VI']
        def entry(page,indent):
            title = (f"第 {page.chapter} 章 · " if page.level == 1 else page.number+" · ") + re.sub(r'\$(.*?)\$',r'\1',page.title).replace(r'\beta','β')
            if not page.children:
                return [' '*indent+f'{{ {q(title)} = {q(page.path)} }},']
            out = [' '*indent+f'{{ {q(title)} = [', ' '*(indent+2)+f'{{ "概览" = {q(page.path)} }},']
            for child in page.children: out += entry(child,indent+2)
            return out+[' '*indent+'] },']
        for part in self.structure['parts']:
            lines.append(f'  {{ {q("第 "+roman[part["number"]-1]+" 部分 · "+part["title"])} = [')
            lines.append(f'    {{ "本部分目录" = {q(part["path"])} }},')
            for chapter in part['chapters']:
                page = next(p for p in self.pages if p.path == chapter['path'])
                lines += entry(page,4)
            lines.append('  ] },')
        lines += ['  { "参考文献" = "references.md" },','  { "来源与署名" = "about.md" },',']']
        config = ROOT / 'zensical.toml'
        text = config.read_text(encoding='utf-8')
        start = text.index('nav = [')
        end = text.index('\n[project.theme]',start)
        config.write_text(text[:start]+'\n'.join(lines)+'\n'+text[end:],encoding='utf-8')

    def fragment(self, text):
        index = len(self.fragments)
        self.fragments.append(visible_titles(text))
        return index

    def special(self, data):
        index = len(self.specials)
        self.specials.append(data)
        return f"\n\n\\href{{webblock:{index}}}{{WEBBLOCK}}\n\n"

    def preprocess(self, text, page):
        out, pos = [], 0
        while match := TOKEN.search(text, pos):
            out.append(text[pos:match.start()])
            if match[1] and match[1] not in SPECIAL_ENVS:
                out.append(match[0])
                pos = match.end()
                continue
            env = match[1]
            if env:
                start = match.end()
                opt = ""
                if text[start:start+1] == "[":
                    opt, start = argument(text, start, "[", "]")
                end, finish = environment_end(text, start, env)
                body = text[start:end]
            else:
                env = "display"
                closing = r"\]" if match[0] == r"\[" else "$$"
                end = text.index(closing, match.end())
                finish = end + len(closing)
                body, opt = text[match.end():end], ""
            self.counts[page.chapter][env] += 1
            data = dict(kind=env, page=page.path)
            if env in BOXES:
                data.update(title=self.fragment(opt), body=self.fragment(self.preprocess(body, page)))
                if page.chapter == 28 and env == "questionbox":
                    number = re.match(r"(Q[\w]+)", opt)
                    data["question"] = number[1] if number else f"Qsource-{self.counts[page.chapter][env]}"
            elif env == "lstlisting":
                attrs = options(opt)
                code = body.removeprefix("\n").removesuffix("\n")
                title = attrs.get("caption", "")
                label = attrs.get("label", "")
                number = ""
                if title:
                    self.counters[page.chapter]["listing"] += 1
                    number = f"{page.chapter}.{self.counters[page.chapter]['listing']}"
                data.update(code=code, title=self.fragment(title), label=label, number=number, language=attrs.get("language", "python"))
                if "TypeScript" in title or "React 前端" in title:
                    data["language"] = "tsx"
                if "Docker 与 Kubernetes" in title:
                    data["language"] = "text"
                if label:
                    self.target(label, page, number)
            elif env in {"table", "tabular", "tabularx"}:
                raw = body if env == "table" else "\\begin{" + env + "}" + body + "\\end{" + env + "}"
                caption, raw = command_argument(raw, "caption")
                labels = re.findall(r"\\label\{([^}]+)\}", raw)
                raw = re.sub(r"\\label\{[^}]+\}", "", raw)
                number = ""
                if caption is not None:
                    self.counters[page.chapter]["table"] += 1
                    number = f"{page.chapter}.{self.counters[page.chapter]['table']}"
                ordinal = sum(self.counts[page.chapter][key] for key in ("table","tabular","tabularx"))
                anchor = labels[0] if labels else ("table-"+number.replace(".","-") if number else f"table-{page.chapter}-inline-{ordinal}")
                for label in labels:
                    self.target(label, page, number)
                data.update(title=self.fragment(caption or ""), number=number, anchor=anchor, labels=labels)
                raw = self.normalize_tables(raw)
                data["body"] = self.fragment(raw)
            elif env == "figure":
                caption, raw = command_argument(body, "caption")
                labels = re.findall(r"\\label\{([^}]+)\}", raw)
                self.counters[page.chapter]["figure"] += 1
                number = f"{page.chapter}.{self.counters[page.chapter]['figure']}"
                anchor = labels[0] if labels else "figure-" + number.replace(".", "-")
                for label in labels:
                    self.target(label, page, number)
                images = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", raw)
                if "\\begin{tikzpicture}" in raw:
                    images = [f"diagrams/figure-{number.replace('.', '-')}.svg"]
                    data["tikz"] = raw
                assert len(images) == 1, (page.path, images)
                self.images.add(images[0])
                data.update(title=self.fragment(caption or ""), number=number, anchor=anchor,
                            labels=labels, image=images[0])
            else:
                labels = re.findall(r"\\label\{([^}]+)\}", body)
                numbered = env in {"equation", "align", "multline"}
                rows = split_math_rows(body) if env.startswith("align") else [(body, "")]
                rendered, numbers = [], []
                for row, sep in rows:
                    tag = re.search(r"\\tag\{([^}]+)\}", row)
                    number = tag[1] if tag else ""
                    if numbered and not tag and not re.search(r"\\(?:nonumber|notag)\b", row):
                        self.counters[page.chapter]["equation"] += 1
                        number = f"{page.chapter}.{self.counters[page.chapter]['equation']}"
                        row += r"\tag{" + number + "}"
                    if number:
                        numbers.append(number)
                    for label in re.findall(r"\\label\{([^}]+)\}", row):
                        self.target(label, page, number)
                    row = re.sub(r"\\label\{[^}]+\}", "", row)
                    rendered.append(row + sep)
                math = "\n".join(rendered).strip()
                if env.startswith("align") or env.startswith("multline"):
                    name = env.rstrip("*") + "*"
                    math = "\\begin{"+name+"}\n" + math + "\n\\end{"+name+"}"
                data.update(math=math, labels=labels, numbers=numbers)
            out.append(self.special(data))
            pos = finish
        out.append(text[pos:])
        return "".join(out)

    def normalize_tables(self, text):
        for match in reversed(list(re.finditer(r"\\begin\{(tabularx|tabular)\}", text))):
            spec, end = argument(text, match.end())
            if match[1] == "tabularx":
                spec, end = argument(text, end)
            spec = re.sub(r"Z\{[^}]+\}|p\{[^}]+\}|X", "l", spec)
            if re.search(r"[^lcr@{}\s|]", spec):
                raise ValueError(f"Unsupported table columns: {spec}")
            text = text[:match.start()] + r"\begin{tabular}{" + spec + "}" + text[end:]
        text = re.sub(r"(\\multicolumn\{\d+\})\{@\{\}([lcr])\}", r"\1{\2}", text)
        return text.replace(r"\end{tabularx}", r"\end{tabular}")

    def read_fragments(self):
        WORK.mkdir(parents=True, exist_ok=True)
        text = "\n\n".join(r"\chapter{WEBFRAGMENT" + str(i) + "}\n" + t for i, t in enumerate(self.fragments))
        source = WORK / "fragments.tex"
        source.write_text(text, encoding="utf-8")
        result = subprocess.run([self.pandoc, "-f", "latex+raw_tex", "-t", "json", str(source)],
                                capture_output=True, text=True, encoding="utf-8", check=True)
        if result.stderr.strip():
            (WORK / "pandoc-warnings.txt").write_text(result.stderr, encoding="utf-8")
            print(result.stderr)
        ast = json.loads(result.stdout)
        current = None
        for block in ast["blocks"]:
            if block["t"] == "Header" and len(block["c"][2]) == 1:
                title = block["c"][2][0]
                if title["t"] == "Str" and re.fullmatch(r"WEBFRAGMENT\d+", title["c"]):
                    current = int(title["c"][11:])
                    self.asts[current] = []
                    continue
            if current is None:
                raise ValueError(f"Unexpected fragment preamble: {block}")
            self.asts[current].append(block)
        assert len(self.asts) == len(self.fragments)
        (WORK / "fragments.json").write_text(result.stdout, encoding="utf-8")


class Renderer:
    def __init__(self, book, page):
        self.book, self.page = book, page
        self.cites = Counter()
        self.notes = []
        self.emitted = set()

    def url(self, path, anchor=""):
        if path == self.page.path and anchor:
            return "#" + anchor
        relative = os.path.relpath(path, Path(self.page.path).parent).replace("\\", "/")
        return relative + ("#" + anchor if anchor else "")

    def anchors(self, labels):
        result = []
        for label in labels:
            if label and label not in self.emitted:
                result.append(f'<span id="{ESC(label, quote=True)}"></span>')
                self.emitted.add(label)
        return "".join(result)

    def inline(self, nodes):
        out = []
        for node in nodes:
            t, c = node["t"], node.get("c")
            if t == "Str": out.append(ESC(c))
            elif t in ("Space", "SoftBreak"): out.append(" ")
            elif t == "LineBreak": out.append("<br>")
            elif t in {"Strong", "Emph", "Superscript", "Subscript", "Strikeout", "SmallCaps"}:
                tag = {"Strong":"strong", "Emph":"em", "Superscript":"sup", "Subscript":"sub", "Strikeout":"del", "SmallCaps":"span"}[t]
                out.append(f"<{tag}>{self.inline(c)}</{tag}>")
            elif t == "Code": out.append("<code>" + ESC(c[1]) + "</code>")
            elif t == "Math":
                assert c[0]["t"] == "InlineMath", c
                out.append('<span class="arithmatex">\\(' + ESC(c[1]) + '\\)</span>')
            elif t == "Quoted":
                a, b = ("“", "”") if c[0]["t"] == "DoubleQuote" else ("‘", "’")
                out.append(a+self.inline(c[1])+b)
            elif t == "Span": out.append(self.anchors([c[0][0]]) + self.inline(c[1]))
            elif t == "Link":
                attrs, children, dest = c
                if dest[0].startswith("webblock:"):
                    raise ValueError("Block placeholder embedded in prose")
                ref = dict(attrs[2]).get("reference")
                if ref:
                    path, number = self.book.targets[ref]
                    out.append(f'<a href="{ESC(self.url(path,ref),quote=True)}">{ESC(number)}</a>')
                else:
                    out.append(f'<a href="{ESC(dest[0],quote=True)}">{self.inline(children)}</a>')
            elif t == "Cite":
                links = []
                for cite in c[0]:
                    key = cite["citationId"]
                    assert not cite["citationPrefix"] and not cite["citationSuffix"]
                    number = self.book.numbers[key]
                    self.cites[key] += 1
                    anchor = f"cite-{key}-{self.cites[key]}"
                    self.book.backlinks[key].append((self.page.path, anchor, self.page.number))
                    links.append(f'<a class="book-citation" id="{anchor}" href="{self.url("references.md", "ref-"+key)}" aria-label="参考文献 {number}">[{number}]</a>')
                out.append("".join(links))
            elif t == "Note":
                self.book.note_counters[self.page.chapter] += 1
                n = self.book.note_counters[self.page.chapter]
                self.notes.append((n,c))
                out.append(f'<sup id="note-ref-{n}"><a href="#note-{n}" aria-label="注 {n}">{n}</a></sup>')
            elif t == "RawInline": out.append(self.raw(c[1]))
            else: raise ValueError(f"Unhandled inline {t}: {c}")
        return "".join(out)

    def raw(self, text):
        text = text.strip()
        if LAYOUT.fullmatch(text.strip()): return ""
        if re.fullmatch(r"\\(?:par|bfseries)(?:\{\})?", text): return ""
        if text in {r"\checkmark", r"\checkmark{}"}: return "✓"
        if m := re.fullmatch(r"\\label\{([^}]+)\}", text): return self.anchors([m[1]])
        if m := re.fullmatch(r"\\ref\{([^}]+)\}", text):
            path, number = self.book.targets[m[1]]
            return f'<a href="{ESC(self.url(path,m[1]),quote=True)}">{ESC(number)}</a>'
        if m := re.fullmatch(r"\\footnotemark\[(\d+)\]", text):
            self.book.source_notes.append(dict(page=self.page.path, missing_note=int(m[1])))
            return f'<sup aria-label="原文注号 {m[1]}">{m[1]}</sup>'
        if m := re.fullmatch(r"\\newblock", text): return " "
        if m := re.fullmatch(r"\\natexlab\{([^}]+)\}", text): return ESC(m[1])
        if m := re.fullmatch(r"\\doi\{([^}]+)\}", text):
            return f'<a href="https://doi.org/{ESC(m[1],quote=True)}">doi: {ESC(m[1])}</a>'
        if re.fullmatch(r"\\vspace\*?\{[^}]+\}",text): return ""
        if re.fullmatch(r"\\penalty\s*-?\d+",text): return ""
        raise ValueError(f"Unhandled LaTeX in {self.page.path}: {text[:500]}")

    def fragment_inline(self, index):
        out = []
        for b in self.book.asts[index]:
            if b["t"] in ("Plain", "Para"): out.append(self.inline(b["c"]))
            elif b["t"] == "RawBlock": out.append(self.raw(b["c"][1]))
            else: raise ValueError(f"Non-inline caption: {b}")
        return " ".join(out)

    def blocks(self, blocks, markdown_headings=False):
        result = []
        for block in blocks:
            t, c = block["t"], block.get("c")
            if t in ("Para", "Plain"):
                meaningful = [n for n in c if n["t"] not in ("Space", "SoftBreak")]
                if len(meaningful) == 1 and meaningful[0]["t"] == "Link" and meaningful[0]["c"][2][0].startswith("webblock:"):
                    result.append(self.special(int(meaningful[0]["c"][2][0].split(":")[1])))
                else:
                    value = self.inline(c)
                    if "<strong>复习：</strong>" in value:
                        chapters = {str(p.chapter):p.path for p in self.book.pages if p.level == 1}
                        def review_link(m):
                            return re.sub(r"\d+",lambda n: f'<a href="{self.url(chapters[n[0]])}">{n[0]}</a>' if n[0] in chapters else n[0],m[0])
                        value = re.sub(r"第[^<]*?章",review_link,value)
                    if value: result.append("<p>"+value+"</p>" if t == "Para" else value)
            elif t == "Header":
                level, attrs, nodes = c
                anchor = attrs[0]
                h = self.book.headers.get(anchor)
                if h is None: raise ValueError(f"Unknown heading {anchor}: {c}")
                level = h["level"] - self.page.level + 1
                title = self.inline(nodes)
                number = h["number"]
                prefix = f"第 {number} 章 · " if h["level"] == 1 else (number+"　" if number else "")
                self.emitted.add(anchor)
                aliases = self.anchors([l for l in h["labels"] if l != anchor])
                if markdown_headings:
                    result.append(aliases + "#" * min(6,level) + " " + prefix + title + f" {{#{anchor}}}")
                else:
                    result.append(aliases + f'<h{min(6,level)} id="{ESC(anchor,quote=True)}">{prefix}{title}</h{min(6,level)}>')
            elif t in ("BulletList", "OrderedList"):
                tag = "ul" if t == "BulletList" else "ol"
                items = c if t == "BulletList" else c[1]
                start = "" if t == "BulletList" or c[0][0] == 1 else f' start="{c[0][0]}"'
                result.append(f"<{tag}{start}>\n" + "\n".join("<li>"+self.blocks(item)+"</li>" for item in items) + f"\n</{tag}>")
            elif t == "Div":
                attrs, children = c
                if set(attrs[1]) - {"center"}: raise ValueError(f"Unhandled div {attrs}")
                result.append(self.anchors([attrs[0]]) + self.blocks(children))
            elif t == "Table": result.append(self.table(c))
            elif t == "RawBlock": result.append(self.raw(c[1]))
            elif t == "BlockQuote": result.append("<blockquote>"+self.blocks(c)+"</blockquote>")
            elif t == "HorizontalRule": result.append("<hr>")
            elif t == "DefinitionList":
                result.append('<dl class="book-definitions">'+"".join("<dt>"+self.inline(term)+"</dt><dd>"+"".join(self.blocks(d) for d in defs)+"</dd>" for term,defs in c)+"</dl>")
            else: raise ValueError(f"Unhandled block {t}: {str(c)[:300]}")
        return "\n\n".join(s for s in result if s)

    def table(self, c):
        attrs, caption, specs, head, bodies, foot = c
        def rows(items, header=False):
            lines = []
            for attr, cells in items:
                rendered = []
                for attr, alignment, rowspan, colspan, content in cells:
                    tag = "th" if header else "td"
                    extra = ' scope="col"' if header else ""
                    if rowspan > 1: extra += f' rowspan="{rowspan}"'
                    if colspan > 1: extra += f' colspan="{colspan}"'
                    rendered.append(f"<{tag}{extra}>"+self.blocks(content)+f"</{tag}>")
                lines.append("<tr>"+"".join(rendered)+"</tr>")
            return "\n".join(lines)
        out = ["<table>"]
        if head[1]: out.append("<thead>"+rows(head[1],True)+"</thead>")
        for attr, rowhead, intermediate, data in bodies:
            out.append("<tbody>"+rows(intermediate,True)+rows(data)+"</tbody>")
        if foot[1]: out.append("<tfoot>"+rows(foot[1])+"</tfoot>")
        return "\n".join(out)+"\n</table>"

    def special(self, index):
        d = self.book.specials[index]
        kind = d["kind"]
        if kind in BOXES:
            cls, default_title = BOXES[kind]
            title = self.fragment_inline(d["title"])
            body = self.blocks(self.book.asts[d["body"]])
            if "question" in d:
                anchor = "q-" + d["question"][1:].lower()
                return f'<section class="quiz-question" id="{anchor}"><h2 class="question-title">{title}</h2>\n<details class="question"><summary>展开解答</summary>\n{body}\n</details></section>'
            return f'<div class="admonition {cls}"><p class="admonition-title">{title or default_title}</p>\n{body}\n</div>'
        if kind == "lstlisting":
            title = self.fragment_inline(d["title"])
            code = highlight(d["code"], get_lexer_by_name(d["language"], stripnl=False, ensurenl=False), HtmlFormatter(nowrap=True))
            caption = f'<p class="code-caption"><span class="code-number">代码 {d["number"]}　</span>{title}</p>\n' if title else ""
            return self.anchors([d["label"]]) + caption + '<div class="highlight"><pre><span></span><code>'+code+'</code></pre></div>'
        if kind in {"table", "tabular", "tabularx"}:
            title = self.fragment_inline(d["title"])
            caption = f'<p class="table-caption" id="{d["anchor"]}">'+(f'<span class="table-number">表 {d["number"]}　</span>' if d["number"] else "")+title+"</p>" if title else self.anchors([d["anchor"]])
            self.emitted.add(d["anchor"])
            aliases = self.anchors(d["labels"])
            return aliases+caption+'\n<div class="table-scroll" role="region" aria-label="表格" tabindex="0">\n'+self.blocks(self.book.asts[d["body"]])+"\n</div>"
        if kind == "figure":
            title = self.fragment_inline(d["title"])
            source = "assets/"+d["image"]
            url = self.url(source)
            alt = re.sub(r"<[^>]+>", "", title)
            self.emitted.add(d["anchor"])
            return self.anchors(d["labels"])+f'<figure id="{d["anchor"]}"><a href="{url}" target="_blank" rel="noopener" aria-label="查看图 {d["number"]} 原图" title="点击查看原图"><img src="{url}" alt="{ESC(html.unescape(alt),quote=True)}" loading="lazy"></a><figcaption><span class="figure-number">图 {d["number"]}　</span>{title}</figcaption></figure>'
        return self.anchors(d["labels"])+ '\n\n<div class="arithmatex">\\['+ESC(d["math"])+'\\]</div>'

    def render(self):
        body = self.blocks(self.book.asts[self.page.fragment], markdown_headings=True)
        if self.page.children:
            links = []
            for child in self.page.children:
                # Keep the framework's chapter-index section anchors as well
                # as the canonical anchors in complete reading pages.
                anchor = f' id="{child.anchor}"' if self.page.level == 1 else ""
                title = re.sub(r"\$([^$]+)\$", r'<span class="arithmatex">\\(\1\\)</span>', child.title)
                links.append(f'<li{anchor}><a href="{self.url(child.path)}">{child.number}　{title}</a></li>')
            body += '\n\n## 本章内容\n\n' if self.page.level == 1 else '\n\n## 本节内容\n\n'
            body += '<ul class="chapter-outline">\n'+"\n".join(links)+"\n</ul>"
        if self.notes:
            body += f'\n\n<div class="footnotes"><hr><ol start="{self.notes[0][0]}">\n'
            for i, note in self.notes:
                body += f'<li id="note-{i}">'+self.blocks(note)+f'<a href="#note-ref-{i}" aria-label="返回注 {i}">↩</a></li>\n'
            body += '</ol></div>'
        path = ROOT / "docs" / self.page.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pandoc", default=shutil.which("pandoc"))
    args = parser.parse_args()
    if not args.pandoc:
        parser.error("Pandoc is required; provide --pandoc /path/to/pandoc")
    book = Book(args.pandoc)
    book.model()
    book.bibliography()
    for page in book.pages:
        page.fragment = book.fragment(book.preprocess(page.source, page))
    book.read_fragments()
    for page in book.pages:
        Renderer(book, page).render()
    book.write_references()
    book.write_navigation()
    manifest = {
        "pages": [dict(path=p.path, chapter=p.chapter, number=p.number, anchor=p.anchor) for p in book.pages],
        "counts": dict(book.counts), "counters": dict(book.counters),
        "targets": book.targets, "images": sorted(book.images),
        "backlinks": book.backlinks, "source_notes": book.source_notes,
    }
    (WORK / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (WORK / "specials.json").write_text(json.dumps(book.specials,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Converted {len(book.pages)} pages across 30 chapters.")


if __name__ == "__main__":
    main()

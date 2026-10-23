"""Audit built chapter content against the active original TeX, not a checklist.

Checks each chapter's exact listing text, citation multiset, original labels,
figures, tables and Chinese prose order. Excludes presentation-only bylines,
navigation, numbering prefixes and relocated footnotes from the prose stream.
"""
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from book_structure import ROOT, active_tex, braced, catalog


class Article(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.article = False
        self.stack = []
        self.text = []
        self.pre = None
        self.codes = []
        self.math = None
        self.maths = []
        self.cites = []
        self.labels = set()
        self.figures = self.tables = self.questions = 0
        self.boxes = 0
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article": self.article = True
        if not self.article: return
        if attrs.get("id"): self.labels.add(attrs["id"])
        classes = set(attrs.get("class", "").split())
        skip = tag in {"h1","h2","h3","h4","h5","h6","pre","summary"} or bool(classes & {
            "chapter-outline","admonition-title","table-number","figure-number","code-number","footnotes"})
        if tag not in {"img","br","hr","input","wbr","meta","link"}:
            self.stack.append((tag, skip or bool(self.stack and self.stack[-1][1]), classes))
        if tag == "pre": self.pre = []
        if tag == "figure": self.figures += 1
        if tag == "table": self.tables += 1
        if "quiz-question" in classes: self.questions += 1
        if "admonition" in classes or "quiz-question" in classes: self.boxes += 1
        if "arithmatex" in classes and not any("chapter-outline" in s[2] for s in self.stack):
            self.math = (tag,[])
        if tag == "a" and "book-citation" in classes:
            self.cites.append(attrs["href"].split("#ref-")[1])

    def handle_endtag(self, tag):
        if not self.article: return
        if tag == "pre":
            self.codes.append("".join(self.pre or []).rstrip("\n"))
            self.pre = None
        if self.math is not None and tag == self.math[0]:
            self.maths.append("".join(self.math[1]))
            self.math = None
        if tag == "article": self.article = False
        if self.stack and self.stack[-1][0] == tag: self.stack.pop()

    def handle_data(self, data):
        if self.pre is not None: self.pre.append(data)
        if self.math is not None: self.math[1].append(data)
        if self.article and not (self.stack and self.stack[-1][1]): self.text.append(data)


def remove_braced_command(text, name):
    for m in reversed(list(re.finditer(r"\\"+name+r"\*?\s*(?=\{)",text))):
        _, end = braced(text,m.end())
        text = text[:m.start()]+text[end:]
    return text


def chinese(text):
    return "".join(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]",text))


def source_prose(text):
    # Keep listing captions in the original position; compare code separately.
    def listing(m):
        opt = m[1] or ""
        cap = re.search(r"caption\s*=\s*(?=\{)",opt)
        return braced(opt,cap.end())[0] if cap else ""
    text = re.sub(r"\\begin\{lstlisting\}(\[[^\n]*\])?\n.*?\\end\{lstlisting\}",listing,text,flags=re.S)
    text = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}","",text,flags=re.S)
    for command in ["chapter","section","subsection","subsubsection","paragraph","footnote"]:
        text = remove_braced_command(text,command)
    # Box titles are verified by coverage separately and omitted from both
    # prose streams because quiz titles become headings in the web edition.
    text = re.sub(r"(\\begin\{(?:keybox|intuitionbox|examplebox|warningbox|questionbox|tcolorbox)\})\[[^\n]*\]",r"\1",text)
    return chinese(text)


def math_content(text):
    text = re.sub(r"\\label\{[^}]+\}|\\tag\{\d+\.\d+\}","",text)
    text = re.sub(r"\\(?:begin|end)\{(?:equation|align|multline)\*?\}","",text)
    text = text.strip()
    if text.startswith((r"\[",r"\(")): text=text[2:-2]
    return re.sub(r"\s+","",text)


def source_math(text):
    text = re.sub(r"\\begin\{lstlisting\}(?:\[[^\n]*\])?\n.*?\\end\{lstlisting\}","",text,flags=re.S)
    text = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}","",text,flags=re.S)
    pattern = re.compile(r"\\begin\{(equation\*?|align\*?|multline\*?)\}(.*?)\\end\{\1\}|(?<!\\)\\\[(.*?)\\\]|(?<!\\)\$\$(.*?)\$\$|(?<!\\)\$(?!\$)(.*?)(?<!\\)\$",re.S)
    return [math_content(next(v for v in m.groups()[1:] if v is not None)) for m in pattern.finditer(text)]


def main():
    chapters = [c for p in catalog()["parts"] for c in p["chapters"]]
    n = 0
    report, errors = [], []
    for part in range(1,7):
        text = active_tex(ROOT / f"contents/part{part}.tex")
        starts = list(re.finditer(r"^\\chapter\{",text,re.M))
        for i, start in enumerate(starts):
            n += 1
            source = text[start.start():starts[i+1].start() if i+1<len(starts) else len(text)]
            folder = ROOT / "docs" / Path(chapters[n-1]["path"]).parent
            paths = [folder / "index.md"] + sorted(p for p in folder.glob("section-*.md"))
            # Parent section introductions precede their subsection pages.
            paths.sort(key=lambda p: (p.name != "index.md", tuple(int(x) for x in re.findall(r"\d+",p.stem))))
            docs = []
            for path in paths:
                relative = path.relative_to(ROOT / "docs")
                built = ROOT / "site" / (relative.with_suffix("") / "index.html" if relative.name != "index.md" else relative.with_suffix(".html"))
                docs.append(Article(built))
            source_codes = re.findall(r"\\begin\{lstlisting\}(?:\[[^\n]*\])?\n(.*?)\n\\end\{lstlisting\}",source,re.S)
            code = [c for d in docs for c in d.codes]
            source_cites = [k.strip() for g in re.findall(r"\\cite\{([^}]+)\}",source) for k in g.split(",")]
            cites = [k for d in docs for k in d.cites]
            labels = set(re.findall(r"\\label\{([^}]+)\}",source))
            source_figures = source.count(r"\begin{figure}")
            source_tables = len(re.findall(r"\\begin\{tabularx?\}",source))
            expected = source_prose(source)
            actual = chinese("".join(t for d in docs for t in d.text))
            expected_math = source_math(source)
            actual_math = [math_content(v) for d in docs for v in d.maths]
            expected_boxes = len(re.findall(r"\\begin\{(?:keybox|intuitionbox|examplebox|warningbox|questionbox|tcolorbox)\}",source))
            item = dict(chapter=n,pages=len(paths),codes=len(code),citations=len(cites),
                        figures=sum(d.figures for d in docs),tables=sum(d.tables for d in docs),
                        questions=sum(d.questions for d in docs),chinese_characters=len(expected),
                        exact_code=source_codes==code,citations_preserved=Counter(source_cites)==Counter(cites),
                        missing_labels=sorted(labels-set().union(*(d.labels for d in docs))),
                        prose_preserved=expected==actual, math_count=len(actual_math),
                        math_preserved=expected_math==actual_math,boxes=sum(d.boxes for d in docs))
            report.append(item)
            if not item["exact_code"]: errors.append(f"Chapter {n}: source code differs")
            if not item["citations_preserved"]: errors.append(f"Chapter {n}: citation coverage differs")
            if item["missing_labels"]: errors.append(f"Chapter {n}: missing source labels {item['missing_labels']}")
            if source_figures != item["figures"] or source_tables != item["tables"]:
                errors.append(f"Chapter {n}: figure/table count differs")
            if expected_boxes != item["boxes"]: errors.append(f"Chapter {n}: content box count differs")
            if expected_math != actual_math:
                j=next((j for j,(a,b) in enumerate(zip(expected_math,actual_math)) if a!=b),min(len(expected_math),len(actual_math)))
                errors.append(f"Chapter {n}: math differs at {j} ({len(expected_math)} / {len(actual_math)}):\n  source: {expected_math[j:j+2]}\n  output: {actual_math[j:j+2]}")
            if expected != actual:
                j = next((j for j,(a,b) in enumerate(zip(expected,actual)) if a!=b),min(len(expected),len(actual)))
                errors.append(f"Chapter {n}: prose differs at {j} ({len(expected)} / {len(actual)} chars):\n  source: {expected[max(0,j-25):j+70]}\n  output: {actual[max(0,j-25):j+70]}")
            print(f"Chapter {n:02}: {len(paths)} pages; {len(code)} listings; {len(cites)} citations; prose {'OK' if expected==actual else 'DIFF'}")
    out = ROOT / ".site-work/qa"
    out.mkdir(parents=True,exist_ok=True)
    (out / "book-content-audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if errors: raise SystemExit("\n".join(errors))
    print("All 30 chapters match the source content coverage and Chinese prose order.")


if __name__ == "__main__": main()

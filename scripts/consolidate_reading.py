"""Combine generated source sections into continuous chapter reading pages.

Run after convert_book.py, while its original section files still exist.
The committed layout records old routes so published bookmarks keep working.
"""
from collections import defaultdict
import html
import json
from pathlib import PurePosixPath
import posixpath
import re
from urllib.parse import unquote, urlsplit

from book_structure import ROOT, catalog

LAYOUT_PATH = ROOT / "scripts/reading-layout.json"

# Boundaries follow consecutive source topics; all other chapters are one page.
# Tuple: first source section, reading-page title (original headings stay intact).
GROUPS = {
    1: [(1, "从输入到输出"), (5, "训练理论与实践"),
        (9, "高效模型与训练多样性"), (12, "文本生成与 Prompt 工程"),
        (14, "压缩、加速与安全")],
    12: [(1, "训练基础与方法"), (6, "训练用例"), (8, "前沿方法与交互环境")],
    13: [(1, "推理基础与测试时扩展"), (3, "代表性推理模型"),
         (6, "数学方法、扩展规律与开放问题")],
    28: [(1, "基础、算法与实战"), (7, "硬件、优化与训练系统"),
         (11, "模型架构与效率"), (18, "Agent 训练、推理与评估"),
         (22, "Agent 系统与应用")],
}


def load_layout():
    return json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))


def route(path):
    p = PurePosixPath(path)
    return (str(p.parent) if p.name == "index.md" else str(p.with_suffix(""))) + "/"


def consolidate():
    manifest = json.loads((ROOT / ".site-work/conversion/manifest.json").read_text(encoding="utf-8"))
    originals = manifest["pages"]
    chapters = {c["number"]: c for p in catalog()["parts"] for c in p["chapters"]}
    prior_paths = {p["path"] for p in load_layout()["pages"]} if LAYOUT_PATH.exists() else set()
    texts, aliases = {}, {}
    for page in originals:
        path = ROOT / "docs" / page["path"]
        text = path.read_text(encoding="utf-8")
        # Prevent running twice against already consolidated files.
        if f'{{#{page["anchor"]}}}' not in text:
            raise ValueError(f"Regenerate the source sections before consolidating: {path}")
        aliases[page["path"]] = dict(re.findall(
            r'<li id="([^"]+)"><a href="([^"]+)"', text))
        text = re.sub(r'\n\n## 本[章节]内容\n\n<ul class="chapter-outline">.*?</ul>',
                      '', text, flags=re.S)
        texts[page["path"]] = text.strip()

    pages, redirects, groups = [], {}, defaultdict(list)
    for n, chapter in chapters.items():
        choices = GROUPS.get(n, [(1, chapter["title"])])
        for i, (start, title) in enumerate(choices):
            end = choices[i+1][0] - 1 if i+1 < len(choices) else len(chapter["sections"])
            path = str(PurePosixPath(chapter["path"]).with_name(
                "index.md" if i == 0 else f"reading-{i+1:02}.md"))
            number = f"{n}.{start}–{n}.{end}" if end > start else f"{n}.{start}"
            pages.append(dict(path=path, chapter=n, title=title, number=number,
                              first_section=start, last_section=end))
        chapter_pages = [p for p in pages if p["chapter"] == n]
        for original in (p for p in originals if p["chapter"] == n):
            parts = original["number"].split(".")
            section = int(parts[1]) if len(parts) > 1 else 0
            group = max((i for i, (start, _) in enumerate(choices) if start <= section), default=0)
            destination = chapter_pages[group]["path"]
            redirects[original["path"]] = dict(path=destination, anchor=original["anchor"], hashes={})
            groups[destination].append(original)

    # Citations were numbered per old page; merged pages must have unique IDs.
    for destination, members in groups.items():
        used = set()
        for member in members:
            old = member["path"]
            def unique(match):
                anchor = match[1]
                if anchor not in used:
                    used.add(anchor)
                    return match[0]
                if not anchor.startswith("cite-"):
                    raise ValueError(f"Unexpected duplicate content anchor: {old}#{anchor}")
                renamed = anchor + "--" + PurePosixPath(old).stem
                if renamed in used:
                    raise ValueError(f"Duplicate citation alias: {renamed}")
                used.add(renamed)
                redirects[old]["hashes"][anchor] = dict(path=destination, anchor=renamed)
                return 'id="' + renamed + '"'
            texts[old] = re.sub(r'\bid="([^"]+)"', unique, texts[old])
            used.update(re.findall(r'\{#([^}]+)\}', texts[old]))

    def resolve(old, href):
        url = urlsplit(html.unescape(href))
        if url.scheme or url.netloc or url.path.startswith("/"):
            return None
        target = posixpath.normpath(posixpath.join(str(PurePosixPath(old).parent), unquote(url.path))) if url.path else old
        if target not in redirects:
            return None
        redir = redirects[target]
        anchor = unquote(url.fragment)
        if anchor in redir["hashes"]:
            result = redir["hashes"][anchor]
        else:
            result = dict(path=redir["path"], anchor=anchor or (redir["anchor"] if PurePosixPath(target).name != "index.md" else ""))
        return dict(result, query=url.query)

    # Earlier chapter landing pages exposed section anchors in their outline.
    for old, outline in aliases.items():
        for anchor, href in outline.items():
            result = resolve(old, href)
            if result:
                redirects[old]["hashes"][anchor] = {k: result[k] for k in ("path", "anchor")}

    def href_from(destination, result):
        path = "" if result["path"] == destination else posixpath.relpath(result["path"], str(PurePosixPath(destination).parent))
        return path + ("?" + result["query"] if result.get("query") else "") + ("#" + result["anchor"] if result["anchor"] else "")

    output = {}
    for page in pages:
        destination, n = page["path"], page["chapter"]
        members = groups[destination]
        chunks, notes = [], []
        if not destination.endswith("/index.md"):
            chunks.append(f'# 第 {n} 章 · {page["title"]} {{#reading-{PurePosixPath(destination).stem}}}')
        for member in members:
            old = member["path"]
            text = texts[old]
            def rewrite(match):
                result = resolve(old, match[2])
                return match[1] + html.escape(href_from(destination, result), quote=True) + match[3] if result else match[0]
            text = re.sub(r'(<a\b[^>]*\bhref=")([^"]+)(")', rewrite, text)
            depth = len(member["number"].split(".")) - 1
            # Generated headings always have an explicit source anchor. Plain
            # hash-prefixed lines inside highlighted code are not headings.
            text = re.sub(r'^(#{1,6})( [^\n]* \{#[^}\n]+\})$',
                          lambda m: '#' * min(6, len(m[1])+depth) + m[2], text, flags=re.M)
            text = text.replace('<h2 class="question-title">', '<h3 class="question-title">').replace('</h2>\n<details class="question">', '</h3>\n<details class="question">')
            note = re.search(r'\n\n<div class="footnotes"><hr><ol start="(\d+)">\n(.*?)</ol></div>\s*$', text, re.S)
            if note:
                notes.append((note[1], note[2]))
                text = text[:note.start()]
            chunks.append(text)
            if old.endswith("/index.md") and n in GROUPS:
                links = []
                for sibling in (p for p in pages if p["chapter"] == n):
                    first = groups[sibling["path"]][1] if sibling["path"] == destination else groups[sibling["path"]][0]
                    href = href_from(destination, dict(path=sibling["path"], anchor=first["anchor"]))
                    links.append(f'<li><a href="{html.escape(href, quote=True)}">{sibling["number"]} · {sibling["title"]}</a></li>')
                chunks.append('<ul class="chapter-outline" aria-label="本章阅读">\n' + '\n'.join(links) + '\n</ul>')
        if notes:
            chunks.append(f'<div class="footnotes"><hr><ol start="{notes[0][0]}">\n' + ''.join(s for _, s in notes) + '</ol></div>')
        output[destination] = '\n\n'.join(chunks) + '\n'

    # Update links in manually maintained pages too, including reading samples.
    original_paths = {p["path"] for p in originals}
    for path in (ROOT / "docs").rglob("*.md"):
        relative = path.relative_to(ROOT / "docs").as_posix()
        if relative in original_paths | prior_paths:
            continue
        text = path.read_text(encoding="utf-8")
        def rewrite_other(match):
            result = resolve(relative, match[2])
            return match[1] + html.escape(href_from(relative, result), quote=True) + match[3] if result else match[0]
        updated = re.sub(r'(<a\b[^>]*\bhref=")([^"]+)(")', rewrite_other, text)
        def rewrite_markdown(match):
            result = resolve(relative, match[2])
            return match[1] + href_from(relative, result) + match[3] if result else match[0]
        updated = re.sub(r'(\]\()([^\s)]+)(\))', rewrite_markdown, updated)
        if updated != text:
            output[relative] = updated

    # Only retire explicitly listed generated files, within the chapters tree.
    chapter_root = (ROOT / "docs/chapters").resolve()
    for old in original_paths | prior_paths:
        path = (ROOT / "docs" / old).resolve()
        if not path.is_relative_to(chapter_root):
            raise ValueError(f"Generated path leaves chapters directory: {path}")
        if old not in output:
            path.unlink(missing_ok=True)
    for relative, text in output.items():
        (ROOT / "docs" / relative).write_text(text, encoding="utf-8")
    LAYOUT_PATH.write_text(json.dumps(dict(pages=pages, redirects=redirects), ensure_ascii=False, indent=2) + '\n', encoding="utf-8")
    write_navigation(pages)
    print(f"Consolidated {len(originals)} source pages into {len(pages)} reading pages.")


def write_navigation(pages):
    q = lambda s: json.dumps(s, ensure_ascii=False)
    config = ROOT / 'zensical.toml'
    text = config.read_text(encoding='utf-8')
    start, end = text.index('nav = ['), text.index('\n[project.theme]')
    lines = ['nav = [', '  { "首页" = "index.md" },', '  { "全书目录" = "book-outline.md" },',
             '  { "阅读样例" = "reading-sample.md" },', '  { "卷首" = [']
    for slug, title in [('dedication','献辞'),('disclaimer','免责声明'),('author','关于作者'),('preface','前言'),('introduction','引言'),('glossary','缩略语表')]:
        lines.append(f'    {{ {q(title)} = "front-matter/{slug}.md" }},')
    lines.append('  ] },')
    roman = ['I','II','III','IV','V','VI']
    for part in catalog()['parts']:
        lines += [f'  {{ {q("第 "+roman[part["number"]-1]+" 部分 · "+part["title"])} = [',
                  f'    {{ "本部分目录" = {q(part["path"])} }},']
        for chapter in part['chapters']:
            title = f'第 {chapter["number"]} 章 · {chapter["title"]}'
            members = [p for p in pages if p['chapter'] == chapter['number']]
            if len(members) == 1:
                lines.append(f'    {{ {q(title)} = {q(members[0]["path"])} }},')
            else:
                lines.append(f'    {{ {q(title)} = [')
                for page in members:
                    lines.append(f'      {{ {q(page["number"]+" · "+page["title"])} = {q(page["path"])} }},')
                lines.append('    ] },')
        lines.append('  ] },')
    lines += ['  { "参考文献" = "references.md" },', '  { "来源与署名" = "about.md" },', ']']
    config.write_text(text[:start] + '\n'.join(lines) + '\n' + text[end:], encoding='utf-8')


if __name__ == '__main__':
    consolidate()

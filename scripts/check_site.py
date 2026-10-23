"""Check built pages, source chapter coverage, local assets, and Pages paths."""

from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
from functools import lru_cache
import json
import os
from pathlib import Path
import tomllib

from book_structure import ROOT, catalog


@lru_cache(maxsize=None)
def local_target(path):
    target = Path(path).resolve()
    if target.is_dir():
        target /= "index.html"
    return target, target.is_file()


class Document(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.anchors = set()
        self.duplicates = set()
        self.links = []
        self.language = ""
        self.bare_math = False
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        anchor = attrs.get("id")
        if anchor:
            if anchor in self.anchors:
                self.duplicates.add(anchor)
            self.anchors.add(anchor)
        if tag == "html":
            self.language = attrs.get("lang", "")
        if tag in ("a", "link") and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag in ("img", "script", "source") and attrs.get("src"):
            self.links.append(attrs["src"])

    def handle_data(self, text):
        if any(line.strip() == "$$" for line in text.splitlines()):
            self.bare_math = True


def main():
    root = (ROOT / "site").resolve()
    config = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))["project"]
    base = urlsplit(config["site_url"])
    prefix = base.path.rstrip("/")
    pages = {path.resolve(): Document(path) for path in root.rglob("*.html")}
    errors = []
    if root / "index.html" not in pages:
        raise SystemExit("Build the website before checking it.")
    for path, page in pages.items():
        if not page.language.startswith("zh"):
            errors.append(f"{path.relative_to(root)}: HTML language must be Chinese.")
        if page.duplicates:
            errors.append(f"{path.relative_to(root)}: duplicate anchors {page.duplicates}")
        if page.bare_math:
            errors.append(f"{path.relative_to(root)}: display math was not converted.")
        for href in page.links:
            url = urlsplit(href)
            if url.scheme and url.scheme not in ("http", "https"):
                continue
            if url.netloc and url.netloc != base.netloc:
                continue
            raw = unquote(url.path)
            if raw.startswith("/"):
                if prefix and not (raw == prefix or raw.startswith(prefix + "/")):
                    errors.append(f"{path.relative_to(root)}: URL loses the Pages prefix: {href}")
                    continue
                raw = raw[len(prefix):]
                target = root / raw.lstrip("/")
            else:
                target = path.parent / raw if raw else path
            target, exists = local_target(os.path.abspath(target))
            if not target.is_relative_to(root):
                errors.append(f"{path.relative_to(root)}: URL leaves the website: {href}")
                continue
            if not exists:
                errors.append(f"{path.relative_to(root)}: missing local target: {href}")
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].anchors:
                errors.append(f"{path.relative_to(root)}: missing anchor: {href}")
    book = catalog()
    for part in book["parts"]:
        for chapter in part["chapters"]:
            target = root / chapter["path"].replace(".md", ".html")
            page = pages.get(target)
            if page is None:
                errors.append(f"Missing chapter {chapter['number']}: {target}")
                continue
            for section in chapter["sections"]:
                anchor = section["label"] or f"section-{section['number'].replace('.', '-')}"
                if anchor not in page.anchors:
                    errors.append(f"Missing section entry {section['number']}: {anchor}")
    for relative in (
        "assets/vendor/mathjax/tex-chtml.js",
        "assets/vendor/mathjax/LICENSE",
        "assets/vendor/mathjax/input/tex/extensions/mathtools.js",
        "assets/vendor/mathjax/input/tex/extensions/boldsymbol.js",
        "assets/vendor/mathjax/output/chtml/fonts/woff-v2/MathJax_Main-Regular.woff",
    ):
        if not (root / relative).is_file():
            errors.append(f"Missing runtime asset: {relative}")
    search = json.loads((root / "search.json").read_text(encoding="utf-8"))
    if not {"zh", "en"}.issubset(search.get("config", {}).get("lang", [])):
        errors.append("Chinese and English search segmentation is not enabled.")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(pages)} pages, 30 chapters and {book['section_count']} section anchors; links and assets OK.")


if __name__ == "__main__":
    main()

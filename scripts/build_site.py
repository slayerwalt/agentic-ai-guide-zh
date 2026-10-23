"""Build and check the web edition using the pinned, shared site toolchain."""

from pathlib import Path
import json
import re
import shutil
import subprocess
import sys

from reading_redirects import write_redirects

ROOT = Path(__file__).resolve().parents[1]
# Only assets used by published Markdown are staged; figures/ remains the source.


def main():
    destination = ROOT / "docs/assets/figures"
    destination.mkdir(parents=True, exist_ok=True)
    figures = set()
    for page in (ROOT / "docs").rglob("*.md"):
        figures.update(re.findall(r'assets/figures/([^"<>\s)]+)', page.read_text(encoding="utf-8")))
    for name in sorted(figures):
        shutil.copy2(ROOT / "figures" / name, destination / name)
    renderer = ROOT / "docs/assets/vendor/mathjax/tex-chtml.js"
    if not renderer.is_file():
        raise SystemExit("Run python scripts/vendor_mathjax.py before building.")
    subprocess.run(
        [sys.executable, "-m", "zensical", "build", "--clean", "--strict"],
        cwd=ROOT, check=True,
    )
    index_path = ROOT / "site/search.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["config"]["lang"] = ["zh", "en"]
    index_path.write_text(json.dumps(index, ensure_ascii=False) + "\n", encoding="utf-8")
    write_redirects()
    subprocess.run([sys.executable, "scripts/check_site.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/check_book_content.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

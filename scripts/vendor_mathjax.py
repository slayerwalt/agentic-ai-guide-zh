"""Stage the pinned MathJax CHTML renderer and its runtime dependencies."""

import argparse
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = "3.2.2"
URL = f"https://registry.npmjs.org/mathjax/-/mathjax-{VERSION}.tgz"
DEST = ROOT / "docs/assets/vendor/mathjax"


def required(relative: PurePosixPath) -> bool:
    path = relative.as_posix()
    return path in ("tex-chtml.js", "LICENSE") or path.startswith((
        "input/tex/extensions/", "output/chtml/", "a11y/", "sre/", "ui/",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-directory", type=Path,
                        help="Reuse a local, unmodified MathJax 3.2.2 distribution.")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.from_directory:
        source = args.from_directory.resolve()
        notice = (source / "NOTICE.txt").read_text(encoding="utf-8")
        if f"MathJax {VERSION}" not in notice:
            raise SystemExit("The local MathJax distribution has a different version.")
        for path in source.rglob("*"):
            relative = PurePosixPath(path.relative_to(source).as_posix())
            if path.is_file() and required(relative):
                target = DEST / str(relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    else:
        archive_path = ROOT / ".cache" / f"mathjax-{VERSION}.tgz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if not archive_path.is_file():
            temporary = archive_path.with_suffix(".download")
            try:
                request = urllib.request.Request(URL, headers={"User-Agent": "AgenticGuideWeb/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
                temporary.replace(archive_path)
            finally:
                temporary.unlink(missing_ok=True)
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if not member.isfile():
                    continue
                if path.parts[:2] == ("package", "es5"):
                    relative = PurePosixPath(*path.parts[2:])
                elif member.name == "package/LICENSE":
                    relative = PurePosixPath("LICENSE")
                else:
                    continue
                if not required(relative):
                    continue
                target = (DEST / str(relative)).resolve()
                if not target.is_relative_to(DEST.resolve()):
                    raise ValueError(f"Unsafe archive member: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    for path in ("tex-chtml.js", "LICENSE", "output/chtml/fonts/woff-v2/MathJax_Main-Regular.woff"):
        if not (DEST / path).is_file():
            raise SystemExit(f"MathJax dependency missing: {path}")
    (DEST / "NOTICE.txt").write_text(
        f"MathJax {VERSION}\nSource: {URL}\nLicense: Apache-2.0; see LICENSE.\n"
        "Selected CHTML browser components copied without modification.\n",
        encoding="utf-8",
    )
    print(f"MathJax {VERSION}: CHTML renderer and runtime dependencies ready.")


if __name__ == "__main__":
    main()

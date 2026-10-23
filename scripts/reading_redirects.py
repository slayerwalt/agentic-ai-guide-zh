"""Keep published section URLs and fragments working after page consolidation."""
import html
import json
import posixpath

from book_structure import ROOT
from consolidate_reading import load_layout, route


def built_path(markdown):
    return ROOT / "site" / route(markdown) / "index.html"


def write_redirects():
    layout = load_layout()
    count = 0
    for old, record in layout["redirects"].items():
        old_route = route(old)
        def relative(target):
            result = posixpath.relpath(route(target["path"]).rstrip("/"), old_route.rstrip("/")) + "/"
            return result + ("#" + target["anchor"] if target["anchor"] else "")
        moved = old != record["path"]
        hashes = {key: relative(target) for key, target in record["hashes"].items()
                  if moved or target["path"] != old or target["anchor"] != key}
        if not moved and not hashes:
            continue
        data = dict(default=relative(record) if moved else None, hashes=hashes)
        encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
        script = '''<script>
(() => {
  const routes = ROUTES;
  let fragment = location.hash.slice(1);
  try { fragment = decodeURIComponent(fragment); } catch (_) {}
  const mapped = Object.prototype.hasOwnProperty.call(routes.hashes, fragment)
    ? routes.hashes[fragment] : null;
  const destination = mapped || routes.default;
  if (!destination) return;
  const target = new URL(destination, location.href);
  target.search = location.search;
  if (!mapped && location.hash) target.hash = location.hash;
  if (target.href !== location.href) location.replace(target.href);
})();
</script>'''.replace("ROUTES", encoded)
        path = built_path(old)
        if moved:
            destination = html.escape(data["default"], quote=True)
            page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<link rel="canonical" href="{destination}">
<title>继续阅读 · 智能体 AI 漫游指南</title>
{script}</head><body>
<p>本节已合并到章节阅读页。<a href="{destination}">继续阅读</a></p>
</body></html>
'''
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(page, encoding="utf-8")
            count += 1
        else:
            page = path.read_text(encoding="utf-8")
            path.write_text(page.replace("</head>", script + "</head>", 1), encoding="utf-8")
    print(f"Prepared {count} redirects for previously published section URLs.")


if __name__ == "__main__":
    write_redirects()

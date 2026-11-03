"""Reject builds that break URLs captured from the pre-Zensical published site."""

import json
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urljoin, urlsplit
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.anchors = set()
        self.heading = []
        self.in_heading = False
        self.canonical = None
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.anchors.add(attrs["id"])
        if tag == "h1":
            self.in_heading = True
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical = attrs.get("href")
        if tag == "img" and attrs.get("src"):
            self.images.append(attrs["src"])

    def handle_endtag(self, tag):
        if tag == "h1":
            self.in_heading = False

    def handle_data(self, data):
        if self.in_heading:
            self.heading.append(data)


def check(site):
    manifest = json.loads((ROOT / ".github/legacy-urls.json").read_text())
    base = manifest["site_url"]
    errors = []
    article_count = 0
    anchor_count = 0
    for url, expected in manifest["pages"].items():
        path = site / (url + "index.html" if not url.endswith(".html") else url)
        if not path.is_file():
            errors.append(f"Missing old page: {url or '/'}")
            continue
        page = Page()
        page.feed(path.read_text(encoding="utf-8"))
        if url != "404.html" and page.canonical != urljoin(base, url):
            errors.append(f"Changed canonical URL: {url or '/'}")
        if expected:
            article_count += 1
            if "".join(page.heading).strip() != expected["title"]:
                errors.append(f"Different article at {url}")
            missing = set(expected["anchors"]) - page.anchors
            anchor_count += len(expected["anchors"])
            if missing:
                errors.append(f"Missing anchors at {url}: {', '.join(sorted(missing))}")
        for src in page.images:
            target = urljoin(urljoin(base, url), src)
            if target.startswith(base):
                local = unquote(urlsplit(target).path[len(urlsplit(base).path):])
                if not (site / local).is_file():
                    errors.append(f"Missing image at {url}: {src}")

    for name in manifest["files"]:
        if not (site / name).is_file():
            errors.append(f"Missing old asset or feed: {name}")

    for name in ("feed_rss_created.xml", "feed_rss_updated.xml"):
        path = site / name
        if not path.is_file():
            continue
        for item in ET.parse(path).findall("./channel/item"):
            link = item.findtext("link", "")
            if not link.startswith(base):
                errors.append(f"Unexpected feed URL in {name}: {link}")
                continue
            relative = unquote(urlsplit(link).path[len(urlsplit(base).path):])
            if not (site / relative / "index.html").is_file():
                errors.append(f"Broken feed URL in {name}: {link}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"Verified {len(manifest['pages'])} legacy pages, {article_count} articles, "
        f"{anchor_count} anchors, images and RSS links."
    )


if __name__ == "__main__":
    check(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "site")

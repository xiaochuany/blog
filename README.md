# Xiaochuan's blog

Built with [Zensical](https://zensical.org/) and its native blog, tags, search,
and RSS support. Published at <https://xiaochuany.github.io/blog/>.

## Develop locally

Use Python 3.12:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
zensical serve
```

Build and check all pre-migration links before publishing:

```sh
zensical build --clean --strict
python scripts/check_legacy_urls.py
```

Configuration stays in `mkdocs.yml`, which Zensical supports directly. The
`classic` theme keeps the existing appearance, light/dark toggle, code copying,
MathJax, Mermaid, social links, and analytics.

## Write a post

Add a Markdown file under `docs/posts/` with front matter:

```yaml
---
date:
  created: 2026-11-03
authors: [xy]
categories: [TIL]
tags: [python]
---
```

Follow it with a `# Title`, an introduction, and `<!-- more -->` to end the
excerpt. Supported categories are listed in `mkdocs.yml`. Authors are defined in
`docs/.authors.yml`. Set `draft: true` to preview a post without publishing it.
Dates in metadata determine URLs; filenames can have different dates.

## Existing links

The migration preserves the `/blog/` base path, directory URLs, and the
`yyyy/MM/dd/{slug}/` article format. Existing explicit `slug` values are retained.
The previous `pymdownx.slugs.slugify(case="lower", short=True)` setting is
replaced by Zensical's built-in default, verified against the published output.
No redirects are needed because the article URLs are unchanged.

`.github/legacy-urls.json` records the published routes from `gh-pages` commit
`ad51483314521232f75a0afdf897585e7dad9644`, generated from source commit
`60f8b6309680f47775166205545e2104d889b726`. The checker verifies every captured
page, each article's title and anchors, its canonical URL, images, and RSS links.
CI runs it before deployment to prevent link regressions.

When changing an existing article's title, set `slug:` to its original URL slug
so the address stays stable. If a deliberate title or section change affects the
snapshot, update the corresponding entry only after preserving any old links.
Keep the original baseline entries when adding posts.

## Publish

Pull requests build and check the site. Pushes to `main` or `master` publish the
checked artifact to the existing `gh-pages` branch with `ghp-import`. GitHub
Pages continues serving that branch; no hosting or domain change is needed.

The old `mkdocs gh-deploy` command is replaced by `zensical build` followed by
`ghp-import --no-jekyll --push site`. Dependencies are pinned in
`requirements.txt` and CI uses Python 3.12.

Originally based on the
[Material for MkDocs blog template](https://github.com/mkdocs-material/create-blog/blob/main/mkdocs.yml).

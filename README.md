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

Build before publishing:

```sh
zensical build --clean --strict
```

Configuration stays in `mkdocs.yml`, which Zensical supports directly. The
`classic` theme keeps the existing appearance, light/dark toggle, code copying,
MathJax, Mermaid, social links, and analytics.

## Write a post

Add a Markdown file under `docs/posts/` with front matter:

```yaml
---
date:
  created: 2026-11-04
slug: my-next-article
authors: [xy]
categories: [TIL]
tags: [python]
---
```

Follow it with a `# Title`, an introduction, and `<!-- more -->` to end the
excerpt. Supported categories are listed in `mkdocs.yml`. Authors are defined in
`docs/.authors.yml`. Set `draft: true` to preview a post without publishing it.
The creation date and explicit slug determine the URL; filenames can have
different dates. Choose a unique date-and-slug combination for each new post.

## Existing links

The blog uses the `/blog/` base path, directory URLs, and the
`yyyy/MM/dd/{slug}/` article format. Every existing article, including drafts,
has an explicit `slug` matching its previously generated address. No redirects
are needed because the article URLs are unchanged.

Keep `date.created` and `slug` unchanged after publication. Titles and body text
can be edited independently of the article address; use `date.updated` to record
revisions. Include an explicit slug when creating new articles.

## Publish

Pull requests build the site in strict mode. Pushes to `main` or `master` publish
the built artifact to the existing `gh-pages` branch with `ghp-import`. GitHub
Pages continues serving that branch; no hosting or domain change is needed.

The old `mkdocs gh-deploy` command is replaced by `zensical build` followed by
`ghp-import --no-jekyll --push site`. CI uses Python 3.12. Zensical is unpinned
in `requirements.txt`, allowing clean CI builds to install new releases. Update
your local environment with `pip install --upgrade -r requirements.txt`.

Originally based on the
[Material for MkDocs blog template](https://github.com/mkdocs-material/create-blog/blob/main/mkdocs.yml).

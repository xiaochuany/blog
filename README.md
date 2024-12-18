# blog

made with `material for mkdocs`

## credit

[template by material team](https://github.com/mkdocs-material/create-blog/blob/main/mkdocs.yml)

## first steps to follow

1. `pip install mkdocs-material`
1. init docs locally `mkdocs new .` 
1. create empty remote repo and push local to remote 
1. add ci config file as per [material docs](https://squidfunk.github.io/mkdocs-material/publishing-your-site/)
1. set publish branch to `gh-pages`

after these steps, ci should work and a site is published. new edits of docs in the main branch would trigger rebuild of the site. 

## customize the theme

in `mkdocs.yml` config the theme, then customize it e.g. [dark/light toggle](https://squidfunk.github.io/mkdocs-material/setup/changing-the-colors/)

## add blog plugin

without any setting, the plugin creates a directory structure (`/docs/blog/posts`). 

Setting the following should be straightforward

1. blog_toc
1. archive_date_format
1. categories_allowed
1. pagination_per_page

the `post_slugify` setting makes use of python markdown extension package, which is a dependency of `material`.  

## add tags plugin

add `tags` plugin in `mkdocs.yml` and `tags.md` in `docs`.

## add rss plugin

rss is a third party plugin which requires installation. 

It is important that have the site_name, site_description and site_url settings configured as instructed in the basic blog tutorial. The RSS plugin makes use of this information to construct the feed, so make sure you have configured them.

- `pip install mkdocs-rss-plugin`
- add rss in  `mkdocs.yml`
- add in ci: run `pip install mkdocs-rss-plugin`

## extra 

such as social media accounts in the footer

## mdx: maths

- add `arithmatex` extension in `mkdocs.yml`
- add `mathjax.js` to  extra_javascript (file put under `docs/js`, define macros as needed)   

can use `katex` instead per [documentation](https://squidfunk.github.io/mkdocs-material/reference/math/?h=math)

## mdx: code block

- add highlight, inlinehilite, snippets, superfences

## nav

rename nav (side) bar and/or turn it around. 

add to `features` in `theme` the following   
- navigation.tabs : tabs
- navigation.indexes : index attached to sections (overview page)

also add `nav` section to be explicit what to include in the sidebar/tabs. 

## author

add `docs/blog/.author.yml` and use it in all posts


## metadata of posts

metadata include: 

- date (enough)
- authors
- tags
- categories
- slug (if want to customize)
- readtime

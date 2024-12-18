# blog

made with `material for mkdocs`

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

without setting the plugin creates a directory structure (`/docs/blog/posts`). 

Setting the follows should be straightforward. 

1. blog_toc
1. archive_date_format
1. categories_allowed
1. pagination_per_page




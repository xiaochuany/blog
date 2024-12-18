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

in `mkdocs.yml` config the theme

    theme:
        name: material

then customize it according to needs. 

1. dark/light switch

        palette:

            - media: "(prefers-color-scheme: light)"
            scheme: default
            toggle:
                icon: material/brightness-7
                name: Switch to dark mode

            - media: "(prefers-color-scheme: dark)"
            scheme: slate
            toggle:
                icon: material/brightness-4
                name: Switch to system preference

1. 
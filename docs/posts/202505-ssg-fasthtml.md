---
date:
    created: 2025-05-01
authors: [xy]
categories: [TIL]
tags: [dev-tools]
---

# Static site generator using fasthtml
<!-- more -->
## genesis

Currently I use mkdocs (with the `material` theme) for both blog and documentation. 

The thing I like the most about material for mkdocs is that it is easy to get started. Copy mkdocs.yml from your favorite projects which uses material for mkdocs (polars, uv, fastapi in my case) and start writing your md files. Essentially a no code solution for writing static contents.

The ease of use comes at a price though. It may be hard to tweak one little thing you want which is not supported in the yaml config file or/and mkdocs plugin. To do that, you have to dig into the template, the partials and learn Jinja2 template language if you haven't already. It's too much.

Practically all SSG out there seem to follow the same components: a markdown parser, a template system with Jinja2-like language and html/css files. 

It strikes me that fasthtml can do all that with just python. I am not interested in something achieving feature parity with mkdocs, hugo, jekyll, but rather a minimal system that I have full control and maximum customization for SSG use case. 

## archetecture

Write a function that chains fasttags leaving as variable the things I want to inject into the function from a parsed markdown file (metadata, content etc). This function is the template. All styles and scripts are embedded into the function.  Markdown files can be rendered with mistletoe (which come with monsterui,  fasthtml's UI components library).  

for example

```py

import yaml
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from fasthtml.common import *
from monsterui.all import * # include render_md
    

@dataclass
class Page:
    """from markdown to html"""
    path: Path
    content: str
    meta: dict

    @classmethod
    def from_file(cls, path: Path):
        """Load a page from a file, parsing front matter and content."""
        text = path.read_text()
        if "---" in text:
            meta_str, content = text.split("---", 2)[1:]
            meta = yaml.safe_load(meta_str)
        else:
            meta, content = {}, text
        return cls(path, content, meta)
    
    def render(self) -> str:
        """Plug page data into a template."""        
        content = render_md(self.content)
        title=  self.meta.get("title", "Untitled"),
        return base_template(title, content, self.meta)
        
def base_template(title, content, meta, hdrs=Theme.blue.headers(daisy=True, highlightjs=True)) -> str:
    """Core layout template using MonsterUI components"""
    return to_xml(Html(
            Head(
                Title(title),
                Meta(charset="utf-8"),
                Meta(name="viewport", content="width=device-width, initial-scale=1.0"),
                *hdrs  # Theme is a list
            ),
            Body(
                Container(
                    NavBar(
                        A("Home", href="/"),
                        A("Blog", href="/blog"),
                        brand=H2("My SSG"),
                    ),
                    Main(
                        Article(
                            Header(
                                H1(title),
                                Div(
                                    Small(meta.get('date',None)),
                                    cls=TextPresets.muted_sm
                                ) if meta.get('date') else None,
                            ),
                            Div(Safe(content)),  # Markdown content
                        ),
                        cls="max-w-3xl mx-auto"
                    ),
                    Footer(
                        P(f"© {datetime.now().year} My Static Site", cls=TextPresets.muted_sm),
                        cls="mt-8 text-center")))))
```

Now I can do this to generate one file (or write a loop to handle a directory of md docs). 

```py
Path("index.html").write_text(Page.from_file(Path('README.md')).render())
```

---
date:
    created: 2025-01-03
    updated: 2025-01-09
tags: 
    - dev tools
categories: 
    - Tutorial
slug: packaging
authors:
    - xy
---

# Mini tutorial for python packaging

!!! abstract 
    This mini-tutorial is a beginner's cheatsheet to python packaging. Check [Python packaing user guide](https://packaging.python.org/en/latest/) for an authoritative guidance on the topic.  
<!-- more -->

## Classical way

- install packages:

    ```bash
    pip install wheel build
    ```

- create `setup.py`:  

    ```py
    from setuptools import setup, find_packages

    setup(
        name='my_package',
        version='0.1.0',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'pandas',
        ],
        # Optional: Add more metadata
    )
    ```

- create build distribution and source distribution:

    ```bash
    python setup.py sdist bdist_wheel
    ```

## Alternatively, with `uv`

`uv` is a modern python dev tool, see [features](https://docs.astral.sh/uv/getting-started/features/)
and 
[install guide](https://docs.astral.sh/uv/getting-started/installation/).

Use the project interface of `uv` to init project and add dependencies. 

```bash
uv init
uv add DEPENDENCIES
```

then 

```bash
uv build
```

## Create a tag and release

```sh
git tag -a v0.1.0 -m 'release message'
git push --tags
```

This will create a tag named "v0.1.0" with the message "release message". The distribuion files will be displayed as assets for the tag.


## Publish

Publish to pypi index requires the developer to setup an account and get the API token. Using `uv`, run this


```sh
uv publish --token TOKEN
```

with the dev's API token in place of TOKEN. 

## Further reading

!!! links
    - why `pyproject.toml`? https://peps.python.org/pep-0518/  
    - understand build frontend, build backend. https://peps.python.org/pep-0517

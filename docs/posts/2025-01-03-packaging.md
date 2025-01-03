---
date:
    created: 2025-01-03
tags: 
    - dev tools
categories: 
    - Tutorial
slug: packaging
---

# Mini tutorial for python packaging

!!! abstract 
    This mini-tutorial is a beginner's cheatsheet to python packaging. Check [Python packaing user guide](https://packaging.python.org/en/latest/) for an authoritive guidance on the topic.  
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

- create a wheel with source distribution:

    ```bash
    python -m build --sdist
    ```

## With `uv`

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


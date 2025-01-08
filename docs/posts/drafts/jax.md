---
draft: true
date:
    created: 2025-01-08
categories: 
    - Tutorial
authors:
    - xy
tags:
    - scientific computing
---

# Computing with `jax`

!!! abstract
    Mutiple packages came to solve the speed issue of `python` in scientific computing. The de facto standard is of course `numpy`. Think also `numba`, `jax`, `torch` or even a new language :fire: that go beyond what `numpy` offers.  This post is a quick into to my personal preference as of Jan 2025. 


## Numpy operations

The part of `jax` that's straightforward to use for `numpy` users is the `jax.numpy` module, which has the identical API. Like other parts of `jax`, these operations are built on top of the XLA compiler for high performance numerical computations. 

## Random numbers 

Be aware that `jax` follows the functional programming paradigm. This means explicit key handling for samplers. The samplers can be composed with `vmap` to achieve vectorization
across all parameters, e.g. `random.t` has two parameters (key and df), one can supply two arrays to the `vmap(random.t)`. 


```py exec="on" result="text" source="above"
--8<-- "jx.py:random"
```

Notice that we turn the subkeys (list) into array because `jax` expects array for vectorized functions. 

See details in the [official docs](https://jax.readthedocs.io/en/latest/jax.random.html#module-jax.random). 

## `vmap`

We've had a flavor of `vmap`. More control over the operating axes is possible.  

```py exec="on" result="text" source="above"
--8<-- "jx.py:vmap"
```

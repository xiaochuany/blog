---
date: 
    created: 2025-10-31
    updated: 2025-11-05
authors: [xy]
categories: [Analysis]
tags: [low latency programming]
draft: true
---

# Manipulate convolutions in deep learning frameworks with ease 

<!-- more -->

## Original idea

Given a signal $f$ and a measurement function $g$, say a probability density function on a compact subset of $\mathbb R^d$ around the origin, the convolution $f*g$ at point $x$ tells us 
a sort of average of $f$ around the point $x$. 

## Fine grain control in deep learning frameworks 

There are a few knobs to control its behaviour  

- Stride is about how to move kernels.  `stride=k` means you slide kernel `k` position at a time.
- Padding is about adding zeros to the boundary of the input. 
- Input dilation is about adding zeros in between positions of the input, thereby inflating the input size by a factor.  
- Kernel dilation is about adding zeros in between positions of the kenrel, thereby expanding the reach of a kernel with the same parameter size. 

Using/combining these allows to define layers that down-samples or up-samples an input feature map. Let's cover some in this post. 

## 1d batched 

```py
from flax import nnx
import jax.numpy as jnp

rngs: nnx.Rngs
B: int
T: int
K: int
C_in: int
C_out: int

layer = nnx.Conv(C_in, C_out, K, rngs=rngs)
x = jnp.ones((B,T,C_in))
out = layer(x)
```

What is the output shape?  

- By default `padding="valid"` so that answer is `B, T-K+1, C_out`. 
- If `padding="same"` then `B,T, C_out`. 

## 2d batched 

```py
H: int
W: int

layer = nnx.Conv(C_in, C_out, (K,K), rngs=rngs)
x = jnp.ones((B,H,W,C_in))
```

what is the output shape? 
- by default, `B, H-K+1, W-K+1, C_out`
- if padding is "SAME" or "CIRCULAR" or "REFLECT", then `B,H,W,C_out`.  

SAME pads `K-1` zeros on both sides, whereas CIRCULAR identifies opposing  boundaries so it is valid to compute the convolution on all spatial locations. REFLECT creates a symmetry around the boudary. 

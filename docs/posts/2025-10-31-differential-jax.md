---
date: 
    created: 2025-10-31
authors: [xy]
categories: [Analysis]
tags: [low latency programming]
draft: true
---

# Jax as a differntial calculus tool

Back in January, I wrote a tutorial about Jax, highlighting its power in high performance computing and its near-mathematical syntax.
Now I show how to use Jax as a differential calculus tool. 

The idea is to demonstrate the approximation of a differentiable function by a few terms in its Taylor expansion near a fixed point. The statement is 

$$
f(x) = f(x_0) + \nabla f(x_0)(x-x_0) +  \langle Hf(x_0) (x-x_0), (x-x_0)\rangle + O(\|x-x_0\|^3)
$$


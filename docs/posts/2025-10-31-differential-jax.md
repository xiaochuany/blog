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
Now I show how to use Jax as a differential calculus tool for students and educators. 

The goal is to approximate a differentiable function by a few terms in its Taylor expansion near a fixed point. The neat mathematical statement is 

$$
f(x) - f(x_0) = \nabla f(x_0)(x-x_0) +  \frac{1}{2}\langle Hf(x_0) (x-x_0), (x-x_0)\rangle + O(\|x-x_0\|^3)
$$

where $f:\mathbb{R}^d \to \mathbb{R}$ is sufficiently differentiable,  $\nabla f$ is the gradient of $f$ and $Hf$ is its Hessian matrix. 

Based on the expansion, we can use the linear term (first on the right) or the quadratic form (first two terms on the right) to approximate $f$. 
We can go further down the expansion too. In the code below we go down the quadratic route.  

```py
import jax
import jax.numpy as jnp
import jax.random as jr

def fun(x):
    return x.dot(x)

def approx(f, xo):
    df = jax.grad(f)
    hf = jax.jacobian(df)
    def Q(x):
        return f(xo) + df(xo)(x-xo) +  0.5* hf(xo).dot(x-xo).dot(x-xo) 
    return Q

def diff(x,xo):
    return approx(f,xo)(x) - f(x)

d = 3
xo, u = jr.normal(jr.key(0), (2,d))
scales = jnp.logspace(-1,-5,5) # 1e-1, 1e-2, ... ,1e-5
xs = xo + u * scales[:,None] # broadcast to (d,5)
errors = jax.vmap(diff, in_axes=(0,None))(xs, xo) # (5,) one for each scale

print(errors)
```





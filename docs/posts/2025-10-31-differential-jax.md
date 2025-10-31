---
date: 
    created: 2025-10-31
authors: [xy]
categories: [Analysis]
tags: [low latency programming]
---

# Jax as a differential calculus tool
<!-- more -->
Back in January, I wrote a [tutorial](2025-01-08-jax.md) about Jax, highlighting its power in high performance computing and its near-mathematical syntax.
Now I show how to use Jax as a differential calculus tool for students and educators. 

The goal is to approximate a differentiable function by a few terms in its Taylor expansion near a fixed point. The neat mathematical statement is 

$$
f(x) - f(x_0) = \nabla f(x_0)(x-x_0) +  \frac{1}{2}\langle Hf(x_0) (x-x_0), (x-x_0)\rangle + O(\|x-x_0\|^3)
$$

where $f:\mathbb{R}^d \to \mathbb{R}$ is sufficiently differentiable,  $\nabla f$ is the gradient of $f$ and $Hf$ is its Hessian matrix. 

Based on the expansion, we can use the linear term (first on the right) or the quadratic form (first two terms on the right) to approximate $f$. 
We can go further down the expansion too. In the code below we go down the quadratic route.  

The key steps of the experiment

1. define Ackley function
2. define quadratic approximation of a generic function around an arbitrary point
3. compute the diff between a function with its quadratic approximation
4. evaluate the diff over a few scales.  

```py exec="on" result="text" source="above"
import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)

def f(x):
    """Ackley function"""
    a = 20.0
    b = 0.2
    c = 2 * jnp.pi
    d = x.size

    sum_sq_term = -b * jnp.sqrt(jnp.sum(x**2) / d)
    sum_cos_term = jnp.sum(jnp.cos(c * x)) / d

    term1 = -a * jnp.exp(sum_sq_term)
    term2 = -jnp.exp(sum_cos_term)

    return term1 + term2 + a + jnp.exp(1.0)

def approx(f, xo):
    df = jax.grad(f)
    hf = jax.jacobian(df)
    def Q(x):
        return f(xo) + df(xo).dot(x-xo) +  0.5* hf(xo).dot(x-xo).dot(x-xo) 
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

We observe that, roughly speaking, as we reduce the distance from x to xo by 10x we see an improvement of the approximation by 1000x, which 
confirms the cubic term in Taylor expansion described above. 

---

From the example above you see how easy it is to use Jax for showcasing differential caculus results. We can imagine its use in solving differential equations or
designing optimization algorithm. Check out jax based projects `diffrax` and `optax` for those use case.




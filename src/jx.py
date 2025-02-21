# --8<-- [start:array]
import jax.numpy as jnp

x = jnp.array([1, 2, 3])
y = x.at[1:].set(0)
print(y)

z = x.at[1:].add(1)
print(z)
# --8<-- [end:array]

# --8<-- [start:random]
from jax import random, vmap, numpy as jnp

k = random.key(42)
k, k1 = random.split(k)
out1 = random.normal(k1, (3,))

k, k2 = random.split(k)
dfs = jnp.array([1, 2]) # degree of freedom of two t-distributions
out2 = vmap(random.t, in_axes=(None, 0))(k2, dfs)

print(out1)
print(out2)
# --8<-- [end:random]

# --8<-- [start:vmap]
from jax import vmap, numpy as jnp

def f(x, y): return x + y

xs = jnp.array([0, 1, 2, 3])
y = jnp.array([4, 5])
out = vmap(f, in_axes=(0, None), out_axes=1)(xs, y)
print(out)
# --8<-- [end:vmap]


# --8<-- [start:jit_static]
from jax import jit
from functools import partial

@partial(jit, static_argnums=1)
def g(x, n):
    i = 0
    while i < n: i += 1
    return x + i

print(g(1, 5))
# --8<-- [end:jit_static]

# --8<-- [start:jit_while]
from jax import jit
from jax.lax import while_loop

def cond_fun(val):
    i, n = val
    return i < n

def body_fun(val):
    i, n = val
    return i + 1, n

@jit
def g(x, n):
    end, _ = while_loop(cond_fun, body_fun, (0, n))
    return x + end

print(g(1, 5))
# --8<-- [end:jit_while]

# --8<-- [start:jit-dynamic-shape]
# NOT WORKING!
from jax import jit

@jit
def f(x):
  if x > 0: return x
  else: return jnp.stack([x,x])

try: f(3)
except Exception as e: print(e)
# --8<-- [end:jit-dynamic-shape]


# --8<-- [start:jit-dynamic-bound]
# NOT WORKING!
from jax import jit

@jit
def f(x):
  if x > 0: return x
  else: return jnp.stack([x,x])

try: f(3)
except Exception as e: print(e)
# --8<-- [end:jit-dynamic-bound]

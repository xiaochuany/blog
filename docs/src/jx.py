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
nk, *subkeys = random.split(k, num=3)
out = vmap(random.normal)(jnp.array(subkeys))

nnk, *subkeys = random.split(nk, num=3)
dfs = jnp.array([1, 2])
out2 = vmap(random.t)(jnp.array(subkeys), dfs)

print(out)
print(out2)
# --8<-- [end:random]

# --8<-- [start:vmap]
from jax import vmap, numpy as jnp


def f(x, y):
    return x + y


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
  while i < n:
    i += 1
  return x + i

print(g(1,5))
# --8<-- [end:jit_static]

# --8<-- [start:jit_while]
from jax import jit
from jax.lax import while_loop

def cond_fun(val):
  i,n = val
  return i < n

def body_fun(val):
  i,n = val
  return i+1, n

@jit
def g(x, n):
  end, _  = while_loop(cond_fun, body_fun, (0,n))
  return x + end

print(g(1,5))
# --8<-- [end:jit_while]

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

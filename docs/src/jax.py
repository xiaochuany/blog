# --8<-- [start:random]
from jax import random

key = random.PRNGKey(0)
k1,k2 = random.split(key)
out = random.normal(k2,(3,))

print(out)
print(k1, key)
print(type(key))
# --8<-- [end:random]


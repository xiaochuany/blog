---
date: 
    created: 2025-12-02
authors: [xy]
categories: [Analysis]
tags: [low latency programming, quant methods]
---

# Scalable bootstrapping
<!-- more -->
It is frustrating to see the out of memory error. It is equally frurstating to wait hours for the computation to finish.  

Sometimes these frustrations are inevitable, sometimes they are artificial. The goal of this post is to identify some cases 
where we can avoid the frustration, that is, run large scale bootstrap efficiently (no OOM and fast).    

Consider this: we resample with replacement from a set of size one million to form a set of the same size. We replicate this 100k times. 
If each invididual element takes 4 bytes (say Int32), we need 4e12 bytes, approximately 400GB to store all data. 
This is obviously larger than the available RAM on a consumer hardware. Holding all these data in memory is not an option. 

However, it is rarely the case that we need to hold all these data. The purpose of bootstrap is to estimate the variability of 
a handful of quantities across all replications (100k in our example). A reduction step would reduce that memory consumption for a 
single replication to O(1). If we execute these reductions sequentially, we can avoid OOM. 

This is exactly what we do below. An important point here is that  `jax.lax.map` avoids the python native for loop, making the seqeuntial execution more efficient. This runs almost instantly after the first compilation. 

```py
import jax
import jax.numpy as jnp
import jax.random as jr

key = jr.key(0)
seq = jr.choice(key, 2, (5000000,), p=jnp.array([95,5]))

def resample(key, seq):
    return jr.choice(key, seq, shape=(len(seq),)).mean() # reduction op is the mean

@jax.jit
def resample_all_sequential(keys, seq):
    return jax.lax.map(lambda k: resample(k, seq), keys)

resample_all_sequential(jr.split(key, 1000000), seq).shape 
```



## Side note: why not vmap? 

In situation like bootstraping it is natural to use vmap and feed a large number of keys into the vmapped functions. This will not work
because under the hood, JAX will materialize the whole data matrix even if we have a reduction at the end of each replication, which, as we mentioned before, is way too large. 

However, it is possible to take a hybrid approach. use vmap to take 100 keys at a time, then splash a sequential map on top. I don't feel much of a difference for the scale I try. 

```py
@jax.jit
def resample_batched(keys, seq):
    keys_reshaped = keys.reshape(-1, 100)
    return jax.lax.map(
        jax.vmap(lambda k: resample(k, seq)), keys_reshaped).flatten()
```

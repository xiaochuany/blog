---
date: 
    created: 2025-11-20
authors: [xy]
categories: [Tutorial]
tags: [dev tools, low latency programming, data engineering]
---

# High performance data loading with grain

<!-- more -->

With the advent of Gemini 3 Pro and Nano Banana Pro, Google is clearly winning. 

At the same time, developper experience is getting better for people using Google's stack. 
JAX is getting better at utilizing GPU; `flax.nnx` lets people write neural nets with similar syntax to pytorch.
It might be a good time to keep up with this successful stack - from TPU, JAX, all the way to Gemini. 

Today I am going to talk about grain, a data loading library that works best with JAX.


## Caveat on the import 

At the time of writing, the version of grain is 0.2.14. We import like so:

```py
import grain
```

Part of the official documentation uses a different path: `import grain.python as grain`. 
Digging into the codebase shows that the latter exists for backwards compatibility. 


## DataLoader API is verbose

Grain provides two APIs: the lower level functional API and the higher level imperative API. 

To use the imperative DataLoader API, you collect the data source, the transformations and the sampler to instantiate the `grain.DataLoader` class which returns an iterator that can be used in for loops.   

```py
dataloader = grain.DataLoader(
    data_source = data_source,
    transformations = transformations,
    sampler = sampler
)
```

There are more to it, e.g. multiprocessing, prefetching, sharding etc but this is the gist of it. 
I am not a fan of this because it is verbose and does not support data mixing.

## Dataset API is cooler

To use the functional API, you basically point to the data source and start chaining a bunch of transformations to it. Lazy evaluation is used to avoid unnecessary computation. There are two main classes: `MapDataset` and `IterDataset`. Think of `MapDataset` as a sequence of elements because it supports indexing and len(), while `IterDataset` is an iterable of elements because it only supports iteration. 

Both are equipped with a bunch of methods to transform the data. e.g. 

- `map`: apply a function to each element
- `random_map`: apply a function to each element with random arguments
- `filter`: filter out elements that do not satisfy a condition
- `batch`: batch elements
- `seed`: set the random seed
- `mix`: mix elements from multiple datasets
- ...

Methods that `MapDataset` have but `IterDataset` does NOT have are 

- `source`: create a dataset from a data source e.g. file, in memoery sequence, etc
- `range`: create a range of elements
- `repeat`: repeat the dataset
- `shuffle`: this is global shuffle
- `to_iter_dataset`: convert to `IterDataset`
- ...

Methods that `IterDataset` have but `MapDataset` does NOT have are 

- `mp_prefetch`: prefetch the dataset

In other words, the starting point has to be `MapDataset` and shuffling has to be done at the `MapDataset` level, while multiprocessing has to be done at the `IterDataset` level. To specify the multiprocessing options you need to cross the boundary from `MapDataset` to `IterDataset` with `to_iter_dataset` and then use `mp_prefetch`. Here is an example:

```py
dataset = grain.MapDataset.range(100)
    .shuffle(0)
    .map(lambda elem:elem) # no op
    .to_iter_dataset()
    .batch(2, drop_remainder=True)
    .mp_prefetch(
        grain.multiprocessing.MultiprocessingOptions(
            num_workers=4
        )        
    )
```


## Taking advange of TPUs/GPUs 

Google is generous enough to provide free instances of TPU and GPU on kaggle kernels. TPU has 224 CPU cores and 
8 local TPU devices. GPU has 4 CPU cores and 2 local T4 devices.  
In both cases, there is only one host (single machine).  

```py
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import grain

devices = jax.local_devices() # 8 TPUs or 2 GPUs
mesh = Mesh(devices, axis_names=('data_axis',))

GLOBAL_BATCH_SIZE = 128 * len(devices) 

dataset = (grain.MapDataset.range(10000)
    .map(lambda x: x) 
    .batch(GLOBAL_BATCH_SIZE) # <--- ONE Big Batch, not list of batches
    .to_iter_dataset()
    .mp_prefetch(grain.multiprocessing.MultiprocessingOptions(num_workers=4))
)
iterator = iter(dataset)

# --- DEFINE SHARDING ---
data_sharding = NamedSharding(mesh, PartitionSpec('data_axis'))

# --- JITTED STEP ---
@jax.jit
def train_step(batch):
    return jnp.sum(batch) * 2

# --- TRAINING LOOP ---
for i in range(5):
    host_batch = next(iterator)
    
    device_batch = jax.device_put(host_batch, data_sharding)
    loss = train_step(device_batch)
    
    print(f"Step {i}, Loss: {loss}, Device_batch_size:{device_batch.shape}, host_batch {host_batch.shape}") # same shapes = GLOBAL_BATCH_SIZE
    for i, shard in enumerate(device_batch.addressable_shards):
        print(f"Device {i} Physical Shape: {shard.data.shape}") # see how each device has a shape 128
```


!!! reference

    https://developers.googleblog.com/en/building-high-performance-data-pipelines-with-grain-and-arrayrecord/

    https://google-grain.readthedocs.io/en/latest/grain.dataset.html#grain.MapDataset

    https://docs.cloud.google.com/tpu/docs/jax-ai-stack
---
date: 
    created: 2025-11-18
authors: [xy]
categories: [Tutorial]
tags: [low latency programming]
draft: true
---

# A guide to convolution parameters in neural nets

<!-- more -->

## Original idea

Given a signal $f$ and a kernel $g$, say a probability density/mass function, the convolution $f*g$ at point $x$ measures 
an "average" of $f$ around the point $x$, mathematically defined as 

$$
f*g = \int f(x-y) g(y) \mu(dy)
$$

Here $\mu$ is some reference measure on the ambient space, say Lebesgue or counting measure on Euclidean space. Because of the averaging, convolution typically improves the smoothness of the signal, provided that $g$ is smooth in some way. 

In deep learning, $g$ is a learnable whereas in mathematics/engineering, the kernel is prescribed by the user. 

Different convolutions would invovle translations and scalings of the signal and the kernel.     

## Concepts that provides a granular control  

There are quite a few knobs at our disposal.

- Stride is about how kernels slide.  `stride=k` means you would slide kernel `k` position at a time when you do the sum of product.
- Padding is about adding zeros to the boundary of the input. 
- Input dilation is about adding zeros in between positions of the input, thereby inflating the input size by a factor. Input dilation is relevant when you up-sample a feature map.   
- Kernel dilation is about adding zeros in between positions of the kenrel, thereby expanding the reach of a kernel by a factor, while keeping the number of parameters unchanged. 

In the rest of the post, we see how to turn these knobs in `flax`, a neural network library built on top of `jax`.    

## Simplest setup 

```py
from flax import nnx

rngs = nnx.Rngs(0)
layer = nnx.Conv(3, 4, (5,5), rngs=rngs)
```

We need to feed three positional arguments: the input channel, output channel, and the kernel size. 
The length of the kernel size matches the number of spatial dimension of the convolution. 
Here we use the word spatial in a loose sense. It can be 1d, 2d or more. 
For instance, if each channel of the input represents the system's state at a timestamp and a 
3d coordinate, the number of spatial dimension is 4 and so should provide a list of 4 integers for the kernel size.  

Here we don't see the knobs because they take a sensible default.  

```
import jax.numpy as jnp
layer(jnp.ones((32,32, 3))).shape # (32,32,4) no batch dimension
layer(jnp.ones((1,32,32, 3))).shape # (1,32,32,4)
layer(jnp.ones((1,1,32,32, 3))).shape # (1,1,32,32,4)
```

The output shape tells a few things:

-  it keeps the batch dimension as is. 
-  default padding is "SAME", refering to keep spatial dimensions the same.
-  the last axis of input is the channel dimension, whose value should match the first parameter of the `nnx.Conv`.

## Padding patterns

When we slide the kernel close to  the boundary, there is not enough token from the input to match the kernel length.
Setting `padding="valid"` respect this. The output spatial dimension is k-1 less where k is the kernel size. 


```py
layer = nnx.Conv(3,4,5, rngs=rngs, padding = "valid")
layer(jnp.ones((128, 100, 3))).shape  # (128, 96, 4)
```

We can provide padding parameter with a list, where each item of the list is either an integer (pad both sides the same way) 
or a pair of integers (asymmetric). 

```py
layer = nnx.Conv(3,4,5, rngs=rngs, padding = [(4,0)])
layer(jnp.ones((100, 3))).shape  # (100, 4)
```

The level of control allows to implement the so-called causal convolution where the ouput at time i depends only on 
the time i or before of the input sequence. 

There are other string choices of padding:  CIRCULAR identifies opposing boundaries the same way a torus is defined. REFLECT is like putting a mirror on the boudary so the value nearby is symmetric as is seen by the kernel. 

## Kernel dilation 

```py
layer = nnx.Conv(5,6, kernel_size= 7, rngs=rngs, padding="valid", kernel_dilation = 2) 

layer(jnp.ones((224,5))).shape    # (212,6)
```

To understand kernel dilation, some basic arithmetics are in place. Let's say dilation factor is d, and kernel size is k. 
Then the dilation adds $(k-1)(d-1)$ zeros. The size of the dilated kernel is therefore $k+(k-1)(d-1)$ and the 
the output spatial dimension reduces by $k+(k-1)(d-1)-1 = (k-1)*d$. 

## Input dilation 

```py
layer = nnx.Conv(5,6, kernel_size= 7, rngs=rngs, 
                input_dilation = 2,
                padding = [(0,0)]  # str padding patterns not supported if input_dilation > 1
                )

layer(jnp.ones((224,5))).shape   # (441, 6)
```

Here dilation adds $(T-1)(d-1)$ zeros to the input sequence. The length of the dilated input is $T+(T-1)(d-1)$ + pad. 
The output spatial shape is T+(T-1)*(d-1) + pad - k + 1. 



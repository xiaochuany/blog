---
date: 
    created: 2025-09-07
authors: [xy]
categories: [TIL]
tags: [machine learning]
draft: true
---

# Repurpose hierarchical clustering 

A while ago, I wrote about [hierarchical clustering](2025-07-15-hierarchical.md) with a focus on clean implementation of this classic method. 

In this short post, my goal is to repurpose it for consolidation of categorical features based on hypothesis testing. By consolidation I mean merge together some  values of a categorical variable which behave similarly with regard to the target variable (think default status in credit risk modelling) in some statistical sense. 

The idea is simple: when we merge clusters in the bottom up fashion a la hierarchical clustering, we could use p-value as a "distance". Here distince is in quote because p-value should actually be viewed as a similarity measure: lower p-value would reject the null that two clusters have the same distribution, therefore suggesting a large distributional distance. On the other hand, higher p-value would suggest a low distributional distance. 

The most similar clusters would have the highest p-value, which we merge first. The clusering stops once p-values for all pairs of current clusters is lower than a pre-set threshold, say 0.05, indicating pairwise heteogeneity.

Implementation is $\varepsilon$-away from the original one. 

```py
import heapq
import numpy as np
from scipy.stats import ks_2samp 

def ks_test(x, y):
    """ks_2samp expects 1d arrays. OK if X is 1d to begin with.
    we can implement a customized test depending on needs.  
    """
    return ks_2samp(x, y).pvalue

def test_clustering(X, test_func=ks_test, alpha=0.05):
    clusters = {i: [i] for i in range(len(X))}
    heap = []
    for i in clusters:
        for j in clusters:
            if i < j:
                p = test_func(clusters[i], clusters[j])
                heapq.heappush(heap, (-p, i, j))
    while heap:
        negp, i, j = heapq.heappop(heap)
        p = -negp
        if i not in clusters or j not in clusters:
            continue
        if p <= alpha:
            break
        clusters[i].extend(clusters[j])
        del clusters[j]
        for k in clusters:
            if k != i:
                p = test_func(X[clusters[i]], X[clusters[k]])
                heapq.heappush(heap, (-p, i, k))
    return list(clusters.values())
```


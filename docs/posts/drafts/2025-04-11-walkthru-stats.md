---
date:
    created: 2025-04-11
draft: true
authors:
    - xy
categories: 
    - Note
tags:
    - industrial-mathematics
---

# Walk through `scipy.stats`: part 1 one-sample tests 

`scipy.stats` is a fundamental module for folks using statistics in industry. In this series, we walk through this module with a focus on hypothesis testing.
In particular, we care about the expected inputs for the correct usage of these tests and the underlying null hypothesis & distribution.

We distinguish the tests according to the input dtypes/shapes. For instance, the simplest case is when the input is a 1D array or 0D scalar, which is the topic of this post. 


## input ndim = 0

First, let's look at an example where the input data is a scalar. 

> `binomtest(k, n, p=0.5, alternative = "two-sided")`

Here the input data is the number of occurrence of some events `k`, and the null hypothesis/distribution is that the value is binomial distributed with params `(n, p)`.
The test is two-sided by default but can be made `less` or `greater` than `p` (a general pattern that arise often when the null hypothesis is concerned with a ordinal parameter). 


| Header 1      | Header 2      |
|---------------|---------------|
| data          | k             |
| null distribution | binom(n,p)  |
| null hypothesis | p=0.5 (or any prescribed value) |








## input ndim=1

There are many examples where the input data is 1DArray. 

#### chisquare

> `chisquare(f_obs, f_exp=None, ddof=0, axis=0, *, sum_check=True)`

where f_obs is 1DArrayLike[int], representing occurrence in each category, and the null hypothesis is the observed values are the outcome of a multinomial distribution with probabilisties stored in f_exp of type 1DArrayLike[float].   

TODO: axis

#### t test

> `ttest_1samp(a, popmean, axis=0, nan_policy='propagate', alternative='two-sided', *, keepdims=False)`

This tests the null hypothesis that the mean of the distribution underlying 1DArray `a` is equal to `popmean`. 

TODO: nan_policy

#### shapiro

> `shapiro(x, *, axis=None, nan_policy='propagate', keepdims=False)`

Null hypothesis `x` is sampled from normal distribution. 


- shapiro:  ndim=1, dist = normality 
- anderson: ndim=1, distributional equality
- cramervonmises: ndim=1, distribuional equality
- ks_1samp: ndim=1, distribuional equality
- wilcoxon: ndim =1 (paired sample i.e. difference of two covariate), distributional equatlity
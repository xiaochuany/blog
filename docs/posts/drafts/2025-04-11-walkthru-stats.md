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


| Function (`scipy.stats.`) | Test Statistic (Conceptual Basis)                                                                                                 | Null Hypothesis (H₀)                                                                | Distribution of Statistic under H₀                                                                                                                                                              |
| :------------------------ | :-------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `binomtest`               | The observed number of successes, `k`.                                                                                            | The true probability of success in a single trial is equal to `p`. (P = p)          | The number of successes (`k`) follows a **Binomial distribution**: `Binomial(n, p)` <br> *(where `n` is the total number of trials and `p` is the hypothesized probability of success)* |
| `poisson_means_test`      | (Using the default/conditional 'score' method): The number of events in the first sample, `k1`, *conditional* on the total events `K = k1 + k2`. | The true underlying rates (means per unit exposure) of the two Poisson processes are equal. (μ₁ = μ₂) | Conditional on the total number of events `K = k1 + k2`, the count `k1` follows a **Binomial distribution**: `Binomial(K, p_cond)` <br> *(where `K = k1 + k2` and `p_cond = n1 / (n1 + n2)`) * |


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

---

binomtest(k,n,p, alternative)

- stat:  k (event counts)
- h0: k is binomial distributed with params n, p

---

ttest_1samp(a, popmean)

1. stat: a.mean()/a.std()  
1. h0: a is sampled from a distribution with mean popmean  
1. D(stat) under h0: t-distributed with degree of freedom a.size - 1

---
quantile_test(x,q,p,alternative)

1. stat: np.sum(x<=q) or np.sum(x<q)
1. h0: x is sampled from distribution with p-quantile = q
1. D(stat|h0): binom(x.size, p)

---
shapiro(x)

1. stat: $$W = \frac{\left( \sum_{i=1}^{n} a_i x_{(i)} \right)^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$
where $x_{(i)}$ is ith order statistic and $a_i = E[x_{(i)}]$ under null iid normality assumption. 
1. h0: x is iid normal
1. D(W|h0): exact distribution unknown. simulate W many times to get an empirical distirbution as proxy. 

TODO. normal approximation may apply after some transformation of W. 

---
monte_carlo_test(data, rvs, statistic,n_resamples)

- rvs: callable that takes size as argument. distribution of data under h0.
- statistic: callable 

returns the statistic, pvalue calculated with empirical distribution obtained by the simulation. 

---
ks_1samp(x,cdf)

1. stat: Kol distance between empirical cdf of x (see below) and cdf
1. h0: x sampled from cdf
1. D(stat|h0): small sample exact distribution by combinatorial arguments,  large sample maximum of brownian bridge over [0,1], aka Kolmogorov distribution.  

```py
def get_ecdf(x):
    def ecdf(v): return np.mean(x<=v)
    return ecdf
```
TODO: empirical process, Donsker etc.  

--- 
chisquare(f_obs, f_exp, ddof)

1. stat:  np.sum((f_obs - f_exp)**2/ f_exp)
1. h0: f_obs is multinomial with probabilities f_exp/f_obs.size
1. D(stat|H0): asymptotic chi2 of degree of freedom f_obs.size -1 - ddof

note: ddof is the number of parameters estimated on the raw data (bin them to get the frequencies) before passing the frequency data to the test. 

--- 
wilcoxon(x,y)

1. stat: see below
1. h0: median of X-Y is 0
1. D(stat|h0): asymptotically normal whose params depend on N, the second returned value in get_stat.

TODO: mean and variance

```py
from scipy import stats
def get_stat(x,y):
    d = x - y
    d = d[d!=0]
    ranks = stats.rankdata(np.abs(d), method="avarage") # break tie with average
    wp = np.sum(ranks[d>0])
    wm = np.sum(ranks[d<0]) # exclusive since d=0 pairs are removed. 
    return np.min([wp,wm]), d.size
```


---
permutation_test(data: Array, statistic: Callable, permutation_type)

a flexible test, a lot to cover...



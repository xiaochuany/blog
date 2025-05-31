---
date:
    created: 2025-04-16
authors:
    - xy
categories: 
    - Note
tags:
    - industrial mathematics
draft: true
---

# Walk through `scipy.stats`: part 1 one-sample tests 
<!-- more -->

`scipy.stats` is a fundamental module for applied statistical work. In this series, we walk through this module with a focus on hypothesis testing.
In particular, we care about what the tests expect as data inputs and the underlying null hypothesis & distribution.
We pay attention to the input dtypes/shapes. For instance, the simplest case is when the input is a 1D array or 0D scalar, the topic of this post.  

## Simple null (asymptotic) distributions


|test | data | statistic | H0 | D(stat\|H0) |
|-|-|-|-|-|
|`binomtest(k,n[,p,alternative])` |k| k | k ~ Bin(n,p) |  Bin(n,p) |
| `ttest_1samp(a,popmean)`| a | a.mean()/a.std() | true mean is popmean | t(dof = a.size-1) |
| `quantile_test(x, *[, q, p, alternative])` | x | np.sum(x<q) | p-quantile is q |  Bin(x.size, p) |
| `chisquare(f_obs[, f_exp, ddof])` | f_obs | np.sum((f_obs - f_exp)**2/ f_exp) | f_obs is multinomial with probabilities f_exp/f_obs.size | chi2(dof= f_obs.size -1 - ddof)|


Note: for chisquare test, ddof is the number of parameters estimated on the raw data (bin them to get the frequencies) before passing the frequency data to the test. 

## Non canonical null distributions

> `shapiro(x)`

1. statistic: $$W = \frac{\left( \sum_{i=1}^{n} a_i x_{(i)} \right)^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$
where $x_{(i)}$ is ith order statistic and $a_i = E[x_{(i)}]$ under null iid normality assumption. 
1. h0: x is iid normal
1. D(W|h0): exact distribution unknown. simulate W many times to get an empirical distirbution as proxy. 

TODO. normal approximation may apply after some transformation of W. 

> `ks_1samp(x,cdf)`

1. statistic: Kol distance between empirical cdf of x (see below) and cdf
1. h0: x sampled from cdf
1. D(stat|h0): small sample exact distribution by combinatorial arguments,  large sample maximum of brownian bridge over [0,1], aka Kolmogorov distribution.  

```py
def get_ecdf(x):
    def ecdf(v): return np.mean(x<=v)
    return ecdf
```
TODO: empirical process, Donsker etc.  

> `wilcoxon(x,y)`

Signed test.

1. statistic: see below
1. h0: median of X-Y is 0
1. D(stat|h0): asymptotically normal whose params depend on N, the second returned value in get_stat.

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

TODO: mean and variance


## Association tests

> `spearmanr`

1. statistic: 

## Approximate null distribution by resampling

> `monte_carlo_test(data, rvs, statistic: Callable, n_resamples)`

Note: rvs is callable that takes size as argument. distribution of data under h0.

returns the statistic, with the pvalue calculated from empirical distribution obtained by the simulation. 


> `bootstrap(data, statistic: Callable)` 


> `permutation_test(data: tuple[Array], statistic: Callable, permutation_type)`

permutation_type| action | suitable H0 
|-|-| - |
`parings`| permute all obs in one sample | no association/correlation (rho, tau, r = 0)
`samples`| permute sample membership (pairing unchanged) | T[f(X,Y)] = T[f(Y,X)]
`independent`| permute pooled sample (default) |  



a flexible test, a lot to cover...



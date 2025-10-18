---
date:
    created: 2025-10-17
draft: true
authors: [xy]
categories: [TIL]
tags: [quant methods]
---

# F tests

F distribution is the ratio of two independent random variables. Each comes from a chisquare distribution
scaled by its own degree of freedom so that its expected value is one. If the first chisquare has $d$ degrees of freedom and the second $e$, the ratio 

$$
F = \frac{\chi^2_{d}/d}{\chi^2_{e}/e}
$$

defines the F distribution. 

We can write $F = F(d,e)$ to indicate that F is actually a parametric family of distributions. 

Its density function can be spelled out with Gamma functions and such, but that's not what 
we care about in this post. Our goal is to explore test statistics equal or approaching F distribution as the sample size grows. [TODO: cover large sample case in more detail]

## Equality of variance

As we've discussed before, when the data is iid Gaussian, sample variance is distributed exactly as 
scaled chisquare. So, a natural use of the F distribution is to test equality of variance of two groups. 

Given two iid samples $\{X_i\}$ and $\{Y_i\}$ with equal variance, our null hypothesis, the sample variance 

$$
\frac{1}{n-1} \sum_{i=1}^n (X_i - \bar X)^2 
$$

has the mean value equal to the population variance $\sigma^2$. Similarly for $Y_i$. 

By taking the ratio of two sample variances, the unknown true variance cancels out. 
If the data distribution were Gaussian, this ratio follows the F distribution, which we  
use to find the p-value.  

[TODO: cover non gaussian data]

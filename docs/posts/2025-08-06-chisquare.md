---
date: 
    created: 2025-08-06
authors: [xy]
categories: [TIL]
tags: [statistics]
---

# Unpacking k-1 in chisquare test 

## Setting 

Let $\mathbf{X}=(X_1,...,X_k)$ be multinomial distribution with $k$ categories, total $n$ and probabilities $\mathbf{p}=(p_1,...,p_k)$. Pearson's chisquare test statistic is 

$$
\chi^2 = \sum_{i=1}^k \frac{(X_i - np_i)^2}{np_i}.
$$

The asymptotic distribution is chisquare with degree of freedom $k-1$ as $n$ grows to infinity. 

The aim of this post is to explain the rationale behind $k-1$. 

This post can be read independently from my previous one about [t-test](2025-07-23-ttest.md), but a core part of the argument has already been presented there, so it could be beneficial to read that post first. 

The core ideas are as follows:

- represent multinomial random vector as iid sum (normal approximation applies)
- chisquare statistic is then close to the squared norm of Gaussian vector in k-1 dimensional subspace 
- spectral decomposition shows that the latter can be written as the sum of $k-1$ independent standard Gaussians.

The last step is a slight generalization of the linear algebra tricks we pull off in the post about t-test.  

## Multivarite central limit theorem

Let $E=\{e_1,...,e_k\}$ be the canonical basis of $R^k$ i.e. $e_i$ has entries 0 everywhere except at the $i$-th coordinate, where the value is 1. 

Let $I_i$ denote a random vector taking values in $E$ with 

$$
P[I_i = e_i] = p_i 
$$

Then we have 

$$
(X_1,...,X_k) = \sum_i^n I_i. 
$$

The mean of this vector is $n\mathbf{p}$ and its covariance normalized by $n$
$Cov[X_i,X_j]/n$ does not depend on $n$, which we denote by $\Sigma = \Sigma(\mathbf{p})$. 


By the central limit theorem, 

$$
\frac{1}{\sqrt{n}}\Big(\sum_i^n I_i - n \mathbf{p}\Big) \to N(0,\Sigma)
$$


Let $G$ denote the normal vector in the limit. It remains to show that 

$$
\sum_{i=1}^k \frac{G_i^2}{p_i} = \| (G_1/\sqrt{p_1},...,G_k/\sqrt{p_k}) \|^2
$$

is chisquare distributed with degree of freedom k-1. 


## Checking the covariance

We claim 
 $Cov[X_i,X_j]/n = (p_i 1_{i=j} - p_i p_j)$.


It follows that the Gaussian vector $H = (G_1/\sqrt{p_1},...,G_k/\sqrt{p_k})$
has covariance 

$$
V_{ij} = 1_{i=j} - p_j.
$$

We have shown that $\chi^2$ can be approximated by (in fact converges to) $\|H\|^2$. The rest of the argument is very much the same as in my [previous post about t-test](2025-07-23-ttest.md) so I only sketch it. 


## Linear algebra again

We check easily that $V=V^2$ which implies that its eigenvalues are either 0 or 1. Since $H= V Z$ where $Z$ is standard Gaussian in $\mathbb{R}^k$, we can use spectral decomposition to represent the squared norm of $H$ as a sum of independent standard Gaussian squared, where the number of terms is equal to the rank of the $V$. 

We now check the rank of $V$ is equal to $k-1$. 

We notice

$$
\Sigma = diag(mathbf{p}) - \mathbf{p} \mathbf{p}^T 
$$

so if $\Sigma x = 0$, we necessarily have $x_1=...=x_k= \sum_{i}x_i p_i$ or $x_1=...=x_k=0$. The former condition holds when $\sum_i p_i=1$ which is the case. Hence the dimension of the null space of $\Sigma$ is 1, therefore the dimension of the range of $\Sigma$ (which is the rank!) has to be $k-1$. It follows right away that $V$ is of rank $k-1$. CQFD. 
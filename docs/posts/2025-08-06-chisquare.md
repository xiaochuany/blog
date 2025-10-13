---
date: 
    created: 2025-08-06
authors: [xy]
categories: [Analysis]
tags: [quant methods]
---


# Unpacking the k-1 in the chi-square test
<!-- more -->

This post is a continuation of my previous one about [t-test](2025-07-23-ttest.md). The aim, as before, is to spell out all the details about the degrees of freedom in the chi-square test. 

## Setting

Let $\mathbf{X} = (X_1, \dots, X_k)$ be a multinomial random vector with $k$ categories, total count $n$, and probabilities $\mathbf{p} = (p_1, \dots, p_k)$. Pearson’s chi-square test statistic is

$$
\chi^2 = \sum_{i=1}^k \frac{(X_i - n p_i)^2}{n p_i}.
$$

As $n \to \infty$, the distribution of $\chi^2$ converges to a chi-square distribution with $k - 1$ degrees of freedom.

The goal of this post is to explain where the $k - 1$ comes from.

This post can be read independently of my previous one on the [t-test](2025-07-23-ttest.md), but a core part of the argument was already presented there, so reading that first might help.

Core ideas:

- Represent the multinomial vector as a sum of i.i.d. random vectors (so normal approximation applies).
- The chi-square statistic becomes close to the squared norm of a Gaussian vector lying in a $(k - 1)$-dimensional subspace.
- Spectral decomposition shows that this squared norm is distributed as a sum of $k - 1$ independent standard normal squares.

That last part is a generalization of the linear algebra trick we used in the t-test post.


## Multivariate central limit theorem

Let $E = \{e_1, \dots, e_k\}$ be the canonical basis of $\mathbb{R}^k$ where each $e_i$ has a 1 in the $i$-th position and 0 elsewhere.

Let $I_i$ be a random vector taking values in $E$ such that

$$
P[I_i = e_j] = p_j.
$$

Then

$$
(X_1, \dots, X_k) = \sum_{i=1}^n I_i.
$$

The mean of this sum is $n \mathbf{p}$. Its covariance, normalized by $n$, does not depend on $n$, so we denote it by $\Sigma = \Sigma(\mathbf{p})$.

By the multivariate central limit theorem:

$$
\frac{1}{\sqrt{n}}\left( \sum_{i=1}^n I_i - n \mathbf{p} \right) \to \mathcal{N}(0, \Sigma).
$$

Let $G$ denote this limiting Gaussian vector. Then the chi-square statistic is asymptotically approximated by

$$
\sum_{i=1}^k \frac{G_i^2}{p_i} = \left\| \left( \frac{G_1}{\sqrt{p_1}}, \dots, \frac{G_k}{\sqrt{p_k}} \right) \right\|^2.
$$

We now want to show that this norm squared follows a chi-square distribution with $k - 1$ degrees of freedom.


## Checking the covariance

We claim:

$$
\text{Cov}[X_i, X_j]/n = p_i \delta_{ij} - p_i p_j.
$$

Hence, the Gaussian vector $H = (G_1 / \sqrt{p_1}, \dots, G_k / \sqrt{p_k})$ has covariance matrix

$$
V_{ij} = \delta_{ij} - p_j.
$$

So we’ve shown that $\chi^2$ converges in distribution to $\|H\|^2$. The rest of the argument is nearly identical to the [t-test post](2025-07-23-ttest.md), so I'll just sketch it here.

## Linear algebra again

Note that $V = V^2$, so it is a projection matrix. Its eigenvalues are either 0 or 1.

We can write $H = VZ$, where $Z$ is a standard Gaussian in $\mathbb{R}^k$. Then $\|H\|^2 = \|VZ\|^2$ is the sum of squares of the projections of $Z$ onto the image of $V$.

Spectral decomposition lets us write this as a sum of $r$ independent standard normal squares, where $r = \text{rank}(V)$.

We now check that $\text{rank}(V) = k - 1$.

Recall:

$$
\Sigma = \text{diag}(\mathbf{p}) - \mathbf{p} \mathbf{p}^\top.
$$

So if $\Sigma x = 0$, then necessarily $x_1 = x_2 = \dots = x_k = \sum_i x_i p_i$. That is, the null space consists of constant vectors, hence it's one-dimensional.

Therefore, $\text{rank}(\Sigma) = k - 1$, and since $V$ is derived from $\Sigma$, we also have $\text{rank}(V) = k - 1$.

So $\|H\|^2$ is the sum of $k - 1$ independent standard normal squares, namely, chi-square distributed with $k - 1$ degrees of freedom. CQFD.


## Generalization again

Now consider contingency tables which are pivot tables of bivariate categorical data. We assume iid sequence $\{(X_i,Y_i), i=1,...,n \}$. Both are categorical with $a$ and $b$ categories respectively. The data can be stored as a table of shape (n,2). Pivoting on the second column (Y's) with the first column (X's) as index, we get frequencies of each one of the $ab$ combinations. The pivot table is of shape (a,b). In dataframe land, this is done as follows:

```py
import polars as pl
import numpy as np

x = pl.from_numpy(np.random.randint(0,2,size=(10,2)))
x.pivot(on="column_1",index="column_0",values="column_0",aggregate_function="len")
```

The cells in the pivot table sum to n.

It is again possible to express the frequencies as sum of $n$ iid random matrices $A_1, \dots, A_n$. Each is of shape (a,b) matrix. If $(X_1,Y_1)=(k,l)$ then $A_{1,kl}=1$ and 0 elsewhere. Again we have multinomial distribution with parameters $(n, p_{ij})$ with $p_{ij}=P[X_1=i, Y_1=j]$. 

Except from a fancier way of indexing n values (using 2d array), the situation is exactly the same as before, so that when n get larger, we can approximate the statistic

$$
C = \sum_{1\le i\le a, 1\le j\le b} \frac{(X_{ij}-n p_{ij})^2}{n p_{ij}}
$$

by a chisquare with $ab-1$ degrees of freedom. 

In fact, a common setup for contingency tables is to assume independence between variables. In such case the matrix $(p_{ij})$ is rank 1 because $p_{ij}=r_i q_j$ where $\mathbf{r} = (r_1, \dots, r_a)$ and $\mathbf{q} = (q_1, \dots, q_b)$ are marginal distributions of the pair $(X_1,Y_1)$.

We claim that $C$ is asymptotically chisquare with degrees of freedom $(a-1)(b-1)$. It suffices to show that the covariance matrix of $(X_{ij})$ (a projection as we have shown already), divided by $n$,  is of rank $(a-1)(b-1)$. We can show this by investigating the null space of the covariance matrix (show it's of dimension $a+b-1$). We leave this as an exercise to the interested reader :grinning:


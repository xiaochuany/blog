---
date:
    created: 2026-02-20
authors: [xy]
categories: [TIL]
tags: [quant methods]
draft: true
---

# Asymptotic Distribution of AR(1) OLS Estimators: MDS and Limit Theorems

<!-- more -->

We consider the first-order autoregressive model without an intercept:
$$X_t = \phi X_{t-1} + \epsilon_t, \quad t=1, \dots, n$$
with the OLS estimator:
$$\hat{\phi}_n = \phi + \frac{\sum_{t=1}^n X_{t-1} \epsilon_t}{\sum_{t=1}^n X_{t-1}^2}$$
We assume $\{\epsilon_t, \mathcal{F}_t\}$ is a Martingale Difference Sequence (MDS), i.e., $E[\epsilon_t | \mathcal{F}_{t-1}] = 0$ almost surely.

---

## 0. Theoretical Prerequisites: The CLTs

The asymptotic behavior depends on whether the regressor $X_{t-1}$ is stationary. We separate the conditions for the discrete-sum CLT and the functional CLT.

### Theorem A: Martingale Central Limit Theorem (MCLT)
Used when the regressor is stationary ($|\phi| < 1$). A widely accepted version is by **Brown (1971)**. For a martingale difference array $\{ \xi_{ni}, \mathcal{F}_{ni} \}$, if:
1.  **Conditional Variance Convergence**: $\sum_{i} E[ \xi_{ni}^2 | \mathcal{F}_{n, i-1} ] \xrightarrow{p} \sigma^2$ as $n \to \infty$.
2.  **Lindeberg Condition**: For all $\epsilon > 0$, $\sum_{i} E[ \xi_{ni}^2 I(|\xi_{ni}| > \epsilon) | \mathcal{F}_{n, i-1} ] \xrightarrow{p} 0$.

### Theorem B: Functional Martingale CLT (FMCLT)
Used for the random walk case ($\phi = 1$). A standard version is by **McLeish (1974)**. Let $W_n(r) = \sum_{t=1}^{\lfloor nr \rfloor} \xi_{nt}$. $W_n(r) \Rightarrow W(r)$ in $D[0,1]$ if:
1.  **Conditional Variance Convergence (Pathwise)**: $\sum_{t=1}^{\lfloor nr \rfloor} E[ \xi_{nt}^2 | \mathcal{F}_{t-1} ] \xrightarrow{p} r \sigma^2$ for all $r \in [0, 1]$.
2.  **Lindeberg Condition**: For all $\epsilon > 0$, $\sum_{t=1}^n E[ \xi_{nt}^2 I(|\xi_{nt}| > \epsilon) ] \to 0$.

---

## 1. The Stationary Case ($|\phi| < 1$)

### Minimal Setting & Assumptions
We require $\{\epsilon_t\}$ to be a stationary ergodic MDS with $E[\epsilon_t^2 | \mathcal{F}_{t-1}] = \sigma^2$ and $E[|\epsilon_t|^r] < \infty$ for some $r > 2$.

### Statement
$$\sqrt{n}(\hat{\phi}_n - \phi) \xrightarrow{d} \mathcal{N}(0, 1 - \phi^2)$$

### Key Proof Steps
1.  **Regressor Stability**: Since $|\phi| < 1$, $\{X_t\}$ is stationary and ergodic. By the **Ergodic Theorem** for MDS, $\frac{1}{n} \sum X_{t-1}^2 \xrightarrow{p} E[X_0^2] = \frac{\sigma^2}{1-\phi^2}$.
2.  **Applying MCLT**: Let $\xi_{nt} = \frac{1}{\sqrt{n}} X_{t-1} \epsilon_t$. 
    - Conditional Variance: $\sum E[ \xi_{nt}^2 | \mathcal{F}_{t-1}] = \sigma^2 (\frac{1}{n} \sum X_{t-1}^2) \xrightarrow{p} \frac{\sigma^4}{1-\phi^2}$.
    - Lindeberg: Satisfied by stationarity and the $2+\delta$ moment condition.
3.  **Conclusion**: $\frac{1}{\sqrt{n}} \sum X_{t-1} \epsilon_t \xrightarrow{d} \mathcal{N}(0, \frac{\sigma^4}{1-\phi^2})$. Dividing by the denominator limit via Slutsky's gives the result.

---

## 2. The Unit Root Case ($\phi = 1$)

### Minimal Setting & Assumptions
Same as above, but with $\phi=1$. We need $X_0$ for initialization (usually $X_0 = 0$).

### Statement
$$n(\hat{\phi}_n - 1) \xrightarrow{d} \frac{\int_0^1 W(r) dW(r)}{\int_0^1 W(r)^2 dr} = \frac{\frac{1}{2}(W(1)^2 - 1)}{\int_0^1 W(r)^2 dr}$$

### Key Proof Steps
1.  **Applying FMCLT**: Let $\xi_{nt} = \frac{1}{\sigma \sqrt{n}} \epsilon_t$. The conditions of Theorem B hold, so $\frac{X_{\lfloor nr \rfloor}}{\sigma \sqrt{n}} \Rightarrow W(r)$.
2.  **Continuous Mapping Theorem**:
    - Denominator: $\frac{1}{n^2 \sigma^2} \sum X_{t-1}^2 \xrightarrow{d} \int_0^1 W(r)^2 dr$.
    - Numerator: $\frac{1}{n \sigma^2} \sum X_{t-1} \epsilon_t \xrightarrow{d} \int_0^1 W(r) dW(r) = \frac{1}{2}(W(1)^2-1)$.
3.  **Super-consistency**: The rate is $n$, and the distribution is the **Dickey-Fuller** distribution (with no intercept).

---

## 3. The Explosive Case ($|\phi| > 1$)

### Minimal Setting & Assumptions
Assume $|\phi| > 1$, $X_0=0$ (for simplicity), and $\epsilon_t \sim IID \, N(0, \sigma^2)$.

### Statement
$$\frac{\phi^n}{\phi^2 - 1} (\hat{\phi}_n - \phi) \xrightarrow{d} \text{Cauchy}(0, 1)$$

### Why the Invariance Principle Fails

In the stationary and unit root cases, the limit distributions are **invariant** to the specific distribution of the errors $\{\epsilon_t\}$ (provided they satisfy MDS conditions). This is because:
- **Stationary Case**: We average $n$ terms, and the CLT "washes away" the non-normalities of individual shocks.
- **Unit Root Case**: We essentially integrate a path, and the FCLT (Brownian Motion) acts as a universal limit for all well-behaved random walks.

**In the explosive case, this averaging effect disappears.** 

The limit of the OLS estimator is the ratio $\frac{Z}{W_\infty}$. Both $Z$ and $W_\infty$ are **infinite series** of the form $\sum_{j=1}^\infty a^j \epsilon_j$. 
1.  **Distributional Memory**: If the shocks $\{\epsilon_t\}$ are not Gaussian, the distribution of the infinite sum $W_\infty$ is determined by the specific characteristic function of $\epsilon$. Unlike the CLT, which forces a Normal limit regardless of the starting distribution, an infinite weighted sum of non-Gaussian variables remains non-Gaussian.
2.  **No "Large Member" Filter**: The OLS denominator $\sum X_{t-1}^2$ is dominated by the very last few terms (the geometric growth). Because the estimator is effectively "looking" at a few large realizations rather than an average of many equal-sized members, the specific tail behavior of the MDS errors $\epsilon_t$ is preserved in the limit.
3.  **Result**: Since the ratio of two non-Gaussian random variables is almost never Cauchy, the White (1958) result is a mathematical curiosity of the Gaussian distribution rather than a universal property. For a general MDS, the limit distribution typically lacks a closed-form density and must be simulated based on the suspected distribution of the innovations.

---

## Summary and Hypothesis Testing

| Regime | Rate | Limit Distribution | Inference Method |
| :--- | :--- | :--- | :--- |
| **Stationary** ($\| \phi \| < 1$) | $\sqrt{n}$ | $\mathcal{N}(0, 1-\phi^2)$ | **Asymptotic Normality**: Use standard $t$-statistics and $z$-tables. |
| **Unit Root** ($\phi = 1$) | $n$ | Dickey-Fuller | **DF Test**: Use simulated critical values. $t$-stat is NOT normal. |
| **Explosive** ($\| \phi \| > 1$) | $\phi^n$ | Non-standard | **Heuristic**: Standard inference is invalid. Often signaled by $R^2 \approx 1$. |

**Discussion on Testing**: 
In the **Stationary** case, the CLT "works" because the regressor is well-behaved and the information matrix (denominator) grows at the standard rate of $n$.

In the **Unit Root** case, we have super-consistency ($n$ rate) for $\hat{\phi}$. However, the numerator $\sum X_{t-1} \epsilon_t$ can be rewritten using the algebraic identity $\sum X_{t-1} \Delta X_t = \frac{1}{2}(X_n^2 - \sum \Delta X_t^2)$. Normalized by $n \sigma^2$, this becomes $\frac{1}{2}[ (\frac{X_n}{\sigma \sqrt{n}})^2 - \frac{1}{n \sigma^2} \sum \epsilon_t^2 ]$, which converges to $\frac{1}{2}(W(1)^2 - 1)$. 

Because both the numerator and denominator converge to random functionals of the same Brownian motion, their ratio results in a non-standard distribution. Testing $H_0: \phi=1$ using standard normal critical values is a mathematical error that results in severe size distortion (usually vastly over-rejecting the null).


1. The Core Obstacle: Dependency
In the classical i.i.d. CLT, we prove convergence by showing: $$E\left[e^{it \sum \xi_j}\right] = \prod E\left[e^{it \xi_j}\right] \to e^{-t^2/2}$$ For a Martingale Difference Sequence (MDS), the first equality fails because $\xi_j$ are dependent. You cannot simply "pull out" the expectation for each term.

2. McLeish’s "Product Lemma" Trick
McLeish’s insight was to replace the exponential $e^{it \sum \xi_j}$ with a specific product that "mimics" the behavior of independent variables. He used the following Taylor-like approximation: $$e^{ix} \approx (1 + ix) e^{-x^2/2}$$ Summing this up, the characteristic function of the sum $S_n = \sum \xi_j$ can be approximated as: $$e^{it S_n} \approx \underbrace{\prod_{j=1}^n (1 + it \xi_j)}{\text{The "Linear" Part}} \times \underbrace{e^{-\frac{t^2}{2} \sum \xi_j^2}}{\text{The "Quadratic" Part}}$$

3. Why This Works for Martingales
This decomposition is powerful because of two properties that play perfectly with the Martingale structure:

The "Linear" Part sums to 1 (in expectation): By the definition of an MDS, $E[\xi_j | \mathcal{F}{j-1}] = 0$. This implies: $$E[1 + it \xi_j | \mathcal{F}{j-1}] = 1$$ Using the iterated law of expectations, the entire product $\prod (1 + it \xi_j)$ has an expected value of exactly 1, regardless of the dependency between the terms. This "magic" property replaces the factorization used in the i.i.d. case.
The "Quadratic" Part captures the Variance: The second part of the product involves $\sum \xi_j^2$. If the Conditional Variance Convergence condition holds ($\sum E[\xi_j^2 | \mathcal{F}_{j-1}] \xrightarrow{p} \sigma^2$), then by a Law of Large Numbers for martingales, the actual sum of squares $\sum \xi_j^2$ also converges to $\sigma^2$. Thus, $e^{-\frac{t^2}{2} \sum \xi_j^2} \xrightarrow{p} e^{-\sigma^2 t^2 / 2}$.
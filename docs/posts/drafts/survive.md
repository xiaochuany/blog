---
draft: true
date:
    created: 2025-02-07
authors:
    - xy
category:
    - Note
tags:
    - nonparametric statistics
    - survival analysis
slug: survive
---

# Cox hazard model 


## Hazard 

The cumulative hazard function for a time-to-event (e.g. death in medical trials) distribution can be defined in the most generality using the Lebesgue-Stieltjes integral, 
which accommodates both continuous and discrete distributions and more. 

For a survival time \( T \) with survival function \( S(t) = P(T > t) \), the cumulative hazard function \( H(t) \) is defined as:

$$
H(t) = \int_0^t \frac{dF(s)}{S(s^-)},
$$

where \( F(s) = P(T \leq s) \) and \( S(s^-) = \lim_{u \uparrow s} S(u) \) is the left-continuous survival function.   
Here the integral is a Lebesgue-Stieltjes integral (TLDR this makes sense because F is monotone thus of finite variation).

**Continuous Distributions**:

If \( T \) is a continuous random variable with density $f$, then \( S(s^-) = S(s) \) and the cumulative hazard reduces to 
     
$$
\begin{align}
H(t) &= \int_0^t h(s) ds \notag\\
h(t) &= \frac{f(t)}{S(t)} \label{e-hazard-func}
\end{align} 
$$

Notice that $f(t)=-S'(t)$ so by $\eqref{e-hazard-func}$ the survival function satisfies the ODE  

$$H'(t) = - S'(t)/S(t)$$

 with boundary values $S(0)=1, S(\infty)=0$. One concludes that \( S(t) = \exp(-H(t)) \). 

**Discrete Distributions**:

If \( T \) is discrete with support $\{ t_1, t_2, \dots \}$, the cumulative hazard is a sum over discrete hazards:

\[
H(t) = \sum_{t_j \leq t} h(t_j), \quad \text{where } h(t_j) =  \frac{F(t_j)-F(t_j^-)}{P[T\ge t_j]} =  P[T = t_j \mid T \geq t_j].
\]

We can and will assume that $t_1<t_2<t_3$ and so on. Any $t\ge t_1$ falls in exactly one interval composed of two consecutive points in the support i.e.  $t_j\le t<t_{j+1}$
for some $j$. For $t\ge t_1$, we have

$$
\begin{align*}
S(t) &= P[T>t] = P[T>t_i, \forall i = 1, ..., j] \\
& = P[T\ge t_1] P[T>t_1|T\ge t_1] P[T>t_2 | T>t_1] ... P[T>t_j|T>t_{j-1}]
\end{align*}
$$

where we used the fact that $P[T\ge t_1]=1$. Notice that $\{T>t_{i-1}\}= \{T\ge t_i\}$, hence 

$$
S(t) = \prod_{t_j \leq t} \left(1 - h(t_j)\right)
$$

Note that \( H(t) \neq -\log S(t) \) here; instead, \( -\log S(t) = \sum_{t_j \leq t} -\log(1 - h(t_j)) \), which differs from \( H(t) \).

**Other cases**:

For mixed distributions with both continuous and discrete components, \( H(t) \) combines integrals over continuous regions and sums over discrete jumps. 
But it can get more interseting, think Cantor's function (devil's staircase), how would you express the hazard differently? 

## Cox model

Original paper 

https://www.medicine.mcgill.ca/epidemiology/hanley/c626/cox_jrssB_1972_hi_res.pdf

Cox proportional hazard model is de facto standard model in survival anlysis. This lecture note is a quick intro to the model definition, estimator of covariate 
coefficients and hypothesis testing procedures.   

https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture28.pdf

metrics for evaluation of models

so called concordance index ipcw, this is a consistent estimator of the C index. this is C index with weights where weights are 
the inverse of KM estimtor of the censoring tail proba squared.  Seen as better than concordance index alone (number of concordant pairs/ total number of pairs).

https://biostats.bepress.com/cgi/viewcontent.cgi?referer=&httpsredir=1&article=1108&context=harvardbiostat

TODO:

- review KM estimator for censoring time
- KM for survival function under cencoring
- define concordance index, consistency.
- define concordance index ipcw,  consistency
- describe procedure to estimate covariate coefficients, consistency
- investigate sksurv. fit some data. visualize.  
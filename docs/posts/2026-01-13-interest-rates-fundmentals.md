---
date:
    created: 2026-01-13
authors: [xy]
categories: [TIL]
tags: [quant methods]
draft: true
---

# Interest rates fundamentals
<!-- more -->

## First principle

There are a gazillion of interests rates quoted in the markets. Not all rates are born equal. 

Some are determined by supply and demand i.e. the price at which sellers agree to sell and buyers agree to buy.
Some are derived/computed from the observed prices. The purpose of this post is to introduce the main obserables and how to derive more rates from them.

## Observed cost of borrowing overnight

ESTR, Euro Short Term Rate
- published by ECB every business day based on transactions of previous day
- volumn weighted trimmed mean (trim out top and bottom 25 percent)
- unsecured

SOFR, Secured Overnight Financing Rate
- published by NY Fed every business day based on transactions of previous day
- volume weighted median
- secured: based on repo transactions collateralized by U.S. Treasury securities

also SONIA (GBP), TONAR (JPY), SARON (SWF) published by  central banks of the corresponding currency. 

## Observed prices

Bonds
- tenor and coupon are observed   

OIS: overnight index swap 
- the float leg is overnight index such as ESTR, SOFR
- the fixed leg is observed per supply/demand

## Discounting factor is calculated

The discount factor is the present value of 1 ccy at time $T$. More broadly, we want to know the value at time $T_0$ of 1 ccy at time $T$.
The general notation is $P(T_0,T)$. From this factor one can derive several curves. We postpone the discussion of curves in later sections. 

This discounting factor is tied to the input prices, hence is specific to the markets we are considering.

### bond market

The inputs are bond prices and the coupon they pay. For one bond with price $B$, with coupon payment $c_i$ at time $T_i$ and principle payment $F$ at $T_n$, the discount factor are unknown factors such that the following equation holds: it says that the present value of all future cashflows is equal to the price, which is fair.

$$
B  =  \sum_{i=1}^n  c_i P(0,T_i) + F P(0,T_n)
$$

More generally, we have a large collection of bonds and therefore a system of equations 

$$
B_k  =  \sum_{i=1}^{n_k} c_{k,i} P(0,T_{k,i}) + F_k P(0,T_{k, n_k})
$$

Bonds that are most liquidly traded widely offer a discrete set of marutiry and interset payment dates e.g. 

- short term (often called bills): 3M 6M 9M 12M, zero coupon
- medium term (often called notes): 2,3,5,7,10 years, semi-annual coupon
- long term: 20, 30Y, semi-annual coupon

Therefore the set of maturities is not huge, neither is set of $T_{k,i}$. This reduces the number of unknowns for the discounting factors to about 22. On the other hand, the number of equations is equal to the number of AAA-bonds in this context, which is larger. Solving for least squares give a collection of the discounting factors (on the observed subset of dates mentioned above).  

Let's vibe code a calculator and visualization tool.

### derivatives

Pricing derivatives involves discounting future cashflow. The discount facor has to be computed from input data that is relevant to the derivative market at hand. 

Take OIS as an example. As said before, both legs are observed: fix leg is the price at which this instrument is being traded and the float leg is publisehd by central banks (computed as the average transaction rate of previous day). OIS with longer maturity (than overnight) has the float leg calculated using the compounded rate of historical overnight index. For example, at time $t$, OIS with maturity 2 days is computed with $(1+r_{t-1})(1+r_t) - 1$, where $r_s$ is the index rate on day $s$. 

To find the discoutning factor, we equate present value of the fix leg to the present value of the float leg and solve for $P(0,T_i)$ 

$$
K(T_n) \sum_{i=1}^n P(0, T_i) = 1 - P(0,T_n)
$$

The left hand is 




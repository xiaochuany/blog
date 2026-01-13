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

OIS
- the float leg is overnight index such as ESTR, SOFR
- the fixed leg is observed per supply/demand

## Derive the discounting factor

The discount factor is the present value of 1 ccy at time $T$. More broadly, we can talk about the value at time $T_0$ of 1 ccy at time $T$.
The general notation is $P(T_0,T)$. From this factor one can derive several curves. We postpone the discussion of curves in later sections. 



This discounting factor is tied to the market  

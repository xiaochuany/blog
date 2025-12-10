---
date: 
    created: 2025-12-10
authors: [xy]
categories: [Analysis]
tags: [quant methods]
draft: true
---

# A tale of convexity in optimal binning 

<!-- more -->

As I was exploring methodologies in credit rating assignment (say in the PD model), 
an elegant approach caught my attention. Here is the formulation: 

> given a collection of continuous scores and coresponding binary status (good or bad), find a partition that maximize the the information value
> aka the symmetrized KL divergence, subject to the constraint that
> - monotonicity (e.g. default rate is increasing wrt score)
> - consecutive parts are statitically different (e.g. z test for binomials results in a p value < 0.05)




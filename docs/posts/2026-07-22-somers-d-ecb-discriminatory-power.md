---
date:
    created: 2026-07-22
authors: [xy]
categories: [Analysis]
tags: [quant methods]
---


# ECB gAUC: discrimination of the model or of the rating system?

<!-- more -->

For LGD and CCF models, the ECB validation reporting template defines gAUC by first discretising estimated and realised values, constructing an ordered contingency table, and then computing Somers' \(D\):

\[
\mathrm{gAUC} = \frac{D+1}{2}.
\]

This is more than just an efficient implementation of a rank statistic.

For models with \(r \leq 20\) grades or pools, the estimated values define the row ordering, while the realised values are discretised using the grade-level estimates as boundaries. The result is an \(r \times (r+1)\) contingency table from which concordant and discordant pairs are counted.

From a supervisory perspective, this makes perfect sense: the metric evaluates the discriminatory power of the **implemented rating system**—its grades, associated estimates, and their agreement with realised outcomes.

## But model validation asks a different question

When validating the **risk differentiation** component of a model, I would argue that the primary question is simpler:

> Does the model rank observations correctly?

For this purpose, rank invariance is fundamental.

Suppose a model perfectly ranks all observations. Multiplying every predicted value by \(1000\) does not change the ranking, and therefore should not change our assessment of discrimination.

A direct pairwise Somers' \(D\), computed on the raw predicted and realised values, has exactly this property. It depends only on the ordering, not on the numerical scale.

The ECB gAUC does not necessarily share this invariance because the prediction scale is also used to define the discretisation of the realised values.

Consequently, the ECB metric mixes together two questions:

1. Does the model rank observations correctly?
2. Is the rating scale itself well aligned with realised outcomes?

Both are useful, but they are not the same validation objective.

## Segmentation is different

The ECB construction feels much more natural when validating a segmentation or a binned continuous risk driver.

Here, the predictor is already categorical (typically fewer than 20 categories), so constructing a contingency table is natural. One can order the categories by their target means, discretise the realised values accordingly, and compute Somers' \(D\) from the resulting table.

There is, however, one subtle point.

The choice of boundaries for the realised values is subjective.

At the other extreme, we could leave the realised values continuous. The contingency table would simply become very wide—potentially one column for every distinct realised value. Mathematically this is still valid, and in fact corresponds exactly to the direct pairwise computation.

## An interesting open question

The question becomes even more interesting in out-of-sample (OOS) or out-of-time (OOT) validation.

Should the realised-value boundaries be:

- **frozen** from the training sample, or
- **recomputed** using the realised distribution of the OOS/OOT sample?

The first tests whether the original segmentation generalises.

The second adapts the evaluation to the new data—but may also remove exactly the instability that the validation is supposed to detect.

I don't think there is an obvious answer.

More fundamentally, perhaps the choice of binning is not merely an implementation detail, but a reflection of **what we actually mean by discriminatory power**.

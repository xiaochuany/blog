---
date:
    created: 2026-07-23
authors: [xy]
categories: [TIL]
tags: [quant methods]
---

# Cross-Validation Isn't About the Loop
<!-- more -->

Technically, cv is just a loop so we shouldn't be scared of writing a custom one. 

This is particularly relevant when we combine sklearn orchestration API (e.g. pipeline of custom transformers) with a trainer API of boosted tree libraries: the native cv function of these tree libs expects internal dataset object which cannot call fit_transform for each fold. A manual loop solves this cleanly.

For pure sklearn workflow, the builtin cross_validate function is mostly sufficient because the integration of estimator with transformer in a single pipeline is complete and the post-cv inspection is possible through arguments like return_indices and return_estimator. 

For model validation, the important question is not whether we use a built-in function or write the loop ourselves. The EBA expects validation to challenge the intermediate steps of risk differentiation (say e.g. binning, selection, estimation ...), whereas a standard CV result mainly reports the performance of the final pipeline. A useful pattern is therefore to retain auditable evidence of each modelling choice e.g.  binning decisions, encoding mappings, correlation-based exclusions and variable-selection paths, all as fitted attributes that can be inspected for every fold. 
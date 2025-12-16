---
date:
    created: 2025-12-16
authors: [xy]
categories: [TIL]
tags: [AI assistant, regulations]
draft: true
---

# A case study for AI assisted review of regulatory texts

This post records my attempt of using AI to help reviewing a large number of regulatory texts, in particular, in the context of risk model development and validation in a highly regulated environment. 

In such environment, model development and validation is the art of constrainnt optimization. 
Regulators and supervisors set the constraints, and institutions optimize their business metrics over the space of constraint solutions.

Reading regulatory texts is a necessity in this context. Given the sheer amount of regulataions, directives, guidelines, guides ..., it is hard (if not unrealistic) for anyone to read and remember all the contraints for the compliance of their risk mdoels.   

My goal here is to lay out a workflow that uses AI to facilitate the process of retrieving the constraints, therefore allowing developers and validators spend more time on the interesting optimization part of their job. 

## Keep it simple 

For starters, just use the chat UI of your choice and a collection of prompts saved in a text file. When the workflow stabiliszes, it might worth developing an app tailored to the task and save a few clicks and copy paste, but not now. 

## Initial context

Give AI the right context is important for retrieval. A good starting point is to provide a 5-tuple 

(risk type, parameter, model type, data regime, use)

For examplle, (credit risk, PD, IRB, high-default porfolio, pillar 1 capital). 

## Breakdown of constraints by category

Asking "list all requirements for X per EBA" and hoping to get a comprehensive full coverage one-shot is unrealistic. A better approach is to think of any model as a function $f(t, \cdot)$, where we stress the dependance on time. Regulatory texts can be translated into some  structural properties of the function, such as 

1. existence
2. inequalities
3. invariants
4. scope of coverage
5. temporal dynamics

To consider a concrete example, let's consider the tuple 

(risk type, parameter, model type, data regime, use) = (IRRBB, average maturity, internal behaviour model, non-contractual data + customer behaviour, ALM + reporting + ICAAP). 

1. must exist (which article?)
2. when data volume decreases -> widen estimation error
3. some thing should be monotone
4. which portfolio
5. does it change in one year.

Fo each propertie

1. regulatory extraction: For IRB PD Margin of Conservatism, list binding ECB/EBA/CRR requirements related only to (existence / directionality / shape / scope / governance). Cite article or paragraph.
2. f

Prompt 1 — Regulatory extraction (per bucket)

“For IRB PD Margin of Conservatism, list binding ECB/EBA/CRR requirements related only to (existence / directionality / shape / scope / governance).
Cite article or paragraph.”

(run once per bucket)

Prompt 2 — Completeness test

“Are there any IRB PD MoC requirements that do not fall into the five buckets (existence, directionality, shape, scope, governance)?
If yes, list them and explain why.”

If “no” → coverage achieved.

Prompt 3 — Independence test

“Construct an IRB PD MoC implementation that satisfies all requirements except monotonicity.
What supervisory finding would ECB raise?”

Repeat for each bucket.

If ECB raises a distinct finding each time → model is complete.

Prompt 4 — Minimal counterexample

“Give the simplest IRB PD MoC implementation that looks reasonable but would be rejected by SSM.
Specify the violated requirement and reference.”

This catches fake-safe designs.

Prompt 5 — JST flip (threat model)

“You are an ECB JST validator reviewing this IRB PD MoC.
Which part would you challenge first and why?”

If answer mentions data scarcity / uncertainty coverage / rank violations, you’re aligned.

Prompt 6 — Regression test (future-proofing)

“Would any recent CRR / EBA / ECB updates invalidate any of the five IRB PD MoC bullets?
If yes, which one?”

If none → no rework.
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

In such environment, model development and validation is the art of constrainnt optimization. Regulators and supervisors set the constraints, and institutions optimize their business metrics over the space of constraint solutions.

Reading regulatory texts is a necessity in this context. Given the sheer amount of regulataions, directives, guidelines, guides ..., it is hard (if not unrealistic) for anyone to read and remember all the contraints for the compliance of their risk mdoels.   

My goal here is to lay out a workflow that uses AI to facilitate the process of retrieving the constraints, therefore allowing developers and validators spend more time on the interesting optimization part of their job. 

## Keep it simple 

For starters, just use the chat UI of your choice and a collection of prompts saved in a text file. When the workflow stabiliszes, it might worth developing an app tailored to the task and save a few clicks and copy paste, but not now. 

## Initial context

A good starting point is to provide a 6-tuple to AI: 

- jurisdiction
- risk type
- parameter
- approach
- portfolio
- use case

For examplle, (EU, credit risk, PD, IRB, retail, pillar 1 capital) narrows the scope quite a bit.    

## Break down constraints 

Asking "list all requirements for X per EBA" and hoping to get a comprehensive full coverage is unrealistic. 

Regulatory requirements may be viewed as constraints on the space of parameter estimators. To help AI achieving full coverage, we can design orthogonal dimensions so that *ideally* every requirement can be described with one and only one dimension. This is somewhat reminiscent of data quality dimensions. 

Here is an attempt

1. existence
2. directionality
3. shape
4. scope 
5. governance

Let's consider a concrete example: IRB PD rating assignments for EU credit risk retail portfolio.

## Prompts

1. IRB PD rating assignments for EU credit risk retail portfolio., list binding ECB/EBA/CRR requirements related only to (existence / directionality / shape / scope / governance). Cite article or paragraph.

2. Are there any IRB PD rating assignment requirements that do not fall into the five buckets (existence, directionality, shape, scope, governance)? If yes, list them and explain why.

3. Construct an IRB PD rating assignments implementation that satisfies all requirements except (existence / directionality / shape / scope / governance). What supervisory finding would ECB raise?

4. “Give the simplest IRB PD rating assignments implementation that looks reasonable but would be rejected by SSM.
Specify the violated requirement and reference.”

5. You are an ECB JST validator reviewing this IRB PD MoC. Which part would you challenge first and why?

6. Would any recent CRR / EBA / ECB updates invalidate any of the five IRB PD MoC bullets? If yes, which one?


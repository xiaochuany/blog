---
date:
    created: 2025-12-30
authors: [xy]
categories: [TIL]
tags: [AI assistant, regulations]
draft: true
---

# A case study for AI assisted review of regulatory texts

This post records my attempt of using AI to help reviewing a large number of regulatory texts, in particular, in the context of risk model development and validation in a highly regulated environment. 

In such environment, model development and validation is the art of constrainnt optimization. Regulators and supervisors set the constraints, and institutions optimize their business metrics over the space of constraint solutions.

Reading regulatory texts is a necessity in this context. Given the sheer amount of regulataions, directives, guidelines, technical standards, guides ..., it is hard (if not unrealistic) for anyone to read and remember all the contraints for the compliance of their risk mdoels.   

My goal here is to lay out a workflow that uses AI to facilitate the process of retrieving the constraints, therefore allowing developers and validators spend more time on the interesting optimization part of their job. 

## Keep it simple 

For starters, just use the chat UI of your choice and a collection of prompts saved in a text file. When the workflow stabiliszes, it might worth developing an app tailored to the task and save a few clicks and copy paste, but not now. 

## Initial context

A good starting point is to hand over a 6-tuple to AI: 

- jurisdiction: who rules
- risk type: broad category
- model component:  building block of the model e.g. parameter, sensitivity, scenario ... 
- methodology: regulatory approach e.g. IRB, IMA ... 
- exposure scope: which book/portfolio
- objective: usage

For example, 

- (EU, credit risk, PD, IRB, retail, pillar 1 capital)
- (EU, IRRBB, delta NII, SA, BB, pillar 2 SOT)

narrows down the scope of search quite a bit. 

## Break down constraints 

Asking "list all requirements for X per EBA" and hoping to get a comprehensive full coverage is unrealistic. 

Regulatory requirements may be viewed as constraints on the space of parameter estimators. To help AI achieving full coverage, we can design orthogonal dimensions so that *ideally* every requirement can be described with one and only one dimension. 

Here is an attempt

1. data standards: data quality, grouping, inclusion/exclusion, proxy data
2. quantifications: formula, floor/cap, conservatism, assumption
3. performance monitoring: backtesting, discriminatory power, stability, frequency
4. governance: roles/responsabilities, processes
5. documentation and reporting: templates/what should be included

## High level process

The idea is to retreive one bucket of requirements at a time. Then stress test it to verify completeness of the list and catch missing items.
The test phase can be done via search for clarifications, construct counterexamples, JST role play etc. We can be creative here.  

## Prompt template

Context: I am analyzing the following regulatory model:

- jurisdiction: EU (CRR)
- risk type: credit risk
- model component: PD
- methodology: IRB Advanced
- exposure scope: retail
- objective: pillar 1 capital

Prompt 1. retrieval

List CRR/EBA/ECB requirements specifically for the category: [INSERT CATEGORY HERE]. Cite article or paragraph. 

- If Category = data standards: Look for length of history, data quality, grouping, inclusion/exclusion, proxy data
- If Category = quantifications: Look for formula, floor/cap, conservatism, assumption
- If Category = performance monitoring: Look for backtesting, discriminatory power, stability, frequency
- If Category = governance: Look for roles/responsabilities, processes
- If Category = documentation and reporting: Look for templates, inclusion of topics

Prompt 2. clarifications

"Review the text for {Component}. Create a table with two columns:
Explicit Rules: (Must do X).
Implicit Expectations: (Should consider Y).
Highlight any requirement where the regulation is vague (uses words like 'adequate', 'sufficient', 'appropriate') and explain how a strict regulator interprets that vagueness in the context of {Objective}."

Prompt 3. devil's advocate

"I am proposing a simplified approach for {Component} to save costs.
The Proposal: [Insert a 1-sentence logic, e.g., 'I will use 3 years of data instead of 5 because the market structure changed.']
Your Task: Act as a strict regulator (e.g., ECB/PRA). Even if this sounds logical from a business perspective, find the specific {Category} regulation that explicitly forbids this. Quote the article and explain why my 'reasonable' logic is non-compliant."

Prompt 4. hidden constraint

"I have satisfied all the explicit mathematical formulas in the Methodology category.
However, look at the Governance and Data categories. Are there any 'qualitative overrides' or 'human judgment' requirements that effectively force me to alter the mathematical output?
Example: Does the regulation require a 'margin of conservatism' or a 'human overlay' that makes the pure math insufficient?"


Prompt 5. role play

"Adopt the persona of a lead investigator from the {Jurisdiction}. You are conducting an on-site inspection of my {Risk Stripe} model.
Based on the regulatory texts you retrieved regarding {Category}, ask me the single most difficult question you would pose to the Head of Modeling to expose a weakness.
Wait for my answer, and then evaluate if my defense holds up against the regulation."


Prompt 6. timeliness

"Would any recent {Jurisdiction} updates (e.g., CRR III, Basel IV, or recent EBA/PRA Consultation Papers) invalidate or modify the requirements we just discussed for {Component}?
Specifically, check for:
Transitional Arrangements: Are we currently in a phase-in period where two rules apply?
New RTS/GL: Are there recent Regulatory Technical Standards that clarify a previously vague article?
Sunset Clauses: Has the permission to use this specific {Methodology} been withdrawn for this {Exposure Scope} (e.g., removal of Advanced IRB for certain asset classes)?"

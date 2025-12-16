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

## Technical preparation 

For starters, just use the chat UI of your choice and a collection of prompts saved in a text file. When the workflow stabiliszes, it might worth developing an app more tailored to the task, but not now. 

## Initial context

Give the right context is important for retrieval. A good starting poitn is to provide a 5-tuple 

(risk type, parameter, model type, data regime, use)

For examplle, (credit risk, PD, IRB, high-default porfolio, pillar 1 capital). 

## Breakdown of constraints by category

Asking "list all requirements for X per EBA" and hoping to get a comprehensive full coverage one-shot is unrealistic. A better approach is to think of any model as a function $f(t, \cdot)$, where we stress the dependance on time. Regulatory texts can be translated into some  structural properties of the function, such as 

1. existence
2. inequalities
3. invariants
4. scope of coverage
5. temporal dynamics

To consider a concrete example, let's consider the tuple (IRRBB, average maturity, internal behaviour model, ..., ALM and reporting). 

1. must exist (which article?)
2. when data volume decreases -> widen estimation error
3. some thing should be monotone
4. which portfolio
5. does it change in one year.

Fo each propertie

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

## the setup 

For starters, just use the chat UI of your choice and a collection of prompts saved in a text file. When the workflow stabiliszes, it might worth developing an app more tailored to the task, but not now. 

## initial questions 


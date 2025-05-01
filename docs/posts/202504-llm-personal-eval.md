---
date:
    created: 2025-04-24
authors:
    - xy
categories:
    - TIL
tags:
    - LLM
---


# LLM eval for my personal usage 
<!-- more -->

As common evaluation metrics for LLM performance are being gamified, everyone starts to run some evaluation thingy most relevant for their own use case.

In this post I keep track of queries that I asked multiple LLMs and record the winner who gave the best answer. 
Here are the contesters: 

| contester | stands for |
|--------------|------------|
| G           | Gemini     |
| D            | deepseek   |
| C            | ChatGPT    |
| X           | Grok       |

The reason for excluding Claude is personal: the last time I checked, it appears to be not as generous as other frontier labs to offer their best-ish model to free tier users. And now I have too many chatbot tabs to manage :man_shrugging:

I include the date of the query to highlight the dynamic nature of these evals and help identify model versions if I/anyone wanted to. 



By default, I always enable the most powerful model available  to a free tier user (think and search for ChatGPT. Gemini 2.5 Pro etc in April 2025) for a "fair" comparison.



This post is regularly updated. 



## Paste image to markdown - 20250424

> in vscode in WSL, paste image from clipboard to markdown in customized directory.

C  wins. Other LLMs refer to some obsolete vscode extension which does not work. In think mode C started thinking about using the same extensions, but then made some search in vscode github project and elsewhere,  ended up finding  that this is a feature already built-in in vscode after v1.79 and provided the correct json for me to copy to user settings.   


## search the url of a file in github - 20250501

> show me the link to the source code of the new polars streaming engine. this should be hosted in github

G wins. C and X provide irrelevant links. D cannot search (functionality not working in the region where I am which is EU). 
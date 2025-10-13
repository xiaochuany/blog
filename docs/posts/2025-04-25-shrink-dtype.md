---
date:
    created: 2025-04-25
    updated: 2025-06-15
authors: [xy]
categories: [TIL]
tags: [data engineering, python]
---
# Puzzling query optimization behaviour and dtypes shrinking in polars

<!-- more -->
The official blog[^case] of `polars` shared a recent case study that's part counterintuitive, part educational. 

[^case]: https://pola.rs/posts/case-metro-digital/

## Counterintuitive bits

The authors found that for large DataFrames (~400M rows), using lazy execution with the streaming engine for a massive query plan -- built via the `pipe` method --  was *not* the optimal approach. This is basically saying: the query engine, one of polars' key selling point, struggles to optimize giant plans effectively.

The solution they found is to use lazy + streaming only on very expensive joins, and run  other parts of the query plan  eagerly. 

??? "Update on 2025-06-15"
    The new streaming engine achieves some impressive benchmark results per official blog in May 2025, potentially changing the verdict of this case study if it were conducted now. 

## Shrink dtype

Feels obvious, but it's a solid reminder: why would you use int64 if all your values never go above a few millions? Use  `shrink_dtype` on expressions/series and `shrink_to_fit` on dataframe to downcast dtypes to just what's needed. 

Watch out when joining on downcast columns - dtypes need to match. 

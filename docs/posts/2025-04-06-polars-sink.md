---
date:
    created: 2025-04-06
    updated: 2025-05-01
authors:
    - xy
categories: 
    - TIL
tags:
    - data-engineering
slug: polars-stream
---

# Polars streaming tricks
<!-- more -->

!!! background
    Polars' lazy execution model (see [my previous post](2024-12-21-polars.md)) allows query optimizer to "rewrite" the query to avoid loading all data into memory such as predicate/project pushdown, constant folding etc. However, the default execusion engine of the optimized query is still in-memory, i.e. loading all the data **required in the optimized query** into memory, therefore can be problematic if the required data is too large.  The streaming engine is an alternative.    
    
`polars` offers a convenient streaming capability via the `LazyFrame.collect(streaming=True)` method. Under the hood, this processes the large dataset in chunks, process them, cache the intermediate results and so on. 

A key limitation of this approach is that the final result (and possibly some intermediate results) must still fit entirely within the machine's RAM.
When the final result itself is too large for RAM but all the intermediate results fit into RAM, `polars` provides the sink_* methods as an alternative. These methods allow to write the output directly to disk. Here is an example from the official documentation:
```py
lf = pl.scan_csv("my_dataset/*.csv").filter(pl.all().is_not_null())

lf.sink_parquet(
    pl.PartitionMaxSize(
        "my_table_{part}.parquet",  # {part} is replaced by partition number
        max_size=512_000           # bytes
    )
)
```
This code processes the LazyFrame and stores the results as a sequence of Parquet files on disk, ensuring no single file exceeds the max_size. Because the output is written incrementally to disk, this method effectively removes the RAM limitation for the final result size.

One can use this pattern to break a complicated query into sequences of **subquery-sink** routines, which would remove the RAM constraint at the cost of longer query running time (round trips to disc is slower than operating in RAM).   

Github issues tracker for the new streaming engine [^stream] can be used before a dedicated page on streaming engine functionalities/roadmap. In particular it is in the roadmap that the new streaming engine will support automatically writing to disc when intermediate results are too large, so the afore-mentioned pattern can be performed without user figuring out how to break queries themselves. 

[^stream]:https://github.com/pola-rs/polars/issues/20947
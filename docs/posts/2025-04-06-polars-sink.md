---
date:
    created: 2025-04-06
authors:
    - xy
categories: 
    - TIL
tags:
    - data-engineering
---

# Polars streaming tricks

When handling datasets larger than the available RAM on a single machine, `polars` offers a convenient streaming capability via the `LazyFrame.collect(streaming=True)` method. Under the hood, this processes the large dataset in chunks, aggregating the results at the end. However, a key limitation of this approach is that the final aggregated result must still fit entirely within the machine's RAM.

When the final result itself is too large for RAM, `polars` provides the sink_* methods as an alternative. These methods allow to write the output directly to disk. Here is an example from the official documentation:
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
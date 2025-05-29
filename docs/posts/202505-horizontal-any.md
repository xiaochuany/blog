---
date:
    created: 2025-05-29
authors: [xy]
categories: [TIL]
tags: [data-engineering]
---

# Horizontal ops in snowpark

Horizontal ops such as horizontal any (think `np.any(..., axis=1)`) can be achieved with a chain of logical OR e.g.  `(... | ... | ... )`
or by using `F.when().otherwise()`. When the number of conditions/columns increases, I would like to use something similar to polars 

```py
import polars as pl

lf.select(pl.any_horizontal(col_names)) # col_names: list[str]
```

or the general purpose reduce/fold function for horizontal ops in polars. 

In snowpark there is no reduce function. 
But one can use python `functools.reduce`.

```py
import snowflake.snowpark.functions as F
from functools import reduce

any_expr = reduce(lambda a, b: a | b, map(F.col, column_names))
lf.select(any_expr)
```



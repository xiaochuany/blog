---
date:
    created: 2025-04-20
authors:
    - xy
categories:
    - TIL
tags:
    - data-engineering
slug: df-gen-polars
---

# Generate test dataframe with `polars` (powered by `hypothesis`) 

The sub-module `polars.testing.parametric` provide tools for generating fake data for testing purposes.
Here is an example showing what can be done with just `dataframes` and `column` functions in this module

```py
import polars as pl
from polars.testing.parametric import dataframes, column

def generate(size=5):
    return dataframes(
    [
        column("id", dtype=pl.UInt16, unique=True, allow_null=False), 
        column("value", dtype=pl.Int16, allow_null=True), 
        column("cat", dtype =pl.Enum("XYZ"), allow_null=False)
    ], 
    min_size=size, max_size=size)

original = generate().example()
```

The output is random, i.e. evey call to the `example` method would generate a new dataframe with the prescribed characteristics (this method is for interactive use only). One can test their data pipelines on fake data with precise schema and simulated data quality deficiencies (eg null values, nan, inf, etc). 

For unittesting, here is an example from the offical docs

```py
from hypothesis import given
@given(df=dataframes(min_size=3, max_size=5))
def test_df_height(df: pl.DataFrame) -> None:
    assert 3 <= df.height <= 5
```
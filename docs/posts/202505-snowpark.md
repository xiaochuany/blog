---
date:
    created: 2025-05-19
    updated: 2025-05-31 
authors: [xy]
categories: [TIL]
tags: [data engineering]
---

# Intro to Snowpark API

<!-- more -->

Polars users might find the Snowpark/PySpark API verbose.
But the high-level abstractions roughly map: `snowpark.Column` is like `polars.Expr`, and `snowpark.DataFrame` is like `polars.LazyFrame`.

Caveat: the list of methods on `Column` is much smaller than on `Expr`, and there’s no clear namespace separating methods by dtype.
If you don’t see a method for your use case, check the gigantic list under `snowpark.functions`.

Let’s explore.

## First, imports

```py
from snowflake import snowpark as sp
from snowflake.snowpark import functions as F
from snowflake.snowpark import types as T
```

## `Column` class

Accessible via `F.col`, e.g. `lf.select(F.col("A"))`.

A few notable methods on `Column` instances:

* `cast`
* `name` (alias)
* `asc` / `desc`
* `is_null`
* `is_in`
* `over`
* `substr`

Unlike Polars, there’s no method namespace separation by dtype.
Still, `Column` feels a lot like `polars.Expr`.

## `F` namespace (functions)

The `F` namespace provides many ways to manipulate `Column`s.

Note: all functions return a `Column`, not just `F.col`, e.g.

* `F.concat`
* `F.contains`
* `F.array_size`
* `F.year`
* `F.when(...).otherwise(...)`
* `F.explode` and `F.flatten`

Again, no subnamespaces by dtype — everything is dumped into `functions`.

## `Window` class

Provides a bunch of class methods:

* `Window.partition_by`
* `Window.order_by`
* `Window.range_between` (time or int)

And constants like:

* `Window.CURRENT_ROW`

used as an argument to the `over` method.

## `T` (types)

For type hints and casting, use:

* `T.ArrayType`
* `T.DateType`
* `T.StructType`

Caveat: these are callables — instantiate them like `T.DateType()`.

## `DataFrame` class

It’s called a DataFrame, but really behaves like a LazyFrame — call `.collect()` to materialize.

Notable methods (all return a new DataFrame):

* `select`
* `filter`
* `pivot` / `unpivot`
* `join`
* `union`
* `with_columns` (takes `list[str]`, `list[Column]`)
* `with_column`
* `distinct`
* `fillna`

and the `.columns` attribute.

Note: after joining, columns with the same name need to be aliased explicitly to save/view the joined table.
e.g. t1.col.alias("c1"), t2.col.alias("c2").  

## IO

Ways to bring data in/out:

* `session.table("TABLE_NAME")` → DataFrame
* `session.sql("SELECT ...")` → DataFrame
* `session.read.format(...).schema(...).options(...).load(...)`
* `lf.write.format(...).save(...)`, etc.

## `Row` class

You get a list of Rows after `collect()`
The `.as_dict()` method on Row makes it easy to interoperate with Polars:
just pass a list of dicts to construct a Polars DataFrame.

## Testing

* `sp.testing.assert_dataframe_equal`

## Horizontal ops in snowpark

Horizontal ops such as horizontal any (think `np.any(..., axis=1)`) can be achieved with a chain of logical OR e.g.  `(... | ... | ... )`
or by using `F.when().otherwise()`. When the number of conditions/columns increases, I would like to use something similar to polars 

```py
import polars as pl
lf.select(pl.any_horizontal(col_names)) # col_names: list[str]
```

or the general purpose reduce/fold function for horizontal ops in polars. 

```py
lf.select(pl.reduce(lambda a,b:a|b, exprs = pl.col(col_names)))
```

In snowpark there is no any_horizontal, nor reduce function. 
But one can use python `functools.reduce`.

```py
from functools import reduce

any_expr = reduce(lambda a, b: a | b, map(F.col, col_names))
lf.select(any_expr)
```
## Exception

The most common is `sp.exceptions.SnowparkSQLException`, corresponding to 13xx SQL error.
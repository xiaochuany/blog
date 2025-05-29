---
date:
    created: 2025-05-26
authors: [xy]
categories: [TIL]
tags: [data-engineering]
---

# Working with Large Datasets in Snowflake
<!-- more -->
This post covers how to work with large datasets in Snowflake e.g. 100+ GB, billions of rows. You can't really do this kind of stuff on your personal laptop (mine has 6 cores and 16GB RAM), so most people turn to the cloud for this workload.

We'll go over some practical tips, using the Snowpark API version 1.32 to illustrate.

```python
from snowflake import snowpark as sp
from snowflake.snowpark import functions as F
from snowflake.snowpark import types as T
```

## Choose the Right Warehouse

There’s no one-size-fits-all. Larger warehouses cost more, so don’t scale up unless you actually need the extra horsepower. Match the warehouse size to the workload. Start small and scale up only if performance is a problem.

## Cache the Results

Snowpark DataFrames are lazily evaluated. That means if you reuse part of a query in multiple branches, Snowflake will recompute that shared part unless you cache it explicitly.

Example:

```py
a = session.table("...")
query = a.group_by("A").agg(...)  # simulate expensive intermediate step, and this line won't execute anything yet due to lazy evaluation

# branching out
res1 = query.filter(...).collect()
res2 = query.select(...).collect()
```

Both `res1` and `res2` will trigger the same expensive computation. To avoid that, materialize the intermediate result:

### Temp Table (session scoped)

```py
query.write.mode("overwrite").save_as_table("temp_table", table_type="temporary")
lf = session.table("temp_table")
```

### Materialized View (persistent, auto-refreshed, costs money)

```py
session.sql("""
    create materialized view my_view as
    select id, count(*)
    from my_table
    group by id
""").collect()
```

### Parquet File (in stage)

```py
path = f"{session.get_session_stage()}/f.parquet"
a.group_by("A").agg(...).write.format("parquet").save(path)
lf = sp.DataFrameReader.parquet(path)
```

## Predicate and Projection Pushdown

"Predicate" = filter. "Projection" = column selection.

Snowflake tries to optimize automatically, but it only knows what you tell it—and lazy evaluation doesn't help. If you're building up a complex query, be explicit:

* Filter early, especially before joins
* Drop columns you don’t need
* Use `DataFrame.explain()` to inspect the query plan

## Monitoring

Use `QUERY_HISTORY` to monitor performance. Look at:

* `bytes_scanned`
* `partitions_scanned`

Lower is better. If you're scanning too much data, it's a sign that pushdowns or clustering could be improved.

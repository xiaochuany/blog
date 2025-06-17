---
date: 
    created: 2025-06-17
authors: [xy]
categories: [TIL]
tags: [data engineering]
---

# Exploring a large database

Big organizations typically have their data stored in one or multiple database. 
Sometimes, analytics team find themselves buried in the jungle of tables/views with non consistent naming conventions, 
therefore hard to build a holistic view about where to find which data etc.  

In this post we explore a few tricks to ease the initial phase of understanding a large database. 

## Having applications in mind

Very often in business, database are the backend of some frontend applications with GUI that one can interact with. Therefore, it is not a bad idea to use the frontend application first hand to see what's possible, the meaning of a column, relationships of columns in a table etc.

This can be a back-and-forth process, while doing analytics work, check the GUI to see where are the column names appear in the application, gradually completing a good mental picture of the 'geography' of the tables.  

## Use metadata tables

Almost all database systems have a kind of information schema which contains tables that describe the metadata of the database such as the existing schemas, tables, views, columns etc. 

In Oracle it's `ALL_TABLES` `ALL_VIEWS` `ALL_TAB_COLUMNS` (not a schema per se, rather individual tables that start with `ALL_` prefix). 
In Postgres/DuckDB/Snowflake it's the `information_schema`  schema, with tables such as `tables`, `columns` in it. 

## Foreign keys

For obvious reasons, it is important to know these keys for the tables of interest. In Oracle, use `ALL_CONSTRAINTS` table.  
In Snowflake/DuckDB/Postgres, use `information_schema.REFERENTIAL_CONSTRAINTS`.

## Describe a table

Literally `DESCRIBE _NAME_` to show the column names and types. 

## Visualizations

Some DB has their native visualization tools, ER generators. Not a bad idea to take advantage of that.  
# build a validation library

## data assumption 

The library operates on Polars LazyFrames or DataFrames. While column names are configurable via `IRBConfig`, the following concepts are expected:

- **date**: Temporal column (Date or Datetime) for longitudinal analysis.
- **target/default**: Boolean or integer (0/1) indicator of the default event.
- **score**: Continuous numerical output from the model (used for ranking and separation checks).
- **grade**: Categorical or ordinal rating grade (used for heterogeneity and homogeneity checks).
- **pd**: Predicted probability of default (used for calibration checks).
- **lgd / realized_lgd**: Estimated and realized Loss Given Default.
- **ccf / realized_ccf**: Estimated and realized Credit Conversion Factor.
- **attributes**: Any extra categorical or numerical columns. These are **not part of the global config** but are passed as arguments to specific methods (e.g., `check_homogeneity(variables=[...])`).

Note: `dr` (Default Rate) is a calculated metric (actual defaults / total) used during validation, not an input column.

## api design

- [x] register a `irb` namespace for polars lazyframe or dataframe (`.irb.configure()`).
- [x] implementation of a builder pattern with fluent api (`Report`).
- [x] implementation of lazy execution (checks are stored as callables and run on `.show()`).
- [x] implementation of a functional HTML report output.

The dataframe to start with is the training sample. 
The `.add_samples()` method allows binding additional datasets:

- [x] oot / oos
- [x] application sample

## core maths

- [x] **gini**: measures discriminatory power.
- [x] **ks**: Kolmogorov-Smirnov statistic for distribution separation.
- [x] **psi**: Population Stability Index.
- [x] **auc**: (Note: implemented inside gini).
- [ ] **somers_d**: ordinal association.
- [x] **binomial_test**: calibration validation.
- [ ] **brier_score**: accuracy of probability predictions.
- [ ] **anderson_darling**: goodness-of-fit for continuous distributions.
- [x] **t_test**: central tendency comparison (used for LGD/CCF predictive ability).
- [ ] **hosmer_lemeshow**: overall calibration goodness-of-fit.
- [ ] **spiegelhalter**: accuracy/calibration test.
- [ ] **jeffreys**: Bayesian calibration for low-default portfolios.

## methods 

### 1. Risk Differentiation
- [x] **check_ranking**: Gini, AUC.
- [x] **check_separation**: KS.
- [x] **check_heterogeneity**: Between grades; Pairwise Z-tests, ANOVA.
- [x] **check_homogeneity**: Within grades; Pairwise Z-tests across segments (e.g. Chi-Square substitute).
- [x] **check_concentration**: Concentration in rating scale; HHI, Distributional Analysis.

### 2. Risk Quantification (Calibration)
- [x] **check_calibration**: Grade level; Binomial Test (PD Calibration).
- [x] **check_predictive_ability**: LGD/ELBE calibration; paired t-test on residuals.
- [x] **check_ccf_calibration**: CCF calibration; paired t-test on residuals.
- [ ] **check_goodness_of_fit**: Overall model; Hosmer-Lemeshow, Spiegelhalter.
- [ ] **check_low_default**: Jeffreys test for LDPs.
- [ ] **check_monitoring**: Traffic Light System.

### 3. Stability & Transitions
- [x] **check_representativeness**: Overall population stability (PSI) vs baseline.
- [x] **check_segment_migration**: Migration among internal segments.
- [x] **check_grade_migration**: Migration between rating grades.
- [ ] **check_characteristic_stability**: CSI for model inputs/features over time.
- [ ] **check_performance_stability**: Gini trend analysis, Over-time calibration stability.
- [ ] **check_parameter_stability**: Rolling window analysis, Wald tests, Chow test.

## showing and inspecting results

The library provides two main ways to interact with the validation results: visual inspection and programmatic access.

### 1. Visual Inspection (`.show()`)

The `.show()` method is designed for use in Jupyter notebooks. It triggers the execution of all queued checks and renders them using a premium HTML interface.

- **Check Cards**: Each check is displayed in its own card with a title and description.
- **Labeled Artifacts**: Results like charts (Altair) and tables (Polars) are automatically displayed. If a check produces multiple outputs, they are tagged (e.g., `chart`, `table`) for clarity.
- **Lazy Execution**: Checks are only computed when `.show()` (or `.run()`) is called.

### 2. Programmatic Access (`.run()`)

The `.run()` method executes the checks and returns a `ValidationResults` object, which acts as a container for all output.

- **Summary Table**: In a notebook, printing or returning a `ValidationResults` object displays a concise HTML summary table of all checks performed and the types of artifacts they generated.
- **Accessing Tables**: The `.tables` property of the `ValidationResults` object automatically extracts all Polars DataFrames from the results (even those nested in dictionaries) into a single dictionary. This is useful for further analysis or exporting results to Excel/CSV.
- **Iteration**: You can iterate over the results or access them by index (e.g., `results[0]`) to inspect individual `CheckResult` objects.

Example usage:
```python
results = report.run()

# Get a specific table from the results
gini_table = results.tables["Ranking check: Gini of the score"]

# Loop through artifacts
for res in results:
    print(f"Check: {res.name}, Artifact Type: {type(res.artifact)}")
```
## LGD performing 

columns

- facility_id
- date_default_start
- lgd_realized
- lgd_estimatd
- grade

heteromogeneity (one-sided U test for comparison of consecutive grades): lgd_realized, grade, date_default_start
homogeneity (two sided U test for subpopultions inside a grade): lgd_realized, grade, date_default_start, more variables
calibration (t test): lgd_realized, lgd_estimated, grade

## LGD in default 

columns 

- facility
- date_default_start
- date_obs
- lgd_realized_t (0 time into the default, lge_realized_0 is just ldg_realied)
- lgd_estimated_t (elbe)
- grade

calibration: lgd_realized_t, lgd_estimated_t, grades.  








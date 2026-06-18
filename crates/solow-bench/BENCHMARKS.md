# Solow estimator benchmarks

This crate measures the **speed** of the core Solow estimators with
[criterion](https://crates.io/crates/criterion). It is a *performance* harness,
not a correctness one: there is **no reference fixture** here and no golden
values are checked. Numerical correctness of every estimator is covered
separately by the per-crate fixture tests (e.g. `solow-glm/tests/reference.rs`).

All inputs are generated **deterministically** by a small linear-congruential
generator (`Lcg`), so the workloads are fully reproducible and are never seeded
from the wall clock. `black_box` is used on every input and result so the
optimizer cannot fold the work away.

## Running

```sh
# Compile only (no measurement):
cargo bench -p solow-bench --no-run

# Full run (default criterion sampling — minutes):
cargo bench -p solow-bench

# Quick run with a short measurement window (what produced the table below):
cargo bench -p solow-bench --bench estimators -- \
    --warm-up-time 0.5 --measurement-time 1.5 --sample-size 30
```

## Workloads

The benchmarks are organized into five criterion groups:

| Group | Benchmark | Workload |
| --- | --- | --- |
| `ols_fit` | `n100_k5`, `n1000_k5` | OLS fit, 5 predictors + intercept, `n = 100` and `n = 1000` |
| `glm_poisson_irls` | `n500_k5` | GLM Poisson fit via IRLS (log link), `n = 500`, 5 predictors |
| `logit_newton` | `n500_k5` | Discrete Logit fit via Newton's method, `n = 500`, 5 predictors |
| `linalg` | `svd_200x50` | Economy (one-sided Jacobi) SVD of a `200 x 50` matrix |
| `linalg` | `eigh_100x100` | Symmetric eigendecomposition of a `100 x 100` SPD matrix |
| `dist_throughput` | `norm_cdf`, `norm_ppf` | Normal `cdf` / `ppf` over 1024 points per iteration |
| `dist_throughput` | `t_cdf_df5`, `t_ppf_df5` | Student-t (df = 5) `cdf` / `ppf` over 1024 points per iteration |

## Indicative results

The numbers below are the criterion **median estimates** from a short run
(`--measurement-time 1.5 --sample-size 30`). **The machine is unspecified** and
results will vary substantially with CPU, build flags, and load — treat these as
*indicative orders of magnitude only*, not as a benchmark of record. The crate
builds with the workspace release profile (`opt-level = 3`, thin LTO).

| Group / benchmark | Median |
| --- | ---: |
| `ols_fit/n100_k5` | ~15.9 µs |
| `ols_fit/n1000_k5` | ~131.6 µs |
| `glm_poisson_irls/n500_k5` | ~304.1 µs |
| `logit_newton/n500_k5` | ~160.0 µs |
| `linalg/svd_200x50` | ~2.92 ms |
| `linalg/eigh_100x100` | ~72.5 ms |
| `dist_throughput/norm_cdf` (1024 pts) | ~112.4 µs |
| `dist_throughput/norm_ppf` (1024 pts) | ~221.5 µs |
| `dist_throughput/t_cdf_df5` (1024 pts) | ~77.1 µs |
| `dist_throughput/t_ppf_df5` (1024 pts) | ~1.30 ms |

### Notes on the shape of the results

- OLS scales roughly linearly in `n` for fixed `k` (the `1000`-row fit is about
  8x the `100`-row fit), as expected for the `XᵀX` formation that dominates.
- The two iterative MLE fits (`glm_poisson_irls`, `logit_newton`) are
  dominated by repeated weighted least-squares / Newton solves; both land in the
  hundreds-of-microseconds range at `n = 500`.
- The cyclic-Jacobi symmetric eigendecomposition (`eigh_100x100`) is by far the
  heaviest workload — Jacobi sweeps over a `100 x 100` matrix are `O(n³)` per
  sweep with several sweeps to converge — and is the natural candidate for future
  optimization.
- For the distribution functions, the inverse-CDF (`ppf`) routines are
  consistently slower than the forward `cdf`, reflecting the root-finding /
  series-inversion work; the Student-t `ppf` is the most expensive of the four.

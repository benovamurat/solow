# solow-emplike

Empirical-likelihood (EL) inference on descriptive statistics of a univariate
sample, following Owen (2001) and validated against the reference
`emplike.descriptive` module.

The entry point is [`DescStat`], which exposes

* [`DescStat::test_mean`] — EL test of a hypothesized mean `mu0`,
* [`DescStat::ci_mean`] — EL confidence interval for the mean,
* [`DescStat::test_var`] — EL test of a hypothesized variance `sig2_0`.

## Method

For a hypothesized mean `mu0`, the empirical-likelihood weights `w_i` maximize
`sum log(w_i)` subject to `sum w_i = 1` and `sum w_i (x_i - mu0) = 0`. The dual
solution is `w_i = 1 / (n (1 + eta (x_i - mu0)))`, where the Lagrange
multiplier `eta` is the root of `sum (x_i - mu0) / (1 + eta (x_i - mu0)) = 0`.
The EL test statistic is `-2 sum log(n w_i)`, asymptotically chi-squared with
one degree of freedom.

The variance test profiles out a nuisance mean: for each candidate mean it
solves a two-parameter EL dual by a modified Newton iteration, then minimizes
the resulting `-2 logELR` over the nuisance mean.

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-emplike) · License: BSD-3-Clause

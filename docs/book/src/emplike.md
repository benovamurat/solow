# Empirical likelihood

The `solow-emplike` crate performs *nonparametric* likelihood inference on the
descriptive statistics of a univariate sample. Instead of assuming a parametric
error distribution, empirical likelihood (EL) places a discrete probability
\\( w_i \\) on each observation and tests hypotheses by maximizing the product
of those weights subject to moment constraints. The single entry point is the
`DescStat` type, which provides an EL ratio test for the mean, an EL confidence
interval for the mean, and an EL ratio test for the variance. Every test returns
a `TestResult` carrying the \\( -2\log\mathrm{ELR} \\) statistic and its
asymptotic p-value.

> This crate covers the *descriptive-statistics* slice of the reference's
> empirical-likelihood functionality (mean and variance of a single sample). The
> reference additionally offers EL for regression, accelerated-failure-time
> models, and ANOVA; those are not implemented here.

## Background

Given a sample \\( x_1, \dots, x_n \\), the (nonparametric) empirical likelihood
of a distribution that puts mass \\( w_i \ge 0 \\) on \\( x_i \\) is
\\( \prod_{i=1}^{n} w_i \\), and it is maximized by the empirical distribution
\\( w_i = 1/n \\). To test a hypothesis about a parameter \\( \theta \\) expressed
through an estimating equation \\( \sum_i w_i\, g(x_i, \theta) = 0 \\), we
maximize \\( \prod_i w_i \\) subject to that constraint together with
\\( \sum_i w_i = 1 \\). The **empirical-likelihood ratio** compares this
constrained maximum against the unconstrained \\( 1/n \\) optimum, and Wilks'
theorem gives

\\[
-2\log\mathrm{ELR}(\theta) \;=\; -2 \sum_{i=1}^{n} \log\!\big(n\, w_i\big)
\;\xrightarrow{d}\; \chi^2_{q},
\\]

with \\( q \\) the number of constraints. `solow-emplike` returns this statistic
in `TestResult::stat` and the upper-tail probability \\( \Pr(\chi^2_q > \text{stat}) \\)
in `TestResult::pvalue`.

**Mean.** For a hypothesized mean \\( \mu_0 \\) the single constraint is
\\( \sum_i w_i (x_i - \mu_0) = 0 \\). Introducing a Lagrange multiplier
\\( \eta \\), the dual solution is

\\[
w_i \;=\; \frac{1}{n}\,\frac{1}{1 + \eta\,(x_i - \mu_0)},
\qquad
\sum_{i=1}^{n} \frac{x_i - \mu_0}{1 + \eta\,(x_i - \mu_0)} \;=\; 0,
\\]

and \\( \eta \\) is found by solving the scalar equation on the right with a
bracketed Brent root finder. The resulting statistic is referred to
\\( \chi^2_1 \\). Inverting the test — finding the set of \\( \mu_0 \\) for which
\\( -2\log\mathrm{ELR}(\mu_0) \le \chi^2_{1,\,1-\alpha} \\) — yields the EL
confidence interval; `ci_mean` performs this inversion through a re-parameterized
"gamma" root search at each endpoint.

**Variance.** Testing \\( \sigma^2 = \sigma_0^2 \\) uses the two estimating
equations

\\[
g(x_i;\,\mu,\sigma_0^2) \;=\;
\big(\,x_i - \mu,\;\; (x_i - \mu)^2 - \sigma_0^2\,\big),
\\]

where the mean \\( \mu \\) is a *nuisance* parameter. For a fixed \\( \mu \\) the
two-dimensional Lagrange multiplier \\( \eta \in \mathbb{R}^2 \\) is obtained by a
modified Newton iteration on the dual, and the profile statistic is minimized over
\\( \mu \\) with a bounded scalar minimizer. The profiled \\( -2\log\mathrm{ELR} \\)
is again asymptotically \\( \chi^2_1 \\).

## Example

`DescStat::new` takes a plain `&[f64]` sample (at least two observations). Here we
build a small sample, test whether its mean equals several candidate values, form
a 95% confidence interval for the mean, and test a hypothesized variance.

```rust
use solow_emplike::DescStat;

// A small univariate sample.
let x = [
    3.1, 4.2, 2.8, 5.0, 3.7, 4.4, 2.9, 3.5, 4.8, 3.3,
];
let d = DescStat::new(&x);
println!("n = {}", d.nobs());

// EL test that the mean equals 4.0.
let r = d.test_mean(4.0);
println!("test_mean(4.0): stat = {:.4}, p = {:.4}", r.stat, r.pvalue);

// 95% EL confidence interval for the mean (significance level 0.05).
let (lo, hi) = d.ci_mean(0.05);
println!("95% CI for the mean: [{:.4}, {:.4}]", lo, hi);

// EL test that the variance equals 0.8.
let v = d.test_var(0.8);
println!("test_var(0.8): stat = {:.4}, p = {:.4}", v.stat, v.pvalue);
```

What this prints (behavior, not fabricated numbers): `nobs()` reports `10`.
`test_mean` returns a `TestResult` whose `stat` field is the
\\( -2\log\mathrm{ELR} \\) statistic and whose `pvalue` is the
\\( \chi^2_1 \\) upper-tail probability; evaluating `test_mean` exactly at the
sample mean yields a statistic of (numerically) zero and a p-value near one,
and the statistic grows as `mu0` moves away from the sample mean. `ci_mean`
returns a `(low, high)` tuple that brackets the sample mean; by construction the
test statistic at each endpoint equals the critical value
\\( \chi^2_{1,\,0.95} \approx 3.8415 \\). `test_var` returns zero (up to numerical
tolerance) when the hypothesized variance equals the sample's population variance,
and a larger statistic otherwise.

If you need to mirror the reference's tuning knobs for the confidence interval,
`ci_mean_opts` exposes the search parameters explicitly:

```rust
use solow_emplike::DescStat;

let x = [3.1, 4.2, 2.8, 5.0, 3.7, 4.4, 2.9, 3.5, 4.8, 3.3];
let d = DescStat::new(&x);

// Same as ci_mean(0.05), but with the reference's default search bounds spelled out.
let (lo, hi) = d.ci_mean_opts(0.05, 1e-8, -1e10, 1e10);
println!("CI = [{:.4}, {:.4}]", lo, hi);
```

## Module reference

**Models**

| Name | Description |
| --- | --- |
| `DescStat` | Empirical-likelihood inference for a univariate sample; entry point for all EL tests and intervals. |

**Results**

| Name | Description |
| --- | --- |
| `TestResult` | Outcome of an EL hypothesis test: the `stat` (\\( -2\log\mathrm{ELR} \\)) and asymptotic `pvalue` fields. |

**Methods on `DescStat`**

| Name | Description |
| --- | --- |
| `DescStat::new` | Build a `DescStat` from a sample slice (`&[f64]`, panics on fewer than two points). |
| `DescStat::nobs` | Number of observations in the sample. |
| `DescStat::test_mean` | EL ratio test of a hypothesized mean `mu0`; returns a `TestResult`. |
| `DescStat::ci_mean` | EL confidence interval `(low, high)` for the mean at significance level `sig`. |
| `DescStat::ci_mean_opts` | `ci_mean` with explicit search parameters (`epsilon`, `gamma_low`, `gamma_high`). |
| `DescStat::test_var` | EL ratio test of a hypothesized variance `sig2_0`, profiling out the mean; returns a `TestResult`. |

Full API: see the generated rustdoc for `solow-emplike`.

## References

- Owen, A. B. (2001). *Empirical Likelihood*. Chapman & Hall/CRC, Boca Raton.
- Owen, A. B. (1988). "Empirical likelihood ratio confidence intervals for a
  single functional." *Biometrika*, 75(2), 237–249.
- Owen, A. B. (1990). "Empirical likelihood ratio confidence regions."
  *The Annals of Statistics*, 18(1), 90–120.
- DiCiccio, T., Hall, P., and Romano, J. (1991). "Empirical likelihood is
  Bartlett-correctable." *The Annals of Statistics*, 19(2), 1053–1061.

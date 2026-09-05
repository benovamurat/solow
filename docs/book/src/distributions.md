# Distributions & special functions

The [`solow-distributions`] crate implements the statistical distributions and
special functions that underpin all of Solow's inference — entirely in pure
Rust, validated against the reference (`scipy.stats` and `scipy.special`). You
will rarely call it directly when fitting models, but it is the right tool for
computing p-values, critical values, and tail probabilities by hand.

## Free functions

For each continuous distribution there is a family of free functions whose names
mirror `scipy.stats`: `pdf`, `cdf`, `sf` (survival function, `1 − cdf`), `ppf`
(percent-point / inverse cdf), and `isf` (inverse survival function).

```rust
use solow_distributions::{chi2_sf, norm_cdf, norm_ppf, t_sf};

// Normal: Φ(1.96) and the 97.5% critical value.
println!("Phi(1.96) = {:.6}", norm_cdf(1.96));   // 0.975002
println!("z_0.975   = {:.6}", norm_ppf(0.975));  // 1.959964

// Two-sided p-value from a t statistic with 12 degrees of freedom.
println!("two-sided p (t) = {:.6}", 2.0 * t_sf(2.5, 12.0));

// Upper-tail chi-squared probability (3 d.o.f.).
println!("chi2 sf = {:.6}", chi2_sf(7.81, 3.0));
```

The normal functions take no parameters (`norm_*`); `t_*` and `chi2_*` take the
degrees of freedom; `f_*` takes the numerator and denominator degrees of
freedom.

## Distribution objects

Each distribution also has a struct with `pdf`, `cdf`, `sf`, and `ppf` methods,
which is convenient when you reuse the same parameters repeatedly:

```rust
use solow_distributions::{Normal, StudentT};

let z = Normal::new(0.0, 1.0);   // loc, scale
println!("N(0,1).cdf(0)   = {:.4}", z.cdf(0.0));   // 0.5
println!("N(0,1).ppf(0.9) = {:.4}", z.ppf(0.9));

let t = StudentT::new(10.0);     // degrees of freedom
println!("t(10).ppf(0.95) = {:.4}", t.ppf(0.95));  // 1.8125
```

`ChiSquared` and `FDist` follow the same pattern. The `continuous_ext` module
adds `Beta`, `Cauchy`, `Exponential`, `Gamma`, `Laplace`, `Logistic`,
`LogNormal`, `Pareto`, `Uniform`, and `WeibullMin`. Discrete distributions
(`Binomial`, `Geometric`, `NegativeBinomial`, `Poisson`) live in the `discrete`
module.

## Special functions

The `special` module exposes the building blocks used throughout the stack:

| Function | Meaning |
| --- | --- |
| `lgamma`, `gamma`, `digamma` | (log-)gamma and its derivative |
| `betainc`, `betaincinv`, `lbeta` | regularized incomplete beta and inverse |
| `gammainc`, `gammaincc`, `gammaincinv` | regularized incomplete gamma and inverse |
| `erf`, `erfc`, `erfinv` | error function and inverses |

These match `scipy.special` to tight tolerances and are what make the
distribution functions accurate in the tails.

## Empirical distributions

The `empirical` module provides the empirical CDF (`Ecdf`) and a general
right/left-continuous `StepFunction`, useful for goodness-of-fit work.

[`solow-distributions`]: https://github.com/solow-rs/solow

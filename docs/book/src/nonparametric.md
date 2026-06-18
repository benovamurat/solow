# Nonparametric methods

The [`solow-nonparametric`] crate provides smoothers that make no parametric
assumption about the regression function or density: LOWESS scatterplot
smoothing, rule-of-thumb bandwidth selectors, univariate kernel density
estimation, and kernel regression. Every routine is verified against the
canonical Python reference.

## LOWESS

`lowess(y, x, options)` performs locally-weighted scatterplot smoothing
(Cleveland 1979) — a robust, locally-weighted linear regression. The defaults
match the reference: `frac = 2/3` of the data per local fit, `it = 3`
robustifying iterations.

```rust
use ndarray::array;
use solow_nonparametric::{lowess, LowessOptions};

let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let y = array![1.0, 1.9, 3.2, 3.8, 5.1, 5.9, 7.2, 7.8];

let fit = lowess(&y, &x, LowessOptions::default()).unwrap();
println!("x        = {:?}", fit.x);
println!("smoothed = {:?}", fit.fitted);
```

Tune the smoother through `LowessOptions { frac, it, delta }`. A larger `frac`
gives a smoother fit; more iterations `it` increases robustness to outliers; a
nonzero `delta` enables linear interpolation between fit points for speed on
large data.

## Bandwidth selectors

The rule-of-thumb bandwidth functions take a sample and return a scalar
bandwidth:

```rust
use ndarray::array;
use solow_nonparametric::{bw_scott, bw_silverman};

let sample = array![0.1, 0.4, 0.35, 0.8, 1.1, 0.9, 0.2, 0.6];

println!("Silverman = {:.4}", bw_silverman(&sample).unwrap());
println!("Scott     = {:.4}", bw_scott(&sample).unwrap());
```

`bw_normal_reference` provides the normal-reference plug-in rule used as the
default in the KDE.

## Kernel density estimation

`KdeUnivariate::new(sample).fit(bandwidth)` estimates a univariate density on a
generated support grid. The [`Bandwidth`] enum selects the rule (`Scott`,
`Silverman`, `NormalReference`) or a user-supplied value:

```rust
use ndarray::array;
use solow_nonparametric::{Bandwidth, KdeUnivariate};

let sample = array![0.1, 0.4, 0.35, 0.8, 1.1, 0.9, 0.2, 0.6];

let kde = KdeUnivariate::new(sample).fit(Bandwidth::Silverman).unwrap();
println!("support[0] = {:.4}, density[0] = {:.4}", kde.support[0], kde.density[0]);
```

The result `KdeFit` exposes the `support` grid and the matching `density`
values. To evaluate the density at arbitrary points instead of the grid, use
`KdeUnivariate::evaluate(points, bandwidth)`.

## Kernel regression

For nonparametric regression of `y` on one or more predictors, [`KernelReg`]
(with a [`RegType`] selecting local-constant or local-linear fitting) and the
multivariate density [`KdeMultivariate`] are also provided.

[`solow-nonparametric`]: https://github.com/solow-rs/solow
[`Bandwidth`]: ./nonparametric.md
[`KernelReg`]: ./nonparametric.md
[`RegType`]: ./nonparametric.md
[`KdeMultivariate`]: ./nonparametric.md

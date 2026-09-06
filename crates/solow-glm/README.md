# solow-glm

**Generalized linear models for Rust.** Gaussian, Binomial, Poisson, Gamma, Inverse-Gaussian, Negative-Binomial, and Tweedie families with the full link-function set, fit by iteratively reweighted least squares.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-glm = "0.7"
```

## Quick example

```rust
use solow_glm::{Glm, Family, Link};

let m = Glm::builder(y, x)
    .family(Family::Poisson)
    .link(Link::Log)
    .fit()?;
println!("{}", m.summary());
```

## What is inside

### Families and links

| Family | Default link | Alternative links |
|---|---|---|
| `Gaussian` | identity | log, inverse |
| `Binomial` | logit | probit, log, cloglog, cauchit |
| `Poisson` | log | identity, sqrt |
| `Gamma` | inverse | log, identity |
| `InverseGaussian` | 1/mu² | log, identity, inverse |
| `NegativeBinomial` | log | identity, sqrt |
| `Tweedie` | log | identity |

### Estimators

| Estimator | What it does |
|---|---|
| `Glm` | Fit any family + link combination by IRLS. Full results (deviance, Pearson χ², null deviance, AIC, BIC, standard errors, confidence intervals, every residual type) with a printable summary table. |
| `PoissonRegressor` | Ergonomic thin wrapper for `Family::Poisson` + `Link::Log`. |
| `GammaRegressor` | Ergonomic thin wrapper for `Family::Gamma` + `Link::Log`. |
| `TweedieRegressor` | Tweedie GLM with configurable power parameter for compound Poisson-Gamma targets. |

### Diagnostics on every fit

- Deviance residuals, Pearson residuals, working residuals, response residuals.
- Null and residual deviance, dispersion / scale parameter.
- Log-likelihood at the MLE, AIC, BIC.
- Coefficient standard errors, Wald z / t statistics, confidence intervals.
- Full printable summary table matching the classical inference layout.

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run for every family and link combination.
- Closed-form Gaussian identity path agrees bit-wise with OLS to `1e-14`.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.

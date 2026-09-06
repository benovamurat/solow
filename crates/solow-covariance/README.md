# solow-covariance

**Covariance matrix estimators for Rust.** Empirical, shrinkage (Ledoit-Wolf, OAS, arbitrary), robust minimum covariance determinant, sparse inverse via graphical lasso, and elliptic-envelope outlier detection.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-covariance = "0.7"
```

## Quick example

```rust
use solow_covariance::LedoitWolf;

let lw = LedoitWolf::new().fit(&x)?;
let cov = &lw.covariance;
let shrinkage = lw.shrinkage;
```

## What is inside

| Estimator | What it does |
|---|---|
| `EmpiricalCovariance` | Sample covariance with an optional store-precision flag. Log-likelihood, Mahalanobis distance, error norm. |
| `ShrunkCovariance` | Convex shrinkage toward a scaled identity target with a caller-set intensity. |
| `LedoitWolf` | Ledoit-Wolf 2004 optimal shrinkage intensity estimator. |
| `Oas` | Oracle Approximating Shrinkage (Chen 2010), better for small samples. |
| `MinCovDet` | Rousseeuw FAST-MCD robust covariance with consistency correction. |
| `GraphicalLasso` | L1-penalized inverse covariance for sparse precision (partial correlations). |
| `EllipticEnvelope` | Robust ellipsoidal outlier detector built on top of MinCovDet. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.

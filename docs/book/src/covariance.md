# Covariance estimation

`solow-covariance` covers empirical, shrinkage (Ledoit-Wolf, OAS, arbitrary), robust minimum covariance determinant, sparse inverse via graphical lasso, and elliptic-envelope outlier detection.

## Estimators

| Estimator | What it does |
|---|---|
| `EmpiricalCovariance` | Sample covariance with an optional store-precision flag. Log-likelihood, Mahalanobis distance, error norm. |
| `ShrunkCovariance` | Convex shrinkage toward a scaled identity target with a caller-set intensity. |
| `LedoitWolf` | Ledoit-Wolf (2004) optimal shrinkage intensity estimator. |
| `Oas` | Oracle Approximating Shrinkage (Chen 2010), better for small samples. |
| `MinCovDet` | Rousseeuw FAST-MCD robust covariance with consistency correction. |
| `GraphicalLasso` | L1-penalized inverse covariance for sparse precision (partial correlations). |
| `EllipticEnvelope` | Robust ellipsoidal outlier detector built on top of MinCovDet. |

## Example

```rust
use solow_covariance::LedoitWolf;

let lw = LedoitWolf::new().fit(&x)?;
let cov = &lw.covariance;
let shrinkage = lw.shrinkage;
# Ok::<_, solow_core::Error>(())
```

## References

- Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis.
- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). *Shrinkage algorithms for MMSE covariance estimation*. IEEE TSP.
- Rousseeuw, P. J., & Van Driessen, K. (1999). *A fast algorithm for the minimum covariance determinant estimator*. Technometrics.
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). *Sparse inverse covariance estimation with the graphical lasso*. Biostatistics.

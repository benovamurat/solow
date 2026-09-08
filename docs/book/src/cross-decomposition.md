# Cross-decomposition (PLS, CCA)

`solow-cross-decomposition` provides partial least squares in three flavours and canonical correlation via the NIPALS solver.

## Estimators

| Estimator | What it does |
|---|---|
| `PLSRegression` | Predictive PLS: finds latent components that maximise covariance between X and Y. Used for regression with correlated predictors and multi-output targets. |
| `PLSCanonical` | Symmetric PLS with mode-A deflation, giving equally-scaled X and Y projections. |
| `PLSSVD` | Direct SVD of the cross-covariance matrix. |
| `CCA` | Canonical correlation analysis via NIPALS with mode-B deflation. |

## Example

```rust
use solow_cross_decomposition::PLSRegression;

let pls = PLSRegression::new(2).fit(&x, &y)?;
let y_pred = pls.predict(&x_test)?;
# Ok::<_, solow_core::Error>(())
```

## References

- Wold, H. (1966). *Estimation of principal components and related models by iterative least squares*. In Multivariate Analysis.
- Rosipal, R., & Krämer, N. (2006). *Overview and recent advances in partial least squares*. SLSFS.

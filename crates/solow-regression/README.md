# solow-regression

**Linear regression and regularization for Rust.** Ordinary least squares, weighted and generalized least squares, penalised regression, robust regression, quantile regression, online learners, and Bayesian ridge, in one memory-safe pure-Rust crate.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-regression = "0.7"
```

## Quick example

```rust
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;

let design = add_constant(&x, true, HasConstant::Add)?;
let res = LinearModel::ols(y, design)?.fit()?;
println!("{}", res.summary(Some(&["const", "x"])));
```

## What is inside

### Ordinary and generalized least squares

| Estimator | What it does |
|---|---|
| `LinearModel::ols` | OLS via QR/SVD. Never forms XᵀX, so ill-conditioned designs (Longley, cond ~10¹⁰) still match certified coefficients to `1e-13`. |
| `LinearModel::wls` | Weighted least squares with per-observation weights. |
| `LinearModel::gls` | Generalized least squares with a user-supplied covariance. |
| `Glsar` | GLS with AR(p) autocorrelated errors. |
| `LinearModel::rls` | Recursive least squares for online updates. |
| `LinearModel::rolling` | Rolling-window OLS. |
| `QuantReg` | Quantile regression via linear programming. |

### Regularized regression

| Estimator | What it does |
|---|---|
| `Ridge` | L2-penalised regression via closed-form Cholesky. |
| `RidgeCV` | Ridge with automatic α selection via LOO. |
| `Lasso` | L1-penalised regression via coordinate descent. |
| `LassoCV` | Lasso with cross-validated α on a regularization path. |
| `ElasticNet` | Convex mix of L1 and L2 penalties. |
| `ElasticNetCV` | ElasticNet with cross-validated α and l1 ratio. |
| `Lars` | Least-angle regression. |
| `LassoLars` | LARS solving the Lasso path. |
| `LassoLarsIC` | LassoLars with AIC or BIC model selection. |
| `OrthogonalMatchingPursuit` | Greedy k-sparse regression. |
| `MultiTaskLasso`, `MultiTaskElasticNet` | Joint regression across correlated targets with group-sparse coefficients. |
| `KernelRidge` | Kernel ridge regression with RBF, polynomial, sigmoid, cosine kernels. |

### Bayesian regression

| Estimator | What it does |
|---|---|
| `BayesianRidge` | Bayesian ridge with iterated α, λ hyperparameter update. |
| `ARDRegression` | Automatic relevance determination with per-feature precision (relevance vector). |

### Robust regression

| Estimator | What it does |
|---|---|
| `HuberRegressor` | Huber loss regression, robust to moderate outliers. |
| `RansacRegressor` | Random-sample consensus for high-outlier data. |
| `TheilSenRegressor` | Median of pairwise slopes, robust to many outliers. |

### Online and classification

| Estimator | What it does |
|---|---|
| `SgdRegressor`, `SgdClassifier` | Stochastic-gradient online learners with L2, L1, or elastic-net penalty. |
| `Perceptron` | Classical Rosenblatt perceptron. |
| `PassiveAggressiveRegressor`, `PassiveAggressiveClassifier` | Online margin-based updates. |
| `RidgeClassifier`, `RidgeClassifierCV` | One-hot regression classifier with ridge regularization. |
| `DummyRegressor`, `DummyClassifier` | Baseline models for benchmarking. |

### Inference

Every fitted `LinearResults` exposes coefficient standard errors, t and F statistics, R² and adjusted R², AIC and BIC, confidence intervals, the printable summary table, and the full robust-covariance battery: HC0, HC1, HC2, HC3, HAC (Newey-West), and one-way cluster.

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- NIST StRD certified cases pass with worst-case relative error `2.5e-10`.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.

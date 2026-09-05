# Linear regression

The [`solow-regression`] crate estimates linear models by (generalized) least
squares. A single type, [`LinearModel`], constructs ordinary, weighted, and
generalized least squares models; `.fit()` returns a [`LinearResults`] with the
full inference battery.

All three estimators follow the canonical *whitening* formulation: the model
carries an implied error covariance, the design and response are whitened so the
transformed problem is ordinary least squares, and the pseudo-inverse of the
whitened design yields the coefficients and the normalized covariance.

> **Intercepts are explicit.** Solow does not add a constant column for you. Use
> `solow_core::tools::add_constant` to prepend one when you want an intercept.

## Ordinary least squares (OLS)

```rust
use ndarray::{array, Array1};
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;

let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
let y: Array1<f64> = array![1.1, 1.9, 3.2, 3.9, 5.1];

let design = add_constant(&x, true, HasConstant::Add).unwrap();
let res = LinearModel::ols(y, design).unwrap().fit().unwrap();

assert_eq!(res.params.len(), 2);
println!("R-squared = {:.3}", res.rsquared);
```

## Weighted least squares (WLS)

Supply one positive weight per observation (proportional to the inverse of that
observation's error variance). Internally the model treats the error covariance
as `diag(1 / weights)`.

```rust
use ndarray::{array, Array1};
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;

let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
let y: Array1<f64> = array![1.1, 1.9, 3.2, 3.9, 5.1];
let weights: Array1<f64> = array![1.0, 1.0, 0.5, 0.5, 0.25];

let design = add_constant(&x, true, HasConstant::Add).unwrap();
let res = LinearModel::wls(y, design, weights).unwrap().fit().unwrap();

println!("WLS params = {:?}", res.params);
```

## Generalized least squares (GLS)

Pass a full `n × n` symmetric positive-definite error covariance `sigma`. This
covers, for example, autocorrelated errors where you have modeled the
covariance structure directly.

```rust
use ndarray::{array, Array1, Array2};
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;

let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
let y: Array1<f64> = array![1.1, 1.9, 3.2, 3.9, 5.1];

// An AR(1)-style error covariance with rho = 0.5.
let n = 5;
let rho = 0.5;
let mut sigma = Array2::<f64>::zeros((n, n));
for i in 0..n {
    for j in 0..n {
        sigma[[i, j]] = rho.powi((i as i32 - j as i32).abs());
    }
}

let design = add_constant(&x, true, HasConstant::Add).unwrap();
let res = LinearModel::gls(y, design, &sigma).unwrap().fit().unwrap();

println!("GLS params = {:?}", res.params);
```

## Robust (sandwich) covariances

When the spherical-errors assumption behind the textbook covariance
`scale · (XᵀX)⁻¹` fails, replace it with a *sandwich* estimator. The robust
covariance methods on `LinearResults` take the design matrix and a
[`CovType`]; they return a recomputed covariance, standard errors, t-values,
and p-values without re-fitting the model.

`CovType` variants:

- `CovType::Hc0` — White heteroskedasticity-consistent, no small-sample
  correction.
- `CovType::Hc1` — HC0 scaled by `n / (n − k)`.
- `CovType::Hc2` — each squared residual divided by `1 − hᵢᵢ` (leverage).
- `CovType::Hc3` — each residual divided by `1 − hᵢᵢ`, then squared.
- `CovType::Hac { maxlags, use_correction }` — Newey–West HAC with a Bartlett
  kernel over `maxlags` lags (heteroskedasticity- and
  autocorrelation-consistent).
- `CovType::Cluster { groups, use_correction }` — one-way clustered, with an
  integer group id per observation.

```rust
use ndarray::{array, Array1};
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::{CovType, LinearModel};

let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
let y: Array1<f64> = array![1.1, 2.3, 2.9, 4.2, 4.8, 6.3];
let design = add_constant(&x, true, HasConstant::Add).unwrap();

let model = LinearModel::ols(y, design.clone()).unwrap();
let res = model.fit().unwrap();

// Heteroskedasticity-consistent (HC3) standard errors.
let bse_hc3 = res.bse_robust(&design, &CovType::Hc3).unwrap();
println!("HC3 std errors = {:?}", bse_hc3);

// HAC (Newey–West) with 2 lags and the small-sample correction.
let hac = CovType::Hac { maxlags: 2, use_correction: true };
let bse_hac = res.bse_robust(&design, &hac).unwrap();
println!("HAC std errors = {:?}", bse_hac);

// Cluster-robust with one group id per observation.
let clustered = CovType::Cluster {
    groups: vec![0, 0, 1, 1, 2, 2],
    use_correction: true,
};
let cov = res.cov_params_robust(&design, &clustered).unwrap();
println!("clustered cov shape = {:?}", cov.dim());
```

The matching methods are `cov_params_robust`, `bse_robust`, `tvalues_robust`,
and `pvalues_robust`. There is also a free function `robust_cov` for working
directly with a design, residuals, and the normalized covariance.

## Beyond least squares

`solow-regression` also provides:

- **Quantile regression** — [`QuantReg`].
- **Rolling OLS** — [`RollingOLS`] for moving-window estimates.
- **Recursive least squares** — [`RecursiveLS`].
- **Feasible GLS for AR errors** — [`Glsar`].
- **Sliced inverse regression** — [`SlicedInverseReg`] for dimension reduction.

For robust *M-estimation* (Huber, Tukey), see the separate `solow-robust`
crate.

[`solow-regression`]: https://github.com/solow-rs/solow
[`LinearModel`]: ./regression.md
[`LinearResults`]: ./regression.md
[`CovType`]: ./regression.md
[`QuantReg`]: ./regression.md
[`RollingOLS`]: ./regression.md
[`RecursiveLS`]: ./regression.md
[`Glsar`]: ./regression.md
[`SlicedInverseReg`]: ./regression.md

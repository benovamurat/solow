//! Panic-safety / degenerate-input tests for the regression models.
//!
//! These exercise non-finite, empty, single-row, rank-deficient, and
//! zero-variance designs. The goal is *graceful* behavior — a clean `Err`
//! or a finite result via the SVD fallback — never a panic. None of these
//! assert specific numeric values, so they do not affect numerical parity.

use ndarray::{array, Array1, Array2};
use proptest::prelude::*;
use solow_core::error::Error;
use solow_regression::LinearModel;

// ---------------------------------------------------------------------------
// Explicit degenerate cases.
// ---------------------------------------------------------------------------

#[test]
fn ols_rejects_nan_endog() {
    let y = array![1.0, f64::NAN, 3.0];
    let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
    assert!(matches!(LinearModel::ols(y, x), Err(Error::Value(_))));
}

#[test]
fn ols_rejects_inf_exog() {
    let y = array![1.0, 2.0, 3.0];
    let x = array![[1.0, 0.0], [1.0, f64::INFINITY], [1.0, 2.0]];
    assert!(matches!(LinearModel::ols(y, x), Err(Error::Value(_))));
}

#[test]
fn ols_rejects_empty_design() {
    let y: Array1<f64> = Array1::zeros(0);
    let x: Array2<f64> = Array2::zeros((0, 2));
    assert!(LinearModel::ols(y, x).is_err());
}

#[test]
fn wls_rejects_nan_weights() {
    let y = array![1.0, 2.0, 3.0];
    let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
    let w = array![1.0, f64::NAN, 1.0];
    assert!(matches!(LinearModel::wls(y, x, w), Err(Error::Value(_))));
}

#[test]
fn wls_rejects_inf_weights() {
    let y = array![1.0, 2.0, 3.0];
    let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
    let w = array![1.0, f64::INFINITY, 1.0];
    assert!(matches!(LinearModel::wls(y, x, w), Err(Error::Value(_))));
}

#[test]
fn gls_rejects_nan_sigma() {
    let y = array![1.0, 2.0, 3.0];
    let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
    let mut sigma = Array2::<f64>::eye(3);
    sigma[[0, 0]] = f64::NAN;
    assert!(matches!(
        LinearModel::gls(y, x, &sigma),
        Err(Error::Value(_))
    ));
}

#[test]
fn ols_single_row_does_not_panic() {
    // A 1x1 design is rank-deficient for inference but must not panic.
    let y = array![3.0];
    let x = array![[1.0]];
    let model = LinearModel::ols(y, x).expect("constructs");
    // Either succeeds (params finite) or returns an error — but never panics.
    if let Ok(res) = model.fit() {
        assert!(res.params.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn ols_perfectly_collinear_design_uses_svd_fallback() {
    // Column 2 == 2 * column 1: rank-deficient design. The SVD pseudoinverse
    // path must produce finite coefficients rather than panicking.
    let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = array![
        [1.0, 1.0, 2.0],
        [1.0, 2.0, 4.0],
        [1.0, 3.0, 6.0],
        [1.0, 4.0, 8.0],
        [1.0, 5.0, 10.0],
    ];
    let res = LinearModel::ols(y, x).unwrap().fit();
    if let Ok(r) = res {
        assert!(r.params.iter().all(|v| v.is_finite()));
        assert!(r.rsquared.is_finite());
    }
}

#[test]
fn ols_zero_variance_column_does_not_panic() {
    // A duplicate constant column gives a rank-deficient design.
    let y = array![1.0, 2.0, 3.0, 4.0];
    let x = array![
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],
        [1.0, 1.0, 3.0],
    ];
    let res = LinearModel::ols(y, x).unwrap().fit();
    if let Ok(r) = res {
        assert!(r.params.iter().all(|v| v.is_finite()));
    }
}

// ---------------------------------------------------------------------------
// Property-based invariants (modest case counts to keep CI fast).
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// For finite, well-conditioned designs, OLS yields finite coefficients,
    /// R^2 lies in [0, 1], and the covariance matrix is symmetric.
    #[test]
    fn ols_invariants_for_finite_inputs(
        slope in -5.0f64..5.0,
        intercept in -5.0f64..5.0,
        xs in prop::collection::vec(-10.0f64..10.0, 6..24),
        noise in prop::collection::vec(-0.5f64..0.5, 6..24),
    ) {
        // Align lengths.
        let n = xs.len().min(noise.len());
        prop_assume!(n >= 4);
        // Require some spread in x so the design is well-conditioned.
        let xmin = xs[..n].iter().cloned().fold(f64::INFINITY, f64::min);
        let xmax = xs[..n].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        prop_assume!(xmax - xmin > 1.0);

        let y: Array1<f64> = (0..n)
            .map(|i| intercept + slope * xs[i] + noise[i])
            .collect::<Vec<_>>()
            .into();
        let mut x = Array2::<f64>::ones((n, 2));
        for i in 0..n {
            x[[i, 1]] = xs[i];
        }

        let res = LinearModel::ols(y, x).unwrap().fit().unwrap();

        // Params finite.
        prop_assert!(res.params.iter().all(|v| v.is_finite()));
        // R^2 in [0, 1] (allow tiny FP slack).
        prop_assert!(res.rsquared >= -1e-9 && res.rsquared <= 1.0 + 1e-9);
        // Covariance symmetric.
        let cp = &res.cov_params;
        let (r, c) = cp.dim();
        prop_assert_eq!(r, c);
        for i in 0..r {
            for j in 0..c {
                prop_assert!((cp[[i, j]] - cp[[j, i]]).abs() <= 1e-8 + 1e-8 * cp[[i, j]].abs());
            }
        }
    }

    /// Any NaN in endog must yield an error, never a panic.
    #[test]
    fn ols_nan_endog_always_errors(
        mut ys in prop::collection::vec(-10.0f64..10.0, 4..16),
        nan_idx in 0usize..16,
    ) {
        let n = ys.len();
        ys[nan_idx % n] = f64::NAN;
        let y: Array1<f64> = ys.into();
        let mut x = Array2::<f64>::ones((n, 2));
        for i in 0..n {
            x[[i, 1]] = i as f64;
        }
        prop_assert!(LinearModel::ols(y, x).is_err());
    }
}

//! Reference-fixture tests for solow-regression.
//!
//! LinearRegression (OLS) and Ridge (Cholesky solver) are both
//! deterministic closed-form fits, so agreement is bit-wise up to
//! `1e-10` (a slightly wider bar than the elementary-metric fixtures
//! because the reference Ridge uses double-precision LAPACK Cholesky
//! while solow uses a pure-Rust Cholesky — the two accumulate
//! rounding differently but converge to the same coefficients).

use ndarray::{Array1, Array2, ArrayView2};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_regression::{LinearModel, Ridge};

const TOL: f64 = 1e-10;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/linear")
}

fn load(name: &str) -> Value {
    let path = fixtures().join(format!("{name}.json"));
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn f2d(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();
    let (n, d) = (rows.len(), rows[0].len());
    Array2::from_shape_vec((n, d), rows.into_iter().flatten().collect()).unwrap()
}

fn f1d(v: &Value) -> Array1<f64> {
    Array1::from(
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<_>>(),
    )
}

fn assert_close(name: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let d = (a - b).abs();
        assert!(
            d <= TOL,
            "{name}[{i}]: {a:.17} vs the reference {b:.17} (diff {d:.3e})"
        );
    }
}

// ---------------------------------------------------------------------------
// LinearRegression (OLS)
// ---------------------------------------------------------------------------

#[test]
fn linear_regression_matches_reference_bitwise() {
    let f = load("linear_regression");
    let x = f2d(&f["inputs"]["x"]);
    let y = f1d(&f["inputs"]["y"]);
    let lm = LinearModel::ols(y.clone(), add_intercept_col(&x))
        .unwrap()
        .fit()
        .unwrap();
    // solow LinearModel returns `params` = [intercept, β] when the design
    // matrix includes a constant column.
    let expected_intercept = f["expected"]["intercept"].as_f64().unwrap();
    let expected_coef = f["expected"]["coef"].as_array().unwrap();
    let got_intercept = lm.params[0];
    let d = (got_intercept - expected_intercept).abs();
    assert!(
        d <= TOL,
        "intercept: {got_intercept} vs the reference {expected_intercept} (diff {d:.3e})"
    );
    for (i, want) in expected_coef.iter().enumerate() {
        let g = lm.params[1 + i];
        let w = want.as_f64().unwrap();
        let dd = (g - w).abs();
        assert!(
            dd <= TOL,
            "coef[{i}]: {g:.17} vs the reference {w:.17} (diff {dd:.3e})"
        );
    }
}

// ---------------------------------------------------------------------------
// Ridge (Cholesky closed form)
// ---------------------------------------------------------------------------

#[test]
fn ridge_matches_reference_within_cholesky_tolerance() {
    let f = load("ridge");
    let x = f2d(&f["inputs"]["x"]);
    let y = f1d(&f["inputs"]["y"]);
    let alpha = f["inputs"]["alpha"].as_f64().unwrap();
    let ridge = Ridge::fit(y.view(), x.view(), alpha, true).unwrap();
    let expected_intercept = f["expected"]["intercept"].as_f64().unwrap();
    let expected_coef: Vec<f64> = f["expected"]["coef"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let d = (ridge.intercept - expected_intercept).abs();
    assert!(
        d <= TOL,
        "intercept: {} vs the reference {} (diff {d:.3e})",
        ridge.intercept,
        expected_intercept
    );
    assert_close("coef", ridge.coef.as_slice().unwrap(), &expected_coef);
    // Predictions.
    let pred = ridge.predict(x.view()).unwrap();
    let expected_pred: Vec<f64> = f["expected"]["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_close("predictions", pred.as_slice().unwrap(), &expected_pred);
}

// Prepend a column of ones to `x` for the intercept — the OLS entry point
// in solow-regression takes the fully-formed design matrix.
fn add_intercept_col(x: &Array2<f64>) -> Array2<f64> {
    let (n, d) = x.dim();
    let mut out = Array2::<f64>::ones((n, d + 1));
    for i in 0..n {
        for j in 0..d {
            out[[i, j + 1]] = x[[i, j]];
        }
    }
    out
}

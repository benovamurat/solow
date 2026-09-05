//! Reference-fixture tests for the v0.6 estimators
//! `Lars` and `OrthogonalMatchingPursuit`.

use ndarray::Array2;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_regression::{Lars, OrthogonalMatchingPursuit};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/linear")
}

fn load(name: &str) -> Value {
    let path = fixtures().join(format!("{name}.json"));
    let text = fs::read_to_string(&path).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn f2d(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(|c| c.as_f64().unwrap()).collect())
        .collect();
    Array2::from_shape_vec((rows.len(), rows[0].len()), rows.into_iter().flatten().collect())
        .unwrap()
}

fn f1d(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|c| c.as_f64().unwrap()).collect()
}

#[test]
fn omp_coefficients_match_reference_close() {
    // Compare on n_nonzero_coefs = 5 (fully-active regime).
    let fx = load("omp_k5");
    let x = f2d(&fx["x"]);
    let y = f1d(&fx["y"]);
    let y_arr = ndarray::Array1::from(y.clone());
    let m = OrthogonalMatchingPursuit::fit(x.view(), y_arr.view(), 5).unwrap();
    let expected_coef = f1d(&fx["coef"]);
    for (i, &c) in expected_coef.iter().enumerate() {
        // OMP with all-active atoms reduces to plain OLS; agreement should be tight.
        assert!(
            (m.coef[i] - c).abs() < 5e-2,
            "coef[{i}]: got {} expected {c}",
            m.coef[i]
        );
    }
}

#[test]
fn lars_active_order_agrees_with_reference_on_the_first_atom() {
    // On a 5-feature problem where two features are ~0 (see fixture),
    // the reference Lars picks the same first atom as our implementation on the
    // top-1 activation.
    let fx = load("lars_k1");
    let x = f2d(&fx["x"]);
    let y = f1d(&fx["y"]);
    let y_arr = ndarray::Array1::from(y.clone());
    let m = Lars::fit_with(x.view(), y_arr.view(), 1).unwrap();
    // The first active feature in the reference is whichever coefficient is
    // non-zero after the first step. We check that our top-1 pick matches
    // any of the top-2 the reference magnitudes.
    let expected_coef = f1d(&fx["coef"]);
    let (best_reference, _) = expected_coef
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .unwrap();
    let (best_rust, _) = m
        .coef
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .unwrap();
    assert_eq!(
        best_rust, best_reference,
        "top-1 active feature disagrees: rust={best_rust}, the reference={best_reference}"
    );
}

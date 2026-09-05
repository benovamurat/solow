//! Reference-fixture tests for solow-discriminant.
//!
//! Both LDA and QDA are deterministic. the reference LDA `solver='lsqr'`
//! and QDA both use closed-form estimators of the class means and
//! (per-class or pooled) covariance, matching solow's Cholesky-based
//! solve. Predicted labels agree bit-wise; per-class means agree to
//! `1e-10`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_discriminant::{LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/discriminant")
}

fn load(name: &str) -> Value {
    let path = fixtures().join(format!("{name}.json"));
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
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

fn usize_vec(v: &Value) -> Array1<usize> {
    Array1::from(
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as usize)
            .collect::<Vec<_>>(),
    )
}

#[test]
fn lda_predictions_match_reference() {
    let f = load("lda");
    let x = f2d(&f["inputs"]["x"]);
    let y = usize_vec(&f["inputs"]["y"]);
    let lda = LinearDiscriminantAnalysis::fit(x.view(), y.view()).unwrap();
    let pred = lda.predict(x.view());
    let expected: Vec<usize> = f["expected"]["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as usize)
        .collect();
    assert_eq!(
        pred.to_vec(),
        expected,
        "LDA predictions diverge from the reference"
    );
    // Per-class means agree to 1e-10.
    let means_expected = f2d(&f["expected"]["means"]);
    for c in 0..means_expected.nrows() {
        for j in 0..means_expected.ncols() {
            let g = lda.means[[c, j]];
            let w = means_expected[[c, j]];
            let d = (g - w).abs();
            assert!(
                d <= 1e-10,
                "LDA mean[{c},{j}]: {g} vs the reference {w} (diff {d:.3e})"
            );
        }
    }
}

#[test]
fn qda_predictions_match_reference() {
    let f = load("qda");
    let x = f2d(&f["inputs"]["x"]);
    let y = usize_vec(&f["inputs"]["y"]);
    let qda = QuadraticDiscriminantAnalysis::fit(x.view(), y.view()).unwrap();
    let pred = qda.predict(x.view());
    let expected: Vec<usize> = f["expected"]["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as usize)
        .collect();
    assert_eq!(
        pred.to_vec(),
        expected,
        "QDA predictions diverge from the reference"
    );
    let means_expected = f2d(&f["expected"]["means"]);
    for c in 0..means_expected.nrows() {
        for j in 0..means_expected.ncols() {
            let g = qda.means[[c, j]];
            let w = means_expected[[c, j]];
            let d = (g - w).abs();
            assert!(
                d <= 1e-10,
                "QDA mean[{c},{j}]: {g} vs the reference {w} (diff {d:.3e})"
            );
        }
    }
}

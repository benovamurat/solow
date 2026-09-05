//! Reference-fixture tests for solow-naive-bayes.
//!
//! All three tested classifiers (GaussianNB, MultinomialNB, BernoulliNB)
//! are deterministic maximum-likelihood fits; predictions agree bit-wise
//! with the reference.

use ndarray::{Array1, Array2};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_naive_bayes::{BernoulliNB, GaussianNB, MultinomialNB};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/naive_bayes")
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
fn gaussian_nb_matches_reference() {
    let f = load("gaussian");
    let x = f2d(&f["inputs"]["x"]);
    let y = usize_vec(&f["inputs"]["y"]);
    let nb = GaussianNB::fit(x.view(), y.view()).unwrap();
    let pred = nb.predict(x.view());
    let expected: Vec<usize> = f["expected"]["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as usize)
        .collect();
    assert_eq!(pred.to_vec(), expected, "GaussianNB predictions diverge");
    // Per-class means agree to 1e-12.
    let theta_expected = f2d(&f["expected"]["theta"]);
    for c in 0..theta_expected.nrows() {
        for j in 0..theta_expected.ncols() {
            let g = nb.theta[[c, j]];
            let w = theta_expected[[c, j]];
            let d = (g - w).abs();
            assert!(
                d <= 1e-12,
                "GNB theta[{c},{j}]: {g} vs the reference {w} (diff {d:.3e})"
            );
        }
    }
}

#[test]
fn multinomial_nb_matches_reference() {
    let f = load("multinomial");
    let x = f2d(&f["inputs"]["x"]);
    let y = usize_vec(&f["inputs"]["y"]);
    let alpha = f["inputs"]["alpha"].as_f64().unwrap();
    let nb = MultinomialNB::fit_with(x.view(), y.view(), alpha).unwrap();
    let pred = nb.predict(x.view());
    let expected: Vec<usize> = f["expected"]["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as usize)
        .collect();
    assert_eq!(pred.to_vec(), expected, "MultinomialNB predictions diverge");
}

#[test]
fn bernoulli_nb_matches_reference() {
    let f = load("bernoulli");
    let x = f2d(&f["inputs"]["x"]);
    let y = usize_vec(&f["inputs"]["y"]);
    let nb = BernoulliNB::fit(x.view(), y.view()).unwrap();
    let pred = nb.predict(x.view());
    let expected: Vec<usize> = f["expected"]["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as usize)
        .collect();
    assert_eq!(pred.to_vec(), expected, "BernoulliNB predictions diverge");
}

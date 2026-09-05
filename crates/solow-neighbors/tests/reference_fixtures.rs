//! Reference-fixture tests for solow-neighbors.
//!
//! `KNeighborsClassifier` and `KNeighborsRegressor` are deterministic
//! given (i) the same tie-breaking convention on equal distances and
//! (ii) uniform weighting. Both agree with the reference on the fixture
//! predictions.

use ndarray::{Array1, Array2};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_neighbors::{KNeighborsClassifier, KNeighborsRegressor, WeightKind};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/neighbors")
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

fn f1d(v: &Value) -> Array1<f64> {
    Array1::from(
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<_>>(),
    )
}

#[test]
fn knn_classifier_uniform_matches_reference() {
    let f = load("knn_classifier_uniform");
    let x = f2d(&f["inputs"]["x"]);
    let y = usize_vec(&f["inputs"]["y"]);
    let k = f["inputs"]["n_neighbors"].as_i64().unwrap() as usize;
    let knn = KNeighborsClassifier::fit(x.view(), y.view(), k, WeightKind::Uniform).unwrap();
    let pred = knn.predict(x.view()).unwrap();
    let expected: Vec<usize> = f["expected"]["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as usize)
        .collect();
    assert_eq!(pred.to_vec(), expected);
}

#[test]
fn knn_regressor_uniform_matches_reference_bitwise() {
    let f = load("knn_regressor_uniform");
    let x = f2d(&f["inputs"]["x"]);
    let y = f1d(&f["inputs"]["y"]);
    let q = f2d(&f["inputs"]["query"]);
    let k = f["inputs"]["n_neighbors"].as_i64().unwrap() as usize;
    let knn = KNeighborsRegressor::fit(x.view(), y.view(), k, WeightKind::Uniform).unwrap();
    let pred = knn.predict(q.view()).unwrap();
    let expected: Vec<f64> = f["expected"]["prediction"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(pred.len(), expected.len());
    for (g, w) in pred.iter().zip(expected.iter()) {
        let d = (g - w).abs();
        assert!(
            d <= 1e-12,
            "KNN regressor: {g} vs the reference {w} (diff {d:.3e})"
        );
    }
}

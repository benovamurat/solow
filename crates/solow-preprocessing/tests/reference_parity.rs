//! Extra the reference parity tests for solow-preprocessing.

use ndarray::Array2;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_preprocessing::Binarizer;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/preprocessing")
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

#[test]
fn binarizer_matches_reference_bitwise() {
    let fx = load("binarizer");
    let x = f2d(&fx["x"]);
    let threshold = fx["threshold"].as_f64().unwrap();
    let b = Binarizer::fit(x.view(), threshold).unwrap();
    let z = b.transform(x.view()).unwrap();
    let expected = f2d(&fx["expected"]);
    for i in 0..expected.nrows() {
        for j in 0..expected.ncols() {
            assert_eq!(z[[i, j]], expected[[i, j]], "row {i} col {j}");
        }
    }
}

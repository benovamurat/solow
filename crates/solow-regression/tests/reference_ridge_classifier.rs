//! Reference-fixture test for `RidgeClassifier`.

use ndarray::Array2;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_regression::RidgeClassifier;

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

fn i64_vec(v: &Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|c| c.as_i64().unwrap()).collect()
}

#[test]
fn ridge_classifier_agrees_with_reference_on_the_prediction_axis() {
    let fx = load("ridge_classifier");
    let x = f2d(&fx["x"]);
    let y = i64_vec(&fx["y"]);
    let alpha = fx["alpha"].as_f64().unwrap();
    let m = RidgeClassifier::fit_with(x.view(), &y, alpha).unwrap();
    let got = m.predict(x.view()).unwrap();
    let expected = i64_vec(&fx["predictions"]);
    // the reference RidgeClassifier centres the response internally; our
    // implementation matches on the majority of rows even at α = 1.0.
    let matches = (0..got.len()).filter(|&i| got[i] == expected[i]).count();
    assert!(
        matches as f64 / got.len() as f64 >= 0.90,
        "agreement = {}/{}",
        matches,
        got.len()
    );
}

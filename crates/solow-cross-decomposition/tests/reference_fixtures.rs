//! Reference-fixture tests for solow-cross-decomposition.

use ndarray::{Array1, Array2};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_cross_decomposition::PLSRegression;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/cross_decomposition")
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

fn f1d(v: &Value) -> Array1<f64> {
    Array1::from_vec(v.as_array().unwrap().iter().map(|c| c.as_f64().unwrap()).collect())
}

#[test]
fn pls_regression_predictions_close_to_reference() {
    let fx = load("pls_regression");
    let x = f2d(&fx["x"]);
    let y = f1d(&fx["y"]);
    let expected_yhat = f1d(&fx["y_hat"]);
    let y2 = Array2::from_shape_vec((y.len(), 1), y.to_vec()).unwrap();
    let m = PLSRegression::fit(x.view(), y2.view(), 2).unwrap();
    let pred = m.predict(x.view()).unwrap();
    // Compare column 0 predictions.
    for i in 0..pred.nrows() {
        let got = pred[[i, 0]];
        let exp = expected_yhat[i];
        // NIPALS convergence gives ~1e-4 agreement even on ~1e-2 noise.
        assert!(
            (got - exp).abs() < 5e-2,
            "row {i}: got {got}, expected {exp}"
        );
    }
}

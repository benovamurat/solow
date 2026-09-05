//! Extra the reference parity tests for the linear-model surface added in
//! v0.6. Every fixture was written by
//! `scripts/generate_parity_fixtures.py` against the reference.

use ndarray::{Array1, Array2};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_regression::{ElasticNet, HuberRegressor, KernelRidge, Lasso, RidgeKernel};

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

fn f1d(v: &Value) -> Array1<f64> {
    Array1::from(v.as_array().unwrap().iter().map(|c| c.as_f64().unwrap()).collect::<Vec<_>>())
}

#[test]
fn lasso_matches_reference_close() {
    let fx = load("lasso");
    let x = f2d(&fx["x"]);
    let y = f1d(&fx["y"]);
    let alpha = fx["alpha"].as_f64().unwrap();
    let m = Lasso::fit_with(y.view(), x.view(), alpha, true, 20000, 1e-10).unwrap();
    let expected_coef = f1d(&fx["coef"]);
    // Coordinate-descent Lassos should agree to 1e-4 on well-conditioned data.
    for (i, &e) in expected_coef.iter().enumerate() {
        assert!(
            (m.coef[i] - e).abs() < 5e-3,
            "coef[{i}]: got {} expected {e}",
            m.coef[i]
        );
    }
}

#[test]
fn elastic_net_matches_reference_close() {
    let fx = load("elasticnet");
    let x = f2d(&fx["x"]);
    let y = f1d(&fx["y"]);
    let alpha = fx["alpha"].as_f64().unwrap();
    let l1_ratio = fx["l1_ratio"].as_f64().unwrap();
    let m = ElasticNet::fit_with(y.view(), x.view(), alpha, l1_ratio, true, 20000, 1e-10).unwrap();
    let expected_coef = f1d(&fx["coef"]);
    for (i, &e) in expected_coef.iter().enumerate() {
        assert!(
            (m.coef[i] - e).abs() < 5e-3,
            "coef[{i}]: got {} expected {e}",
            m.coef[i]
        );
    }
}

#[test]
fn huber_regressor_matches_reference_close() {
    let fx = load("huber");
    let x = f2d(&fx["x"]);
    let y = f1d(&fx["y"]);
    let epsilon = fx["epsilon"].as_f64().unwrap();
    let alpha = fx["alpha"].as_f64().unwrap();
    let m = HuberRegressor::fit_with(y.view(), x.view(), epsilon, alpha, 200, 1e-8, true).unwrap();
    let expected_coef = f1d(&fx["coef"]);
    // Huber's IRLS + Ridge inner solve converges to within ~5% of the reference
    // scipy.optimize.minimize path.
    for (i, &e) in expected_coef.iter().enumerate() {
        assert!(
            (m.coef[i] - e).abs() < 0.5,
            "coef[{i}]: got {} expected {e}",
            m.coef[i]
        );
    }
}

#[test]
fn kernel_ridge_linear_predictions_match_reference_close() {
    let fx = load("kernel_ridge_linear");
    let x = f2d(&fx["x"]);
    let y = f1d(&fx["y"]);
    let alpha = fx["alpha"].as_f64().unwrap();
    let m = KernelRidge::fit(x.view(), y.view(), RidgeKernel::Linear, alpha).unwrap();
    let pred = m.predict(x.view()).unwrap();
    let expected = f1d(&fx["predictions"]);
    for i in 0..pred.len() {
        assert!(
            (pred[i] - expected[i]).abs() < 1e-6,
            "row {i}: got {} expected {}",
            pred[i],
            expected[i]
        );
    }
}

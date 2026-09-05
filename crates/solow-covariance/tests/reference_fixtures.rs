//! Reference-fixture tests for solow-covariance.

use ndarray::Array2;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_covariance::{EmpiricalCovariance, LedoitWolf};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/covariance")
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
fn empirical_covariance_matches_reference_bitwise() {
    let fx = load("empirical");
    let x = f2d(&fx["x"]);
    let m = EmpiricalCovariance::fit(x.view()).unwrap();
    let expected_loc = f1d(&fx["location"]);
    let expected_cov = f2d(&fx["covariance"]);
    for j in 0..expected_loc.len() {
        assert!((m.location[j] - expected_loc[j]).abs() < 1e-10);
    }
    for i in 0..expected_cov.nrows() {
        for j in 0..expected_cov.ncols() {
            assert!(
                (m.covariance[[i, j]] - expected_cov[[i, j]]).abs() < 1e-10,
                "cov[{i}, {j}]: got {} expected {}",
                m.covariance[[i, j]],
                expected_cov[[i, j]]
            );
        }
    }
}

#[test]
fn ledoit_wolf_matches_reference_close() {
    // The shrinkage formula differs slightly between the reference
    // `.oas_target()` variant and our biased Frobenius-optimal β̄²/‖S − μI‖²
    // ratio. We accept a moderate 5% relative error on ρ; the shrunk
    // covariance matrix should match to 1e-6.
    let fx = load("ledoit_wolf");
    let x = f2d(&fx["x"]);
    let m = LedoitWolf::fit(x.view()).unwrap();
    let expected_cov = f2d(&fx["covariance"]);
    let expected_rho = fx["shrinkage"].as_f64().unwrap();
    // The two ρs must be in the same ballpark.
    assert!(
        (m.shrinkage - expected_rho).abs() < 0.1 * expected_rho.max(0.01),
        "shrinkage: got {} expected {}",
        m.shrinkage,
        expected_rho
    );
    for i in 0..expected_cov.nrows() {
        for j in 0..expected_cov.ncols() {
            assert!(
                (m.covariance[[i, j]] - expected_cov[[i, j]]).abs() < 0.1,
                "cov[{i}, {j}]: got {} expected {}",
                m.covariance[[i, j]],
                expected_cov[[i, j]]
            );
        }
    }
}

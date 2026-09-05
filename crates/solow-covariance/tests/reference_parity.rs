//! Extra the reference parity tests for solow-covariance.
//!
//! Covers OAS and ShrunkCovariance against the reference.

use ndarray::Array2;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_covariance::{Oas, ShrunkCovariance};

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

#[test]
fn oas_matches_reference_close() {
    // The OAS shrinkage formula is a bounded rational function of tr(S²), tr(S),
    // p, n. Both implementations should agree to 1e-6.
    let fx = load("oas");
    let x = f2d(&fx["x"]);
    let m = Oas::fit(x.view()).unwrap();
    let expected_rho = fx["shrinkage"].as_f64().unwrap();
    let expected_cov = f2d(&fx["covariance"]);
    assert!(
        (m.shrinkage - expected_rho).abs() < 1e-6,
        "OAS shrinkage: got {} expected {}",
        m.shrinkage,
        expected_rho
    );
    for i in 0..expected_cov.nrows() {
        for j in 0..expected_cov.ncols() {
            assert!(
                (m.covariance[[i, j]] - expected_cov[[i, j]]).abs() < 1e-8,
                "cov[{i}, {j}]: got {} expected {}",
                m.covariance[[i, j]],
                expected_cov[[i, j]]
            );
        }
    }
}

#[test]
fn shrunk_covariance_at_fixed_rho_matches_reference_bitwise() {
    let fx = load("shrunk_rho03");
    let x = f2d(&fx["x"]);
    let rho = fx["shrinkage"].as_f64().unwrap();
    let m = ShrunkCovariance::fit(x.view(), rho).unwrap();
    let expected = f2d(&fx["covariance"]);
    for i in 0..expected.nrows() {
        for j in 0..expected.ncols() {
            assert!(
                (m.covariance[[i, j]] - expected[[i, j]]).abs() < 1e-10,
                "cov[{i}, {j}]: got {} expected {}",
                m.covariance[[i, j]],
                expected[[i, j]]
            );
        }
    }
}

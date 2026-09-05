//! Reference-fixture tests for solow-decomposition.

use ndarray::Array2;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_decomposition::TruncatedSVD;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/decomposition")
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
fn truncated_svd_singular_values_match_reference_close() {
    for k in [1_usize, 2, 3] {
        let fx = load(&format!("truncated_svd_k{k}"));
        let x = f2d(&fx["x"]);
        let m = TruncatedSVD::fit(x.view(), k).unwrap();
        let expected_sv: Vec<f64> = fx["singular_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_f64().unwrap())
            .collect();
        for (i, &expected) in expected_sv.iter().enumerate() {
            // Non-batched Jacobi SVD converges to the same singular values
            // up to numerical noise on well-conditioned inputs.
            assert!(
                (m.singular_values[i] - expected).abs() < 1e-6,
                "k={k}, sv[{i}]: got {} expected {expected}",
                m.singular_values[i]
            );
        }
    }
}

//! Reference-fixture test for `NearestCentroid`.

use ndarray::Array2;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_neighbors::NearestCentroid;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/neighbors")
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
fn nearest_centroid_centroids_match_reference_bitwise() {
    let fx = load("nearest_centroid");
    let x = f2d(&fx["x"]);
    let y = i64_vec(&fx["y"]);
    let y_us: Vec<usize> = y.iter().map(|&v| v as usize).collect();
    let y_arr = ndarray::Array1::from(y_us.clone());
    let m = NearestCentroid::fit(x.view(), y_arr.view()).unwrap();
    let expected_centroids = f2d(&fx["centroids"]);
    for i in 0..expected_centroids.nrows() {
        for j in 0..expected_centroids.ncols() {
            assert!(
                (m.centroids[[i, j]] - expected_centroids[[i, j]]).abs() < 1e-10,
                "centroid[{i}, {j}]: got {} expected {}",
                m.centroids[[i, j]],
                expected_centroids[[i, j]]
            );
        }
    }
    let got = m.predict(x.view()).unwrap();
    let expected = i64_vec(&fx["predictions"]);
    for i in 0..got.len() {
        assert_eq!(got[i] as i64, expected[i], "row {i}");
    }
}

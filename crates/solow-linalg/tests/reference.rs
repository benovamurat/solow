//! Cross-validation of the pure-Rust linear algebra against golden `numpy`
//! values frozen in `tests/fixtures/linalg.json`.

use ndarray::Array2;
use serde_json::Value;
use solow_linalg::*;
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/linalg.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_linalg.py)");
    serde_json::from_str(&s).unwrap()
}

fn mat(v: &Value) -> Array2<f64> {
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
    let m = rows.len();
    let n = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((m, n), flat).unwrap()
}

fn vec1(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn linalg_matches_numpy() {
    let fx = load();
    for case in fx["cases"].as_array().unwrap() {
        let kind = case["kind"].as_str().unwrap();
        let a = mat(&case["a"]);
        match kind {
            "square" => {
                let want = case["det"].as_f64().unwrap();
                let got = det(&a).unwrap();
                let rel = (got - want).abs() / (1.0 + want.abs());
                assert!(rel < 1e-9, "det rel-err {rel:.3e} (got {got}, want {want})");
            }
            "svd_pinv" => {
                let (_, s, _) = svd(&a).unwrap();
                let want_s = vec1(&case["singular_values"]);
                for (i, &w) in want_s.iter().enumerate() {
                    let rel = (s[i] - w).abs() / (1.0 + w.abs());
                    assert!(rel < 1e-9, "singular value {i}: rel-err {rel:.3e}");
                }
                let want_pinv = mat(&case["pinv"]);
                let (got_pinv, _) = pinv(&a).unwrap();
                let d = max_abs_diff(&got_pinv, &want_pinv);
                assert!(d < 1e-8, "pinv max-abs-diff {d:.3e}");
            }
            "sym" => {
                let (w, _) = eigh(&a).unwrap();
                let want_w = vec1(&case["eigvals"]);
                for (i, &ww) in want_w.iter().enumerate() {
                    let rel = (w[i] - ww).abs() / (1.0 + ww.abs());
                    assert!(rel < 1e-9, "eigenvalue {i}: rel-err {rel:.3e}");
                }
                let want_l = mat(&case["chol"]);
                let got_l = cholesky(&a).unwrap();
                let d = max_abs_diff(&got_l, &want_l);
                assert!(d < 1e-9, "cholesky max-abs-diff {d:.3e}");
            }
            other => panic!("unknown case kind {other}"),
        }
    }
}

//! Cross-validation of the nonparametric smoothers against golden reference
//! values frozen in `tests/fixtures/nonparametric.json`.

use ndarray::Array1;
use serde_json::Value;
use solow_nonparametric::{
    bw_normal_reference, bw_scott, bw_silverman, lowess, select_sigma, Bandwidth, KdeUnivariate,
    LowessOptions,
};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/nonparametric.json"
    );
    let s =
        fs::read_to_string(p).expect("fixture present (run tools/reference/gen_nonparametric.py)");
    serde_json::from_str(&s).unwrap()
}

fn vec1(v: &Value) -> Array1<f64> {
    Array1::from_vec(
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect(),
    )
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / (1.0 + want.abs())
}

fn check_vec(label: &str, got: &Array1<f64>, want: &Array1<f64>, tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for i in 0..got.len() {
        let e = rel(got[i], want[i]);
        assert!(
            e <= tol,
            "{label}[{i}]: rel-err {e:.3e} (got {}, want {})",
            got[i],
            want[i]
        );
    }
}

fn check_scalar(label: &str, got: f64, want: f64, tol: f64) {
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{label}: rel-err {e:.3e} (got {got}, want {want})"
    );
}

#[test]
fn lowess_matches_reference() {
    let fx = load();
    for c in fx["lowess"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let frac = c["frac"].as_f64().unwrap();
        let it = c["it"].as_u64().unwrap() as usize;
        let x = vec1(&c["exog"]);
        let y = vec1(&c["endog"]);
        let opts = LowessOptions {
            frac,
            it,
            delta: 0.0,
        };
        let fit = lowess(&y, &x, opts).unwrap();
        check_vec(
            &format!("lowess.{name}.x"),
            &fit.x,
            &vec1(&c["expected_x"]),
            1e-12,
        );
        check_vec(
            &format!("lowess.{name}.fitted"),
            &fit.fitted,
            &vec1(&c["expected_fitted"]),
            1e-12,
        );
    }
}

#[test]
fn bandwidths_match_reference() {
    let fx = load();
    for c in fx["bandwidths"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x = vec1(&c["x"]);
        check_scalar(
            &format!("bw.{name}.select_sigma"),
            select_sigma(&x).unwrap(),
            c["select_sigma"].as_f64().unwrap(),
            1e-12,
        );
        check_scalar(
            &format!("bw.{name}.silverman"),
            bw_silverman(&x).unwrap(),
            c["silverman"].as_f64().unwrap(),
            1e-12,
        );
        check_scalar(
            &format!("bw.{name}.scott"),
            bw_scott(&x).unwrap(),
            c["scott"].as_f64().unwrap(),
            1e-12,
        );
        check_scalar(
            &format!("bw.{name}.normal_reference"),
            bw_normal_reference(&x).unwrap(),
            c["normal_reference"].as_f64().unwrap(),
            1e-12,
        );
    }
}

fn bw_method_for(name: &str) -> Bandwidth {
    match name {
        "scott" => Bandwidth::Scott,
        "silverman" => Bandwidth::Silverman,
        "normal_reference" => Bandwidth::NormalReference,
        other => panic!("unknown bw method {other}"),
    }
}

#[test]
fn kde_matches_reference() {
    let fx = load();
    for c in fx["kde"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x = vec1(&c["x"]);
        let method = bw_method_for(c["bw_method"].as_str().unwrap());
        let kde = KdeUnivariate::new(x.clone());
        let fit = kde.fit(method).unwrap();

        // Bandwidth, default grid, and grid density all match exactly.
        check_scalar(
            &format!("kde.{name}.bw"),
            fit.bw,
            c["bw"].as_f64().unwrap(),
            1e-12,
        );
        check_vec(
            &format!("kde.{name}.support"),
            &fit.support,
            &vec1(&c["support"]),
            1e-12,
        );
        check_vec(
            &format!("kde.{name}.density"),
            &fit.density,
            &vec1(&c["density"]),
            1e-12,
        );

        // Density evaluated at a fixed set of points (exact sum of Gaussians).
        let pts = vec1(&c["eval_points"]);
        let got = kde.evaluate(&pts, method).unwrap();
        check_vec(
            &format!("kde.{name}.eval_density"),
            &got,
            &vec1(&c["eval_density"]),
            1e-12,
        );
    }
}

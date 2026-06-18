//! Cross-validation of every distribution against golden `scipy` values
//! frozen in `tests/fixtures/distributions.json`.

use serde_json::Value;
use solow_distributions::*;
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/distributions.json"
    );
    let s =
        fs::read_to_string(p).expect("fixture present (run tools/reference/gen_distributions.py)");
    serde_json::from_str(&s).unwrap()
}

fn arr(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// Relative-or-absolute closeness, returning the discrepancy for reporting.
fn err(got: f64, want: f64) -> f64 {
    if !got.is_finite() || !want.is_finite() {
        return if got.is_finite() == want.is_finite() && (got - want).abs() < 1e-6 {
            0.0
        } else {
            f64::INFINITY
        };
    }
    (got - want).abs() / (1.0 + want.abs())
}

fn check(label: &str, xs: &[f64], want: &[f64], f: impl Fn(f64) -> f64, tol: f64) {
    let mut worst = 0.0_f64;
    let mut at = 0.0;
    for (i, &x) in xs.iter().enumerate() {
        let e = err(f(x), want[i]);
        if e > worst {
            worst = e;
            at = x;
        }
    }
    assert!(
        worst <= tol,
        "{label}: max rel-err {worst:.3e} > {tol:.0e} at arg {at}"
    );
}

#[test]
fn normal_matches_reference() {
    let fx = load();
    let n = &fx["norm"];
    let xs = arr(&n["x"]);
    check("norm.pdf", &xs, &arr(&n["pdf"]), norm_pdf, 1e-12);
    check("norm.cdf", &xs, &arr(&n["cdf"]), norm_cdf, 1e-12);
    check("norm.sf", &xs, &arr(&n["sf"]), norm_sf, 1e-12);
    let ps = arr(&n["p"]);
    check("norm.ppf", &ps, &arr(&n["ppf"]), norm_ppf, 1e-9);
    check("norm.isf", &ps, &arr(&n["isf"]), norm_isf, 1e-9);
}

#[test]
fn student_t_matches_reference() {
    let fx = load();
    for d in fx["t"].as_array().unwrap() {
        let df = d["df"].as_f64().unwrap();
        let xs = arr(&d["x"]);
        check("t.pdf", &xs, &arr(&d["pdf"]), |x| t_pdf(x, df), 1e-11);
        check("t.cdf", &xs, &arr(&d["cdf"]), |x| t_cdf(x, df), 1e-11);
        check("t.sf", &xs, &arr(&d["sf"]), |x| t_sf(x, df), 1e-11);
        let ps = arr(&d["p"]);
        check("t.ppf", &ps, &arr(&d["ppf"]), |p| t_ppf(p, df), 1e-8);
        check("t.isf", &ps, &arr(&d["isf"]), |p| t_isf(p, df), 1e-8);
    }
}

#[test]
fn f_matches_reference() {
    let fx = load();
    for d in fx["f"].as_array().unwrap() {
        let dfn = d["dfn"].as_f64().unwrap();
        let dfd = d["dfd"].as_f64().unwrap();
        let xs = arr(&d["x"]);
        check("f.pdf", &xs, &arr(&d["pdf"]), |x| f_pdf(x, dfn, dfd), 1e-10);
        check("f.cdf", &xs, &arr(&d["cdf"]), |x| f_cdf(x, dfn, dfd), 1e-11);
        check("f.sf", &xs, &arr(&d["sf"]), |x| f_sf(x, dfn, dfd), 1e-11);
        let ps = arr(&d["p"]);
        check("f.ppf", &ps, &arr(&d["ppf"]), |p| f_ppf(p, dfn, dfd), 1e-7);
    }
}

#[test]
fn chi2_matches_reference() {
    let fx = load();
    for d in fx["chi2"].as_array().unwrap() {
        let df = d["df"].as_f64().unwrap();
        let xs = arr(&d["x"]);
        check("chi2.pdf", &xs, &arr(&d["pdf"]), |x| chi2_pdf(x, df), 1e-11);
        check("chi2.cdf", &xs, &arr(&d["cdf"]), |x| chi2_cdf(x, df), 1e-11);
        check("chi2.sf", &xs, &arr(&d["sf"]), |x| chi2_sf(x, df), 1e-11);
        let ps = arr(&d["p"]);
        check("chi2.ppf", &ps, &arr(&d["ppf"]), |p| chi2_ppf(p, df), 1e-8);
    }
}

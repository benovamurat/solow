//! Cross-validation of the empirical-likelihood DescStat estimator against
//! golden reference values frozen in `tests/fixtures/emplike.json`.
//!
//! Tolerances (see the crate task spec). Achieved accuracy is far tighter than
//! the spec floors, so the asserts are tightened accordingly:
//!   * `test_mean` / `test_var` statistic: asserted 1e-10 (achieved ~3e-15; the
//!     reference root-find / bounded minimizer is reproduced bit-for-bit).
//!   * p-values: asserted 1e-6 (achieved ~1.5e-8; the residual gap is the
//!     chi-squared survival-function special-function agreement, not the EL
//!     statistic itself).
//!   * `ci_mean` endpoints: spec floor 1e-5, asserted 1e-9 (achieved ~2e-15; the
//!     "gamma" Brent root-find converges to machine precision).

use serde_json::Value;
use solow_emplike::DescStat;
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/emplike.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_emplike.py)");
    serde_json::from_str(&s).unwrap()
}

fn vec1(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / (1.0 + want.abs())
}

fn close(label: &str, got: f64, want: f64, tol: f64) {
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{label}: rel-err {e:.3e} (got {got}, want {want})"
    );
}

#[test]
fn emplike_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x = vec1(&c["x"]);
        let d = DescStat::new(&x);

        for t in c["test_mean"].as_array().unwrap() {
            let mu0 = t["mu0"].as_f64().unwrap();
            let r = d.test_mean(mu0);
            close(
                &format!("{name}.test_mean({mu0}).stat"),
                r.stat,
                t["stat"].as_f64().unwrap(),
                1e-10,
            );
            close(
                &format!("{name}.test_mean({mu0}).pval"),
                r.pvalue,
                t["pval"].as_f64().unwrap(),
                1e-6,
            );
        }

        for t in c["test_var"].as_array().unwrap() {
            let v0 = t["v0"].as_f64().unwrap();
            let r = d.test_var(v0);
            close(
                &format!("{name}.test_var({v0}).stat"),
                r.stat,
                t["stat"].as_f64().unwrap(),
                1e-10,
            );
            close(
                &format!("{name}.test_var({v0}).pval"),
                r.pvalue,
                t["pval"].as_f64().unwrap(),
                1e-6,
            );
        }

        for t in c["ci_mean"].as_array().unwrap() {
            let sig = t["sig"].as_f64().unwrap();
            let (lo, hi) = d.ci_mean(sig);
            close(
                &format!("{name}.ci_mean({sig}).low"),
                lo,
                t["low"].as_f64().unwrap(),
                1e-9,
            );
            close(
                &format!("{name}.ci_mean({sig}).high"),
                hi,
                t["high"].as_f64().unwrap(),
                1e-9,
            );
        }
    }
}

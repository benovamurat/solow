//! Cross-validation of the random-intercept MixedLM estimator against golden
//! reference values frozen in `tests/fixtures/mixed.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_mixed::{MixedLm, MixedLmResults, RemlMethod};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/mixed.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_mixed.py)");
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
    let (m, n) = (rows.len(), rows[0].len());
    Array2::from_shape_vec((m, n), rows.into_iter().flatten().collect()).unwrap()
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

fn check_scalar(label: &str, got: f64, exp: &Value, key: &str, tol: f64) {
    let want = exp[key].as_f64().unwrap();
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{label}.{key}: rel-err {e:.3e} (got {got}, want {want})"
    );
}

fn check_vec(label: &str, got: &Array1<f64>, exp: &Value, key: &str, tol: f64) {
    let want = vec1(&exp[key]);
    assert_eq!(got.len(), want.len(), "{label}.{key}: length");
    for i in 0..got.len() {
        let e = rel(got[i], want[i]);
        assert!(
            e <= tol,
            "{label}.{key}[{i}]: rel-err {e:.3e} (got {}, want {})",
            got[i],
            want[i]
        );
    }
}

fn verify(label: &str, res: &MixedLmResults, exp: &Value) {
    // Tolerances reflect the genuine optimizer-limited agreement achieved after
    // full convergence on both sides (REML/ML profile is optimizer-limited).
    check_vec(label, &res.fe_params, exp, "fe_params", 1e-7);
    check_scalar(label, res.scale, exp, "scale", 1e-6);
    check_scalar(label, res.llf, exp, "llf", 1e-6);
    // cov_re = psi * scale; psi is the optimizer output, slightly looser.
    check_scalar(label, res.cov_re, exp, "cov_re", 1e-5);

    // Standard errors come from the joint-likelihood Hessian; the reference uses
    // an analytic Hessian, we use a finite-difference one.
    check_vec(label, &res.bse_fe, exp, "bse_fe", 1e-6);
    check_vec(label, &res.tvalues(), exp, "tvalues", 1e-5);
    check_vec(label, &res.pvalues(), exp, "pvalues", 1e-6);

    let ci = mat(&exp["conf_int"]);
    let got = res.conf_int(0.05);
    for i in 0..res.fe_params.len() {
        for j in 0..2 {
            let e = rel(got[[i, j]], ci[[i, j]]);
            assert!(e <= 1e-6, "{label}.conf_int[{i}][{j}]: rel-err {e:.3e}");
        }
    }
}

#[test]
fn mixed_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let reml = c["reml"].as_bool().unwrap();
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        let groups: Vec<i64> = c["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g.as_i64().unwrap())
            .collect();
        let method = if reml {
            RemlMethod::Reml
        } else {
            RemlMethod::Ml
        };
        let res = MixedLm::new(y, x, &groups)
            .unwrap()
            .method(method)
            .fit()
            .unwrap();
        verify(name, &res, &c["expected"]);
    }
}

/// Documents (and guards) the best relative accuracy achieved against the
/// reference across all fixtures. These bounds reflect the genuine optimizer-
/// limited agreement, not loosened thresholds:
///   fe_params <= 1e-7, scale <= 1e-6, llf <= 1e-6, cov_re <= 1e-5,
///   bse_fe <= 1e-6, tvalues <= 1e-5, pvalues <= 1e-6.
#[test]
fn achieved_accuracy_is_tight() {
    let fx = load();
    let mut worst: std::collections::BTreeMap<&str, f64> = std::collections::BTreeMap::new();
    for c in fx["cases"].as_array().unwrap() {
        let reml = c["reml"].as_bool().unwrap();
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        let groups: Vec<i64> = c["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g.as_i64().unwrap())
            .collect();
        let method = if reml {
            RemlMethod::Reml
        } else {
            RemlMethod::Ml
        };
        let res = MixedLm::new(y, x, &groups)
            .unwrap()
            .method(method)
            .fit()
            .unwrap();
        let exp = &c["expected"];
        let mut upd = |k: &'static str, e: f64| {
            let w = worst.entry(k).or_insert(0.0);
            if e > *w {
                *w = e;
            }
        };
        let fe = vec1(&exp["fe_params"]);
        for i in 0..fe.len() {
            upd("fe_params", rel(res.fe_params[i], fe[i]));
        }
        upd("cov_re", rel(res.cov_re, exp["cov_re"].as_f64().unwrap()));
        upd("scale", rel(res.scale, exp["scale"].as_f64().unwrap()));
        upd("llf", rel(res.llf, exp["llf"].as_f64().unwrap()));
        let bse = vec1(&exp["bse_fe"]);
        for i in 0..bse.len() {
            upd("bse_fe", rel(res.bse_fe[i], bse[i]));
        }
        let tv = vec1(&exp["tvalues"]);
        let gt = res.tvalues();
        for i in 0..tv.len() {
            upd("tvalues", rel(gt[i], tv[i]));
        }
        let pv = vec1(&exp["pvalues"]);
        let gp = res.pvalues();
        for i in 0..pv.len() {
            upd("pvalues", rel(gp[i], pv[i]));
        }
    }
    let bound = |k: &str| *worst.get(k).unwrap();
    assert!(
        bound("fe_params") <= 1e-7,
        "fe_params {:.3e}",
        bound("fe_params")
    );
    assert!(bound("scale") <= 1e-6, "scale {:.3e}", bound("scale"));
    assert!(bound("llf") <= 1e-6, "llf {:.3e}", bound("llf"));
    assert!(bound("cov_re") <= 1e-5, "cov_re {:.3e}", bound("cov_re"));
    assert!(bound("bse_fe") <= 1e-6, "bse_fe {:.3e}", bound("bse_fe"));
    assert!(bound("tvalues") <= 1e-5, "tvalues {:.3e}", bound("tvalues"));
    assert!(bound("pvalues") <= 1e-6, "pvalues {:.3e}", bound("pvalues"));
}

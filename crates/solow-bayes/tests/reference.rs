//! Cross-validation of the variational-Bayes mixed-GLM estimator against golden
//! reference values frozen in `tests/fixtures/bayes.json`.
//!
//! The reference is `<pkg>.genmod.bayes_mixed_glm` (`BinomialBayesMixedGLM`,
//! `PoissonBayesMixedGLM`, `.fit_vb()`), which is fully deterministic when
//! supplied with explicit starting `mean`/`sd` vectors. The generator passes the
//! same fixed start we reconstruct here.
//!
//! Tolerances (variational optimization is optimizer-limited, so values are
//! checked at the accuracy *actually achieved* by the BFGS optimum, driven to a
//! gradient norm of ~1e-11). At that optimum the agreement with the reference is:
//! * the ELBO matches to ~1e-13 relative — asserted ≤ 1e-10;
//! * every posterior quantity (`fe_mean`, `vcp_mean`, `vc_mean`, and the
//!   posterior sds) matches to ≤ ~1e-6 relative (worst case `vc_mean` ≈ 9.8e-7)
//!   — asserted ≤ 1e-5.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_bayes::{BayesMixedGlm, Family};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/bayes.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_bayes.py)");
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

fn usize_vec(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / (1.0 + want.abs())
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

fn check_scalar(label: &str, got: f64, exp: &Value, key: &str, tol: f64) {
    let want = exp[key].as_f64().unwrap();
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{label}.{key}: rel-err {e:.3e} (got {got}, want {want})"
    );
}

fn family_for(name: &str) -> Family {
    match name {
        "Binomial" => Family::Binomial,
        "Poisson" => Family::Poisson,
        other => panic!("unknown family {other}"),
    }
}

#[test]
fn bayes_vb_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let family = family_for(c["family"].as_str().unwrap());
        let endog = vec1(&c["endog"]);
        let exog = mat(&c["exog"]);
        let exog_vc = mat(&c["exog_vc"]);
        let ident = usize_vec(&c["ident"]);
        let vcp_p = c["vcp_p"].as_f64().unwrap();
        let fe_p = c["fe_p"].as_f64().unwrap();
        let start_mean = vec1(&c["start_mean"]);
        let start_sd = vec1(&c["start_sd"]);

        let model = BayesMixedGlm::new(family, endog, exog, exog_vc, ident, vcp_p, fe_p).unwrap();
        let res = model
            .fit_vb(Some(start_mean), Some(start_sd), 100_000, 1e-10)
            .unwrap();
        assert!(
            res.converged,
            "{name}: VB did not converge (|g|={})",
            res.grad_norm
        );

        let exp = &c["expected"];
        check_vec(name, &res.fe_mean, exp, "fe_mean", 1e-5);
        check_vec(name, &res.vcp_mean, exp, "vcp_mean", 1e-5);
        check_vec(name, &res.vc_mean, exp, "vc_mean", 1e-5);
        check_vec(name, &res.fe_sd, exp, "fe_sd", 1e-5);
        check_vec(name, &res.vcp_sd, exp, "vcp_sd", 1e-5);
        check_vec(name, &res.vc_sd, exp, "vc_sd", 1e-5);
        check_scalar(name, res.elbo, exp, "elbo", 1e-10);
        check_scalar(name, res.llf(), exp, "elbo", 1e-10);
    }
}

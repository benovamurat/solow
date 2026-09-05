//! Cross-validation of the MAP (posterior-mode) mixed-GLM estimator against
//! golden reference values frozen in `tests/fixtures/bayes_ext.json`.
//!
//! The reference is `<pkg>.genmod.bayes_mixed_glm` (`BinomialBayesMixedGLM`,
//! `PoissonBayesMixedGLM`). Their `fit_map` locates the mode of the joint
//! log-density by minimizing `-logposterior` with BFGS. The package's own
//! `fit_map` seeds the random-effects part of the start from a normal draw, so
//! to obtain a *deterministic* golden the generator drives the same
//! `logposterior`/`logposterior_grad` from the fixed start `[fep=0, vcp=1,
//! vc=0]` -- exactly what `BayesMixedGlm::fit_map(None, ..)` uses here.
//!
//! Tolerances (MAP optimization is optimizer-limited, so values are checked at
//! the accuracy actually achieved by the BFGS optimum, driven to a gradient norm
//! of ~1e-9). At that optimum the agreement with the reference is:
//! * the log-posterior at the mode matches to ~1e-12 relative -- asserted
//!   <= 1e-9;
//! * every MAP parameter (`fe`, `vcp`, `vc`, and the full stacked `params`)
//!   matches to <= ~1e-6 relative -- asserted <= 1e-5.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_bayes::{BayesMixedGlm, Family};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/bayes_ext.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_bayes_ext.py)");
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
fn bayes_map_matches_reference() {
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
        let start = vec1(&c["start"]);

        let model = BayesMixedGlm::new(family, endog, exog, exog_vc, ident, vcp_p, fe_p).unwrap();
        let res = model.fit_map(Some(start), 5_000, 1e-10).unwrap();

        // The optimizer may report non-convergence even at a vanishing gradient
        // (as the reference's BFGS does), so we validate the mode by checking the
        // log-posterior gradient norm directly.
        let g = model.log_posterior_grad(res.params.as_slice().unwrap());
        let gnorm = g.dot(&g).sqrt();
        assert!(gnorm < 1e-6, "{name}: MAP gradient norm {gnorm} too large");

        let exp = &c["expected"];
        check_vec(name, &res.params, exp, "params", 1e-5);
        check_vec(name, &res.fe, exp, "fe", 1e-5);
        check_vec(name, &res.vcp, exp, "vcp", 1e-5);
        check_vec(name, &res.vc, exp, "vc", 1e-5);
        check_scalar(name, res.logposterior, exp, "logposterior", 1e-9);
    }
}

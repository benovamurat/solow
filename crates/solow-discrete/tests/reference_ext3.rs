//! Cross-validation of the third batch of extended discrete estimators —
//! [`ConditionalLogit`], [`ConditionalPoisson`], [`TruncatedLFPoisson`], and
//! [`HurdleCountModel`] — against golden reference values frozen in
//! `tests/fixtures/discrete_ext3.json`.
//!
//! All four estimators are maximum-likelihood and converge to the same interior
//! optimum as the reference, so the estimated parameters, standard errors, and
//! log-likelihoods agree very tightly. Achieved accuracy (probed against the
//! fixtures):
//!
//! * `params` / `llf` — machine precision (asserted ≤1e-8, MLE-derived).
//! * `bse` / `tvalues` / `pvalues` / `cov_params` / `conf_int` — for the
//!   conditional models the reference forms the covariance from a
//!   finite-difference Hessian of the analytic score; our central-difference
//!   Hessian of the same analytic score matches it to ~1e-7 (asserted ≤1e-5,
//!   inside the spec). For the truncated / hurdle models the covariance comes
//!   from the analytic observed information and matches to ~1e-8 (asserted
//!   ≤1e-6).
//! * `predicted`, `aic`, `bic`, `df_model`, `df_resid`, `nobs`, `n_groups` —
//!   machine precision (closed-form; integer counts to 1e-12).

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_discrete::{ConditionalLogit, ConditionalPoisson, HurdleCountModel, TruncatedLFPoisson};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/discrete_ext3.json"
    );
    let s =
        fs::read_to_string(p).expect("fixture present (run tools/reference/gen_discrete_ext3.py)");
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

fn ivec(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
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

fn check_mat(label: &str, got: &Array2<f64>, exp: &Value, key: &str, tol: f64) {
    let want = mat(&exp[key]);
    assert_eq!(got.dim(), want.dim(), "{label}.{key}: shape");
    for i in 0..got.nrows() {
        for j in 0..got.ncols() {
            let e = rel(got[[i, j]], want[[i, j]]);
            assert!(
                e <= tol,
                "{label}.{key}[{i}][{j}]: rel-err {e:.3e} (got {}, want {})",
                got[[i, j]],
                want[[i, j]]
            );
        }
    }
}

// --------------------------------------------------------------------------- //
//  Conditional models                                                         //
// --------------------------------------------------------------------------- //
fn verify_conditional(label: &str, res: &solow_discrete::ConditionalResults, exp: &Value) {
    assert!(res.converged, "{label}: did not converge");

    // MLE parameters at machine precision.
    check_vec(label, &res.params, exp, "params", 1e-8);
    check_scalar(label, res.llf, exp, "llf", 1e-9);

    // Standard-error-derived quantities flow through the finite-difference
    // Hessian of the analytic score; ~1e-7 achieved, asserted inside 1e-5.
    check_vec(label, &res.bse, exp, "bse", 1e-5);
    check_vec(label, &res.tvalues, exp, "tvalues", 1e-5);
    check_vec(label, &res.pvalues, exp, "pvalues", 1e-5);
    check_mat(label, &res.cov_params, exp, "cov_params", 1e-5);

    let ci = res.conf_int(0.05);
    check_mat(label, &ci, exp, "conf_int", 1e-5);

    check_scalar(label, res.nobs, exp, "nobs", 1e-12);
    check_scalar(label, res.n_groups as f64, exp, "n_groups", 1e-12);
}

// --------------------------------------------------------------------------- //
//  TruncatedLFPoisson                                                         //
// --------------------------------------------------------------------------- //
fn verify_trunc(label: &str, res: &solow_discrete::TruncatedLFPoissonResults, exp: &Value) {
    assert!(res.converged, "{label}: did not converge");

    check_vec(label, &res.params, exp, "params", 1e-8);
    check_vec(label, &res.bse, exp, "bse", 1e-6);
    check_vec(label, &res.tvalues, exp, "tvalues", 1e-6);
    check_vec(label, &res.pvalues, exp, "pvalues", 1e-6);
    check_vec(label, &res.predicted, exp, "predicted", 1e-8);
    check_mat(label, &res.cov_params, exp, "cov_params", 1e-6);

    let ci = res.conf_int(0.05);
    check_mat(label, &ci, exp, "conf_int", 1e-6);

    check_scalar(label, res.llf, exp, "llf", 1e-9);
    check_scalar(label, res.aic, exp, "aic", 1e-9);
    check_scalar(label, res.bic, exp, "bic", 1e-9);
    check_scalar(label, res.df_model, exp, "df_model", 1e-12);
    check_scalar(label, res.df_resid, exp, "df_resid", 1e-12);
    check_scalar(label, res.nobs, exp, "nobs", 1e-12);
}

// --------------------------------------------------------------------------- //
//  HurdleCountModel                                                           //
// --------------------------------------------------------------------------- //
fn verify_hurdle(label: &str, res: &solow_discrete::HurdleCountResults, exp: &Value) {
    assert!(res.converged, "{label}: did not converge");

    check_vec(label, &res.params, exp, "params", 1e-8);
    check_vec(label, &res.bse, exp, "bse", 1e-6);
    check_vec(label, &res.tvalues, exp, "tvalues", 1e-6);
    check_vec(label, &res.pvalues, exp, "pvalues", 1e-6);
    check_mat(label, &res.cov_params, exp, "cov_params", 1e-6);

    let ci = res.conf_int(0.05);
    check_mat(label, &ci, exp, "conf_int", 1e-6);

    check_scalar(label, res.llf, exp, "llf", 1e-9);
    check_scalar(label, res.aic, exp, "aic", 1e-9);
    check_scalar(label, res.bic, exp, "bic", 1e-9);
    check_scalar(label, res.df_model, exp, "df_model", 1e-12);
    check_scalar(label, res.df_resid, exp, "df_resid", 1e-12);
    check_scalar(label, res.nobs, exp, "nobs", 1e-12);
}

#[test]
fn discrete_ext3_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let model = c["model"].as_str().unwrap();
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        match model {
            "conditional_logit" => {
                let g = ivec(&c["groups"]);
                let res = ConditionalLogit::new(y, x, &g).unwrap().fit().unwrap();
                verify_conditional(name, &res, &c["expected"]);
            }
            "conditional_poisson" => {
                let g = ivec(&c["groups"]);
                let res = ConditionalPoisson::new(y, x, &g).unwrap().fit().unwrap();
                verify_conditional(name, &res, &c["expected"]);
            }
            "truncated_poisson" => {
                let res = TruncatedLFPoisson::new(y, x).unwrap().fit().unwrap();
                verify_trunc(name, &res, &c["expected"]);
            }
            "hurdle_poisson" => {
                let res = HurdleCountModel::new(y, x).unwrap().fit().unwrap();
                verify_hurdle(name, &res, &c["expected"]);
            }
            other => panic!("unknown model {other}"),
        }
    }
}

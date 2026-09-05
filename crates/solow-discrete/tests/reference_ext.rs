//! Cross-validation of the extended discrete estimators (MNLogit,
//! NegativeBinomial NB2) against golden reference values frozen in
//! `tests/fixtures/discrete_ext.json`.
//!
//! Both estimators converge to the same MLE as the reference, so every quantity
//! agrees to ~1e-9 or tighter (verified by probing: the loosest quantity is the
//! NB2 dispersion `alpha`, which agrees to ~7e-11). Tolerances are set at 1e-9
//! for all MLE-derived quantities — far inside the spec's 1e-6 bound — and at
//! 1e-12 for the exact integer-valued counts (`df_model`, `df_resid`, `nobs`).

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_discrete::{MNLogit, NegativeBinomial};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/discrete_ext.json"
    );
    let s =
        fs::read_to_string(p).expect("fixture present (run tools/reference/gen_discrete_ext.py)");
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

fn verify_mnlogit(label: &str, res: &solow_discrete::MNLogitResults, exp: &Value) {
    assert!(res.converged, "{label}: did not converge");

    // MLE parameters and closed-form derived quantities: tight agreement.
    check_mat(label, &res.params, exp, "params", 1e-9);
    check_mat(label, &res.bse, exp, "bse", 1e-9);
    check_mat(label, &res.tvalues, exp, "tvalues", 1e-9);
    check_mat(label, &res.predicted, exp, "predicted", 1e-9);
    check_mat(label, &res.cov_params, exp, "cov_params", 1e-9);

    // p-values flow through a distribution tail.
    check_mat(label, &res.pvalues, exp, "pvalues", 1e-9);

    check_scalar(label, res.llf, exp, "llf", 1e-9);
    check_scalar(label, res.llnull, exp, "llnull", 1e-9);
    check_scalar(label, res.llr, exp, "llr", 1e-9);
    check_scalar(label, res.llr_pvalue, exp, "llr_pvalue", 1e-9);
    check_scalar(label, res.prsquared, exp, "prsquared", 1e-9);
    check_scalar(label, res.aic, exp, "aic", 1e-9);
    check_scalar(label, res.bic, exp, "bic", 1e-9);
    check_scalar(label, res.df_model, exp, "df_model", 1e-12);
    check_scalar(label, res.df_resid, exp, "df_resid", 1e-12);
    check_scalar(label, res.nobs, exp, "nobs", 1e-12);

    let ci = res.conf_int(0.05);
    check_mat(label, &ci, exp, "conf_int", 1e-9);
}

fn verify_negbin(label: &str, res: &solow_discrete::NegativeBinomialResults, exp: &Value) {
    assert!(res.converged, "{label}: did not converge");

    // MLE parameters (beta + alpha) and standard errors: tight agreement.
    check_vec(label, &res.params, exp, "params", 1e-9);
    check_vec(label, &res.bse, exp, "bse", 1e-9);
    check_vec(label, &res.tvalues, exp, "tvalues", 1e-9);
    check_vec(label, &res.predicted, exp, "predicted", 1e-9);

    // p-values flow through a distribution tail.
    check_vec(label, &res.pvalues, exp, "pvalues", 1e-9);

    check_scalar(label, res.llf, exp, "llf", 1e-9);
    check_scalar(label, res.llnull, exp, "llnull", 1e-9);
    check_scalar(label, res.llr, exp, "llr", 1e-9);
    check_scalar(label, res.llr_pvalue, exp, "llr_pvalue", 1e-9);
    check_scalar(label, res.prsquared, exp, "prsquared", 1e-9);
    check_scalar(label, res.aic, exp, "aic", 1e-9);
    check_scalar(label, res.bic, exp, "bic", 1e-9);
    check_scalar(label, res.df_model, exp, "df_model", 1e-12);
    check_scalar(label, res.df_resid, exp, "df_resid", 1e-12);
    check_scalar(label, res.nobs, exp, "nobs", 1e-12);

    // Covariance (includes the alpha row/column).
    check_mat(label, &res.cov_params, exp, "cov_params", 1e-9);

    let ci = res.conf_int(0.05);
    check_mat(label, &ci, exp, "conf_int", 1e-9);
}

#[test]
fn discrete_ext_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let model = c["model"].as_str().unwrap();
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        match model {
            "mnlogit" => {
                let res = MNLogit::new(y, x).unwrap().fit().unwrap();
                verify_mnlogit(name, &res, &c["expected"]);
            }
            "negbin" => {
                let res = NegativeBinomial::new(y, x).unwrap().fit().unwrap();
                verify_negbin(name, &res, &c["expected"]);
            }
            other => panic!("unknown model {other}"),
        }
    }
}

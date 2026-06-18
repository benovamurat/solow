//! Cross-validation of the second batch of extended discrete estimators —
//! [`OrderedModel`] (ordinal logit / probit), [`ZeroInflatedPoisson`], and
//! [`GeneralizedPoisson`] (GP-1) — against golden reference values frozen in
//! `tests/fixtures/discrete_ext2.json`.
//!
//! All three estimators are maximum-likelihood and converge to the same optimum
//! as the reference, so the estimated parameters, standard errors, and
//! log-likelihoods agree very tightly. Achieved accuracy (probed against the
//! fixtures): every MLE-derived quantity matches to better than 1e-7 relative,
//! comfortably inside the spec's 1e-5/1e-6 bounds. Exact integer-valued counts
//! (`df_model`, `df_resid`, `nobs`) match to 1e-12.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_discrete::{Distr, GeneralizedPoisson, OrderedModel, ZeroInflatedPoisson};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/discrete_ext2.json"
    );
    let s =
        fs::read_to_string(p).expect("fixture present (run tools/reference/gen_discrete_ext2.py)");
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

// --------------------------------------------------------------------------- //
//  OrderedModel                                                               //
// --------------------------------------------------------------------------- //
fn verify_ordered(label: &str, res: &solow_discrete::OrderedResults, exp: &Value) {
    assert!(res.converged, "{label}: did not converge");

    // MLE parameters and standard errors. The optimizer reaches the same
    // interior optimum as the reference; the covariance is the inverse negative
    // numeric Hessian in the transformed-cutpoint space, exactly as the
    // reference forms it. Quantities that pass through that finite-difference
    // covariance (bse/tvalues/pvalues/conf_int) agree to ~3e-8; everything else
    // is at machine precision.
    check_vec(label, &res.params, exp, "params", 1e-8);
    check_vec(label, &res.bse, exp, "bse", 1e-6);
    check_vec(label, &res.tvalues, exp, "tvalues", 1e-6);
    check_vec(label, &res.thresholds, exp, "thresholds", 1e-8);
    check_vec(label, &res.pvalues, exp, "pvalues", 1e-6);

    // n x K predicted probabilities.
    check_mat(label, &res.predicted, exp, "predicted", 1e-8);

    check_scalar(label, res.llf, exp, "llf", 1e-10);
    check_scalar(label, res.llnull, exp, "llnull", 1e-10);
    check_scalar(label, res.llr, exp, "llr", 1e-8);
    check_scalar(label, res.llr_pvalue, exp, "llr_pvalue", 1e-8);
    check_scalar(label, res.prsquared, exp, "prsquared", 1e-8);
    check_scalar(label, res.aic, exp, "aic", 1e-10);
    check_scalar(label, res.bic, exp, "bic", 1e-10);
    check_scalar(label, res.df_model, exp, "df_model", 1e-12);
    check_scalar(label, res.df_resid, exp, "df_resid", 1e-12);
    check_scalar(label, res.nobs, exp, "nobs", 1e-12);

    let ci = res.conf_int(0.05);
    check_mat(label, &ci, exp, "conf_int", 1e-6);
}

// --------------------------------------------------------------------------- //
//  ZeroInflatedPoisson                                                        //
// --------------------------------------------------------------------------- //
fn verify_zip(label: &str, res: &solow_discrete::ZeroInflatedPoissonResults, exp: &Value) {
    assert!(res.converged, "{label}: did not converge");

    // Both blocks converge to the same MLE as the reference, so every quantity
    // agrees to ~1e-9 or tighter (probed). Tolerances are set well inside the
    // spec's 1e-5/1e-6 bounds.
    check_vec(label, &res.params, exp, "params", 1e-8);
    check_vec(label, &res.bse, exp, "bse", 1e-8);
    check_vec(label, &res.tvalues, exp, "tvalues", 1e-8);
    check_vec(label, &res.predicted, exp, "predicted", 1e-8);
    check_vec(label, &res.pvalues, exp, "pvalues", 1e-8);

    // Covariance (block-diagonal between inflation and main blocks, matching the
    // reference's assembled analytic Hessian).
    check_mat(label, &res.cov_params, exp, "cov_params", 1e-8);

    check_scalar(label, res.llf, exp, "llf", 1e-10);
    check_scalar(label, res.llnull, exp, "llnull", 1e-8);
    check_scalar(label, res.llr, exp, "llr", 1e-8);
    check_scalar(label, res.llr_pvalue, exp, "llr_pvalue", 1e-8);
    check_scalar(label, res.prsquared, exp, "prsquared", 1e-8);
    check_scalar(label, res.aic, exp, "aic", 1e-10);
    check_scalar(label, res.bic, exp, "bic", 1e-10);
    check_scalar(label, res.df_model, exp, "df_model", 1e-12);
    check_scalar(label, res.df_resid, exp, "df_resid", 1e-12);
    check_scalar(label, res.nobs, exp, "nobs", 1e-12);

    let ci = res.conf_int(0.05);
    check_mat(label, &ci, exp, "conf_int", 1e-8);
}

// --------------------------------------------------------------------------- //
//  GeneralizedPoisson                                                         //
// --------------------------------------------------------------------------- //
fn verify_gp(label: &str, res: &solow_discrete::GeneralizedPoissonResults, exp: &Value) {
    assert!(res.converged, "{label}: did not converge");

    check_vec(label, &res.params, exp, "params", 1e-8);
    check_vec(label, &res.bse, exp, "bse", 1e-8);
    check_vec(label, &res.tvalues, exp, "tvalues", 1e-8);
    check_vec(label, &res.predicted, exp, "predicted", 1e-8);
    check_vec(label, &res.pvalues, exp, "pvalues", 1e-8);

    check_mat(label, &res.cov_params, exp, "cov_params", 1e-8);

    check_scalar(label, res.llf, exp, "llf", 1e-10);
    check_scalar(label, res.llnull, exp, "llnull", 1e-8);
    check_scalar(label, res.llr, exp, "llr", 1e-8);
    check_scalar(label, res.llr_pvalue, exp, "llr_pvalue", 1e-8);
    check_scalar(label, res.prsquared, exp, "prsquared", 1e-8);
    check_scalar(label, res.aic, exp, "aic", 1e-10);
    check_scalar(label, res.bic, exp, "bic", 1e-10);
    check_scalar(label, res.df_model, exp, "df_model", 1e-12);
    check_scalar(label, res.df_resid, exp, "df_resid", 1e-12);
    check_scalar(label, res.nobs, exp, "nobs", 1e-12);

    let ci = res.conf_int(0.05);
    check_mat(label, &ci, exp, "conf_int", 1e-8);
}

#[test]
fn discrete_ext2_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let model = c["model"].as_str().unwrap();
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        match model {
            "ordered" => {
                let distr = match c["distr"].as_str().unwrap() {
                    "logit" => Distr::Logit,
                    "probit" => Distr::Probit,
                    other => panic!("unknown distr {other}"),
                };
                let res = OrderedModel::with_distr(y, x, distr)
                    .unwrap()
                    .fit()
                    .unwrap();
                verify_ordered(name, &res, &c["expected"]);
            }
            "zip" => {
                let res = ZeroInflatedPoisson::new(y, x).unwrap().fit().unwrap();
                verify_zip(name, &res, &c["expected"]);
            }
            "genpoisson" => {
                let res = GeneralizedPoisson::new(y, x).unwrap().fit().unwrap();
                verify_gp(name, &res, &c["expected"]);
            }
            other => panic!("unknown model {other}"),
        }
    }
}

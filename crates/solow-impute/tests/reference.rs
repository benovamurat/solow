//! Cross-validation of the deterministic multiple-imputation core against
//! golden reference values frozen in `tests/fixtures/impute.json`.
//!
//! Two deterministic pieces are checked:
//!   * Rubin's combining rules (pooled estimate, within/between/total
//!     covariance, FMI, Barnard-Rubin df, and the derived inference) fed with
//!     fixed per-imputation inputs, and
//!   * conditional-mean regression imputation of one variable given the others.
//!
//! The RNG-driven posterior draws of a full MICE run cannot be bit-matched and
//! are out of scope.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_impute::{combine, conditional_mean_impute};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/impute.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_impute.py)");
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

#[test]
fn combine_matches_reference() {
    let fx = load();
    for c in fx["combine"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let params_list: Vec<Array1<f64>> = c["params_list"]
            .as_array()
            .unwrap()
            .iter()
            .map(vec1)
            .collect();
        let cov_list: Vec<Array2<f64>> =
            c["cov_list"].as_array().unwrap().iter().map(mat).collect();
        let dfcom = c["dfcom"].as_f64().unwrap_or(f64::INFINITY);

        let res = combine(&params_list, &cov_list, dfcom).unwrap();
        let exp = &c["expected"];

        // Closed-form linear-algebra quantities: tight (1e-8).
        check_vec(name, &res.params, exp, "params", 1e-8);
        check_mat(name, &res.cov_within, exp, "cov_within", 1e-8);
        check_mat(name, &res.cov_between, exp, "cov_between", 1e-8);
        check_mat(name, &res.cov_total, exp, "cov_total", 1e-8);
        check_vec(name, &res.bse, exp, "bse", 1e-8);
        check_vec(name, &res.relative_increase, exp, "relative_increase", 1e-8);
        check_vec(name, &res.fmi, exp, "fmi", 1e-8);
        check_vec(name, &res.df, exp, "df", 1e-8);
        check_vec(name, &res.tvalues(), exp, "tvalues", 1e-8);

        // Quantities through the Student-t inverse: 1e-6.
        check_vec(name, &res.pvalues(), exp, "pvalues", 1e-6);
        let ci = mat(&exp["conf_int"]);
        let got = res.conf_int(0.05);
        for i in 0..res.params.len() {
            for j in 0..2 {
                assert!(
                    rel(got[[i, j]], ci[[i, j]]) <= 1e-6,
                    "{name}.conf_int[{i}][{j}]: got {}, want {}",
                    got[[i, j]],
                    ci[[i, j]]
                );
            }
        }
    }
}

#[test]
fn regression_imputation_matches_reference() {
    let fx = load();
    for c in fx["regression"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let endog_obs = vec1(&c["endog_obs"]);
        let exog_obs = mat(&c["exog_obs"]);
        let exog_miss = mat(&c["exog_miss"]);

        let res = conditional_mean_impute(endog_obs, exog_obs, &exog_miss).unwrap();
        let exp = &c["expected"];

        check_vec(name, &res.params, exp, "params", 1e-8);
        check_vec(name, &res.fitted_observed, exp, "fitted_observed", 1e-8);
        check_vec(name, &res.imputed_missing, exp, "imputed_missing", 1e-8);
        let want_scale = exp["scale"].as_f64().unwrap();
        assert!(
            rel(res.scale, want_scale) <= 1e-8,
            "{name}.scale: got {}, want {}",
            res.scale,
            want_scale
        );
    }
}

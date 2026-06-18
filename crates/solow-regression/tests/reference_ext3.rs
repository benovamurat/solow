//! Cross-validation of the robust (sandwich) covariance estimators layered on
//! top of OLS, against golden reference values frozen in
//! `tests/fixtures/regression_ext3.json`.
//!
//! Coverage: HC0, HC1, HC2, HC3, HAC (Newey-West, Bartlett kernel, with and
//! without the `n/(n-k)` correction), and one-way cluster (with and without the
//! `G/(G-1)*(n-1)/(n-k)` correction).
//!
//! Tolerances follow the crate discipline. Every quantity here is a closed-form
//! function of the OLS fit (no iteration), so once `params` and
//! `normalized_cov_params` match, the sandwich entries reproduce the reference
//! to machine precision. Robust `cov_params` and `bse` are asserted to <=1e-8;
//! the derived `tvalues` and `pvalues` are asserted to <=1e-8 as well.
//!
//! Small-sample-correction nuance mirrored from the reference:
//! * HC1 scales HC0 by `n/(n-k)`.
//! * HAC `use_correction=True` multiplies the sandwich by `n/(n-k)`.
//! * cluster `use_correction=True` multiplies by `G/(G-1)*(n-1)/(n-k)`, where
//!   `G` is the number of distinct groups.
//! * The p-values for the cluster estimator use `G-1` inference degrees of
//!   freedom (the reference's `df_resid_inference` under its default
//!   `df_correction`), while HC*/HAC keep `n-k`. The test reads the fixture's
//!   `df_inference` and confirms our `pvalues_robust` matches.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_regression::{robust_cov, CovType, LinearModel};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/regression_ext3.json"
    );
    let s = fs::read_to_string(p)
        .expect("fixture present (run tools/reference/gen_regression_ext3.py)");
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
    assert_eq!(got.len(), want.len(), "{label}.{key}: length mismatch");
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
    assert_eq!(got.dim(), want.dim(), "{label}.{key}: shape mismatch");
    let (m, n) = got.dim();
    for i in 0..m {
        for j in 0..n {
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

/// Build the `CovType` for one robust block from the fixture.
fn cov_type_of(block: &Value) -> CovType {
    match block["cov_type"].as_str().unwrap() {
        "HC0" => CovType::Hc0,
        "HC1" => CovType::Hc1,
        "HC2" => CovType::Hc2,
        "HC3" => CovType::Hc3,
        "HAC" => CovType::Hac {
            maxlags: block["maxlags"].as_u64().unwrap() as usize,
            use_correction: block["use_correction"].as_bool().unwrap(),
        },
        "cluster" => CovType::Cluster {
            groups: block["groups"]
                .as_array()
                .unwrap()
                .iter()
                .map(|g| g.as_i64().unwrap())
                .collect(),
            use_correction: block["use_correction"].as_bool().unwrap(),
        },
        other => panic!("unknown cov_type {other}"),
    }
}

#[test]
fn robust_cov_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        let res = LinearModel::ols(y, x.clone()).unwrap().fit().unwrap();

        // Sanity: the underlying OLS fit reproduces the reference coefficients
        // and df_resid; otherwise every sandwich entry would be off too.
        check_vec(name, &res.params, c, "params", 1e-10);
        let df_resid = c["df_resid"].as_f64().unwrap();
        assert!(
            rel(res.df_resid, df_resid) <= 1e-12,
            "{name}.df_resid mismatch"
        );

        for block in c["robust"].as_array().unwrap() {
            let ct = cov_type_of(block);
            let tag = format!("{name}/{}", block["cov_type"].as_str().unwrap());

            // Robust covariance and standard errors via the convenience methods.
            let cov = res.cov_params_robust(&x, &ct).unwrap();
            check_mat(&tag, &cov, block, "cov_params", 1e-8);

            let bse = res.bse_robust(&x, &ct).unwrap();
            check_vec(&tag, &bse, block, "bse", 1e-8);

            // Inference: t-statistics and two-sided p-values.
            let tvalues = res.tvalues_robust(&x, &ct).unwrap();
            check_vec(&tag, &tvalues, block, "tvalues", 1e-8);

            let pvalues = res.pvalues_robust(&x, &ct).unwrap();
            check_vec(&tag, &pvalues, block, "pvalues", 1e-8);

            // The free function returns the same covariance as the method.
            let cov_free = robust_cov(
                &x,
                &res.resid,
                &res.normalized_cov_params,
                res.df_resid,
                &ct,
            )
            .unwrap();
            check_mat(&tag, &cov_free, block, "cov_params", 1e-8);
        }
    }
}

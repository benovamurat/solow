//! Cross-validation of the GEE estimator against golden reference values
//! frozen in `tests/fixtures/gee.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_gee::{CovStruct, Gee};
use solow_glm::Family;
use std::fs;

fn load() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/gee.json");
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_gee.py)");
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

fn groups_of(v: &Value) -> Vec<i64> {
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

fn family_for(name: &str) -> Family {
    match name {
        "Gaussian" => Family::Gaussian,
        "Poisson" => Family::Poisson,
        "Binomial" => Family::Binomial,
        other => panic!("unknown family {other}"),
    }
}

fn cov_for(name: &str) -> CovStruct {
    match name {
        "independence" => CovStruct::Independence,
        "exchangeable" => CovStruct::Exchangeable,
        other => panic!("unknown cov_struct {other}"),
    }
}

#[test]
fn gee_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let family = family_for(c["family"].as_str().unwrap());
        let link = family.default_link();
        let cov = cov_for(c["cov_struct"].as_str().unwrap());
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        let groups = groups_of(&c["groups"]);

        let res = Gee::with_link(y, x, &groups, family, link, cov)
            .unwrap()
            .fit()
            .unwrap();
        assert!(res.converged, "{name}: did not converge");

        let exp = &c["expected"];
        // Score equations solved essentially exactly => params closed-form tight.
        check_vec(name, &res.params, exp, "params", 1e-8);
        // Robust sandwich SEs and the full covariance.
        check_vec(name, &res.bse, exp, "bse", 1e-8);
        check_vec(name, &res.bse_naive, exp, "bse_naive", 1e-8);
        check_mat(name, &res.cov_robust, exp, "cov_robust", 1e-8);
        check_mat(name, &res.cov_naive, exp, "cov_naive", 1e-8);
        // z-statistics; p-values pass through the normal survival function.
        check_vec(name, &res.tvalues, exp, "tvalues", 1e-8);
        check_vec(name, &res.pvalues, exp, "pvalues", 1e-7);
        // Working-correlation parameter and dispersion.
        check_scalar(name, res.dep_params, exp, "dep_params", 1e-8);
        check_scalar(name, res.scale, exp, "scale", 1e-8);
        check_vec(name, &res.fittedvalues, exp, "fittedvalues", 1e-8);
    }
}

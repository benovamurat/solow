//! Cross-validation of the GLM estimator against golden reference values
//! frozen in `tests/fixtures/glm.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_glm::{Family, Glm, GlmResults, Link};
use std::fs;

fn load() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/glm.json");
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_glm.py)");
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

fn family_for(name: &str) -> Family {
    match name {
        "Gaussian" => Family::Gaussian,
        "Poisson" => Family::Poisson,
        "Binomial" => Family::Binomial,
        "Gamma" => Family::Gamma,
        "InverseGaussian" => Family::InverseGaussian,
        other => panic!("unknown family {other}"),
    }
}

fn link_for(name: &str) -> Link {
    match name {
        "identity" => Link::Identity,
        "log" => Link::Log,
        "logit" => Link::Logit,
        "probit" => Link::Probit,
        "cloglog" => Link::CLogLog,
        "inverse_power" => Link::InversePower,
        other => panic!("unknown link {other}"),
    }
}

fn verify(label: &str, res: &GlmResults, exp: &Value) {
    check_vec(label, &res.params, exp, "params", 1e-7);
    check_vec(label, &res.bse, exp, "bse", 1e-6);
    check_vec(label, &res.tvalues, exp, "tvalues", 1e-6);
    check_vec(label, &res.pvalues, exp, "pvalues", 1e-6);
    check_vec(label, &res.fittedvalues, exp, "fittedvalues", 1e-7);
    check_vec(label, &res.resid_response, exp, "resid_response", 1e-7);
    check_vec(label, &res.resid_pearson, exp, "resid_pearson", 1e-7);
    check_vec(label, &res.resid_deviance, exp, "resid_deviance", 1e-6);

    check_scalar(label, res.deviance, exp, "deviance", 1e-8);
    check_scalar(label, res.pearson_chi2, exp, "pearson_chi2", 1e-8);
    check_scalar(label, res.null_deviance, exp, "null_deviance", 1e-8);
    check_scalar(label, res.llf, exp, "llf", 1e-7);
    check_scalar(label, res.aic, exp, "aic", 1e-7);
    check_scalar(label, res.bic, exp, "bic", 1e-7);
    check_scalar(label, res.scale, exp, "scale", 1e-8);
    check_scalar(label, res.df_model, exp, "df_model", 1e-12);
    check_scalar(label, res.df_resid, exp, "df_resid", 1e-12);

    let ci = mat(&exp["conf_int"]);
    let got = res.conf_int(0.05);
    for i in 0..res.params.len() {
        for j in 0..2 {
            assert!(
                rel(got[[i, j]], ci[[i, j]]) <= 1e-6,
                "{label}.conf_int[{i}][{j}]"
            );
        }
    }
}

#[test]
fn glm_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let family = family_for(c["family"].as_str().unwrap());
        let link = link_for(c["link"].as_str().unwrap());
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        let res = Glm::with_link(y, x, family, link).unwrap().fit().unwrap();
        assert!(res.converged, "{name}: did not converge");
        verify(name, &res, &c["expected"]);
    }
}

//! Cross-validation of the extended GAM features against golden reference
//! values frozen in `tests/fixtures/gam_ext.json`:
//!
//! * non-canonical-link penalized B-spline GAMs (Gaussian/log, Binomial/probit)
//!   where the effective degrees of freedom come from the observed information;
//! * penalized cyclic cubic regression spline GAMs (canonical link).
//!
//! Regenerate the fixture with
//! `SOLOW_REFERENCE=<pkg> python3 tools/reference/gen_gam_ext.py`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_gam::{BSplines, CyclicCubicSplines, GamExtResults, GlmGamExt};
use solow_glm::{Family, Link};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/gam_ext.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_gam_ext.py)");
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

fn family_for(name: &str) -> Family {
    match name {
        "Gaussian" => Family::Gaussian,
        "Poisson" => Family::Poisson,
        "Binomial" => Family::Binomial,
        other => panic!("unknown family {other}"),
    }
}

fn link_for(name: &str) -> Link {
    match name {
        "identity" => Link::Identity,
        "log" => Link::Log,
        "logit" => Link::Logit,
        "probit" => Link::Probit,
        other => panic!("unknown link {other}"),
    }
}

fn verify_bspline(c: &Value) {
    let name = c["name"].as_str().unwrap();
    let family = family_for(c["family"].as_str().unwrap());
    let link = link_for(c["link"].as_str().unwrap());
    let df = c["df"].as_u64().unwrap() as usize;
    let degree = c["degree"].as_u64().unwrap() as usize;
    let alpha = c["alpha"].as_f64().unwrap();
    let x = vec1(&c["x"]);
    let y = vec1(&c["endog"]);
    let exp = &c["expected"];

    // Basis (closed-form): tight tolerances.
    let bs = BSplines::new(&x, df, degree).unwrap();
    check_vec(name, bs.knots(), exp, "knots", 1e-12);
    assert_eq!(bs.dim_basis(), exp["dim_basis"].as_u64().unwrap() as usize);
    check_mat(name, bs.basis(), exp, "basis", 1e-11);
    check_mat(name, bs.cov_der2(), exp, "cov_der2", 1e-9);

    // Fit with the non-canonical link via the observed-information edf path.
    let res: GamExtResults = GlmGamExt::new(
        y,
        bs.basis().clone(),
        bs.cov_der2().clone(),
        alpha,
        family,
        link,
    )
    .unwrap()
    .fit()
    .unwrap();
    assert!(res.converged, "{name}: P-IRLS did not converge");

    // MLE params/fitted from a closed-form penalized solve: very tight.
    check_vec(name, &res.params, exp, "params", 1e-8);
    check_vec(name, &res.fittedvalues, exp, "fittedvalues", 1e-8);
    // Observed-information edf: matches the reference's hat-matrix diagonal.
    check_vec(name, &res.edf, exp, "edf", 1e-8);
    check_scalar(name, res.edf_total, exp, "edf_total", 1e-8);
    check_scalar(name, res.edf_total, exp, "hat_matrix_trace", 1e-8);
    check_scalar(name, res.scale, exp, "scale", 1e-8);
    check_scalar(name, res.deviance, exp, "deviance", 1e-8);
    check_scalar(
        name,
        res.penalized_deviance,
        exp,
        "penalized_deviance",
        1e-8,
    );
}

fn verify_cyclic(c: &Value) {
    let name = c["name"].as_str().unwrap();
    let family = family_for(c["family"].as_str().unwrap());
    let link = link_for(c["link"].as_str().unwrap());
    let df = c["df"].as_u64().unwrap() as usize;
    let alpha = c["alpha"].as_f64().unwrap();
    let x = vec1(&c["x"]);
    let y = vec1(&c["endog"]);
    let exp = &c["expected"];

    // Cyclic basis with the centering constraint (closed-form): tight
    // tolerances.
    let cc = CyclicCubicSplines::with_centering(&x, df).unwrap();
    check_vec(name, cc.knots(), exp, "all_knots", 1e-12);
    assert_eq!(cc.dim_basis(), exp["dim_basis"].as_u64().unwrap() as usize);
    check_mat(name, cc.basis(), exp, "basis", 1e-10);
    check_mat(name, cc.cov_der2(), exp, "cov_der2", 1e-8);

    let res: GamExtResults = GlmGamExt::new(
        y,
        cc.basis().clone(),
        cc.cov_der2().clone(),
        alpha,
        family,
        link,
    )
    .unwrap()
    .fit()
    .unwrap();
    assert!(res.converged, "{name}: P-IRLS did not converge");

    check_vec(name, &res.params, exp, "params", 1e-8);
    check_vec(name, &res.fittedvalues, exp, "fittedvalues", 1e-8);
    check_vec(name, &res.edf, exp, "edf", 1e-8);
    check_scalar(name, res.edf_total, exp, "edf_total", 1e-8);
    check_scalar(name, res.edf_total, exp, "hat_matrix_trace", 1e-8);
    check_scalar(name, res.scale, exp, "scale", 1e-8);
    check_scalar(name, res.deviance, exp, "deviance", 1e-8);
    check_scalar(
        name,
        res.penalized_deviance,
        exp,
        "penalized_deviance",
        1e-8,
    );
}

#[test]
fn gam_ext_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        match c["kind"].as_str().unwrap() {
            "bspline" => verify_bspline(c),
            "cyclic" => verify_cyclic(c),
            other => panic!("unknown case kind {other}"),
        }
    }
}

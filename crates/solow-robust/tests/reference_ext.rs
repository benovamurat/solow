//! Cross-validation of the M-estimator norm extensions (Hampel, RamsayE,
//! TrimmedMean) and a Hampel-norm RLM against golden reference values frozen in
//! `tests/fixtures/robust_ext.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_robust::norms::RobustNorm;
use solow_robust::{Hampel, RamsayE, Rlm, RlmResults, TrimmedMean};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/robust_ext.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_robust_ext.py)");
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

fn norm_by_name(name: &str) -> Box<dyn RobustNorm> {
    match name {
        "Hampel" => Box::new(Hampel::default()),
        "RamsayE" => Box::new(RamsayE::default()),
        "TrimmedMean" => Box::new(TrimmedMean::default()),
        other => panic!("unknown norm {other}"),
    }
}

/// rho / psi / weights / psi_deriv on the dense [-8, 8] grid. All four are
/// closed-form, so we hold them to 1e-10.
#[test]
fn norm_ext_units_match_reference() {
    let fx = load();
    for u in fx["norm_units"].as_array().unwrap() {
        let name = u["name"].as_str().unwrap();
        let norm = norm_by_name(name);
        let z = vec1(&u["z"]);
        let zs = z.as_slice().unwrap();

        let rho = Array1::from_vec(norm.rho_arr(zs));
        let psi = Array1::from_vec(norm.psi_arr(zs));
        let w = Array1::from_vec(norm.weights_arr(zs));
        let pd = Array1::from_vec(norm.psi_deriv_arr(zs));

        check_vec(name, &rho, u, "rho", 1e-10);
        check_vec(name, &psi, u, "psi", 1e-10);
        check_vec(name, &w, u, "weights", 1e-10);
        check_vec(name, &pd, u, "psi_deriv", 1e-10);
    }
}

fn fit_case(c: &Value) -> RlmResults {
    let y = vec1(&c["endog"]);
    let x = mat(&c["exog"]);
    let norm = c["norm"].as_str().unwrap();
    assert_eq!(c["scale_est"].as_str().unwrap(), "mad");

    // Converge tightly so we land on the same optimum as the reference fixture.
    macro_rules! run {
        ($n:expr) => {{
            Rlm::new(y.clone(), x.clone(), $n)
                .unwrap()
                .tol(1e-12)
                .maxiter(300)
                .fit()
                .unwrap()
        }};
    }
    match norm {
        "Hampel" => run!(Hampel::default()),
        "RamsayE" => run!(RamsayE::default()),
        "TrimmedMean" => run!(TrimmedMean::default()),
        other => panic!("unknown norm {other}"),
    }
}

fn verify_case(label: &str, res: &RlmResults, exp: &Value) {
    // params/bse/scale and everything derived from the MLE converge to ~1e-6.
    check_vec(label, &res.params, exp, "params", 1e-6);
    check_vec(label, &res.bse, exp, "bse", 1e-6);
    check_vec(label, &res.tvalues, exp, "tvalues", 1e-6);
    check_vec(label, &res.pvalues, exp, "pvalues", 1e-6);
    check_vec(label, &res.fittedvalues, exp, "fittedvalues", 1e-6);
    check_vec(label, &res.resid, exp, "resid", 1e-6);
    check_vec(label, &res.sresid, exp, "sresid", 1e-6);
    check_vec(label, &res.weights, exp, "weights", 1e-6);

    check_scalar(label, res.scale, exp, "scale", 1e-6);
    check_scalar(label, res.df_model, exp, "df_model", 1e-12);
    check_scalar(label, res.df_resid, exp, "df_resid", 1e-12);
    check_scalar(label, res.nobs, exp, "nobs", 1e-12);

    let ci = mat(&exp["conf_int"]);
    let got = res.conf_int(0.05);
    for i in 0..res.params.len() {
        for j in 0..2 {
            let e = rel(got[[i, j]], ci[[i, j]]);
            assert!(e <= 1e-6, "{label}.conf_int[{i}][{j}]: rel-err {e:.3e}");
        }
    }
}

#[test]
fn rlm_hampel_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let res = fit_case(c);
        assert!(res.converged, "{name}: did not converge");
        verify_case(name, &res, &c["expected"]);
    }
}

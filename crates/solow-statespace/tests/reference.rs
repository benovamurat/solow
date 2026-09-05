//! Cross-validation of the SARIMAX estimator and the Kalman-filter
//! log-likelihood against golden reference values frozen in
//! `tests/fixtures/statespace.json` (run `tools/reference/gen_statespace.py`).

use ndarray::Array1;
use serde_json::Value;
use solow_statespace::{Sarimax, SarimaxOrder};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/statespace.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_statespace.py)");
    serde_json::from_str(&s).unwrap()
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
    assert_eq!(
        got.len(),
        want.len(),
        "{label}.{key}: length {} vs {}",
        got.len(),
        want.len()
    );
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

fn order_for(order: &[i64], sorder: &[i64]) -> SarimaxOrder {
    let (p, d, q) = (order[0] as usize, order[1] as usize, order[2] as usize);
    let (sp, sd, sq, s) = (
        sorder[0] as usize,
        sorder[1] as usize,
        sorder[2] as usize,
        sorder[3] as usize,
    );
    if s == 0 {
        SarimaxOrder::new(p, d, q)
    } else {
        SarimaxOrder::seasonal(p, d, q, sp, sd, sq, s)
    }
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

#[test]
fn sarimax_matches_reference() {
    let fx = load();
    let series = vec1(&fx["series"]);

    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let order = order_for(&ints(&c["order"]), &ints(&c["seasonal_order"]));
        let model = Sarimax::new(series.clone(), order).unwrap();

        // 1. Fixed-parameter log-likelihood: validates the state-space matrices
        //    and the Kalman recursion independent of the optimizer. Closed-form,
        //    so it must match to ~1e-9.
        let check_params = vec1(&c["check_params"]);
        let y = model.differenced();
        let ss = solow_statespace::sarimax::build_state_space_for_test(&check_params, &order)
            .expect("state space");
        let out = ss.filter(&y, 0);
        check_scalar(name, out.loglike, c, "check_loglike", 1e-9);

        // 2. Full ML fit.
        let res = model.fit().unwrap();
        assert!(res.converged, "{name}: optimizer did not converge");
        assert_eq!(
            res.nobs,
            c["nobs"].as_u64().unwrap() as usize,
            "{name}: nobs"
        );

        // MLE parameters: both implementations are polished to the true optimum
        // in the same unconstrained space, agreeing to <=4e-8 here.
        check_vec(name, &res.params, c, "params", 1e-7);
        // Log-likelihood and information criteria are essentially exact (<=1e-12).
        check_scalar(name, res.llf, c, "llf", 1e-9);
        check_scalar(name, res.aic, c, "aic", 1e-9);
        check_scalar(name, res.bic, c, "bic", 1e-9);
        check_scalar(name, res.hqic, c, "hqic", 1e-9);

        // Standard errors come from a finite-difference Hessian of the loglike,
        // computed with the same central-difference scheme on both sides; they
        // agree to <=4e-7.
        check_vec(name, &res.bse, c, "bse", 1e-6);
        check_vec(name, &res.zvalues, c, "zvalues", 1e-6);
        check_vec(name, &res.pvalues, c, "pvalues", 1e-6);

        // In-sample one-step-ahead fitted values and residuals (<=3e-7).
        check_vec(name, &res.fittedvalues, c, "fittedvalues", 1e-6);
        check_vec(name, &res.resid, c, "resid", 1e-6);
    }
}

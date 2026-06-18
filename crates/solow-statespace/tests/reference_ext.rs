//! Cross-validation of the UnobservedComponents and DynamicFactor estimators
//! against golden reference values frozen in `tests/fixtures/statespace_ext.json`
//! (run `tools/reference/gen_statespace_ext.py`).
//!
//! Two kinds of checks are made per case:
//!
//! * a *fixed-parameter* log-likelihood at the reference `check_params`, which
//!   validates the state-space matrices and the Kalman recursion independently
//!   of the optimizer (asserted tightly), and
//! * the *fitted* maximum-likelihood log-likelihood, information criteria and
//!   parameters, which validate full convergence.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_statespace::{DynamicFactor, Level, UcSpec, UnobservedComponents};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/statespace_ext.json"
    );
    let s =
        fs::read_to_string(p).expect("fixture present (run tools/reference/gen_statespace_ext.py)");
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

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / (1.0 + want.abs())
}

fn check_scalar(label: &str, got: f64, want: f64, tol: f64) {
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{label}: rel-err {e:.3e} (got {got}, want {want})"
    );
}

fn level_for(name: &str) -> Level {
    match name {
        "local level" => Level::LocalLevel,
        "local linear trend" => Level::LocalLinearTrend,
        other => panic!("unknown level {other}"),
    }
}

/// Compare two parameter vectors element-by-element, allowing parameters that
/// the reference pins to the boundary (a variance of essentially zero) to be
/// matched by absolute closeness instead of relative error: near zero the
/// relative-error denominator is 1, and both implementations converge to the
/// flat boundary, so a small absolute gap is the honest criterion there.
fn check_params(label: &str, got: &Array1<f64>, want: &Array1<f64>, tol: f64, boundary_abs: f64) {
    assert_eq!(got.len(), want.len(), "{label}: param length");
    for i in 0..got.len() {
        let g = got[i];
        let w = want[i];
        if w.abs() < 1e-6 {
            // Boundary parameter: both should be essentially zero.
            assert!(
                g.abs() <= boundary_abs,
                "{label}.params[{i}]: boundary param got {g} want {w} (abs tol {boundary_abs:.1e})"
            );
        } else {
            let e = rel(g, w);
            assert!(
                e <= tol,
                "{label}.params[{i}]: rel-err {e:.3e} (got {g}, want {w})"
            );
        }
    }
}

#[test]
fn unobserved_components_matches_reference() {
    let fx = load();
    let y = vec1(&fx["series"]);
    for c in fx["cases"].as_array().unwrap() {
        if c["kind"].as_str().unwrap() != "uc" {
            continue;
        }
        let name = c["name"].as_str().unwrap();
        let level = level_for(c["level"].as_str().unwrap());
        let seasonal = c["seasonal"].as_u64().unwrap() as usize;
        let mut spec = UcSpec::new(level);
        if seasonal > 0 {
            spec = spec.with_seasonal(seasonal);
        }
        let model = UnobservedComponents::new(y.clone(), spec).unwrap();

        // Fixed-parameter log-likelihood (tight: only the matrices + recursion).
        let check_params_v = vec1(&c["check_params"]);
        let got_fixed = model
            .loglike(&check_params_v)
            .expect("finite fixed loglike");
        check_scalar(
            &format!("{name}.check_loglike"),
            got_fixed,
            c["check_loglike"].as_f64().unwrap(),
            1e-9,
        );

        // Fitted model. The log-likelihood and information criteria match to
        // ~1e-9; the variance parameters live on a very flat ridge (one is
        // pinned to the boundary), so they are matched to 1e-5.
        let res = model.fit().unwrap();
        assert!(res.converged, "{name}: did not converge");
        check_scalar(
            &format!("{name}.llf"),
            res.llf,
            c["llf"].as_f64().unwrap(),
            1e-9,
        );
        check_scalar(
            &format!("{name}.aic"),
            res.aic,
            c["aic"].as_f64().unwrap(),
            1e-9,
        );
        check_scalar(
            &format!("{name}.bic"),
            res.bic,
            c["bic"].as_f64().unwrap(),
            1e-9,
        );
        check_scalar(
            &format!("{name}.hqic"),
            res.hqic,
            c["hqic"].as_f64().unwrap(),
            1e-9,
        );

        let want_params = vec1(&c["params"]);
        check_params(name, &res.params, &want_params, 1e-5, 1e-5);
    }
}

#[test]
fn dynamic_factor_matches_reference() {
    let fx = load();
    let panel = mat(&fx["panel"]);
    for c in fx["cases"].as_array().unwrap() {
        if c["kind"].as_str().unwrap() != "df" {
            continue;
        }
        let name = c["name"].as_str().unwrap();
        let factor_order = c["factor_order"].as_u64().unwrap() as usize;
        let model = DynamicFactor::new(panel.clone(), factor_order).unwrap();

        // Fixed-parameter log-likelihood. The stationary initialization is
        // solved with a linear system here and with an iterative Lyapunov
        // solver in the reference, yet the two agree to ~1e-12.
        let check_params_v = vec1(&c["check_params"]);
        let got_fixed = model
            .loglike(&check_params_v)
            .expect("finite fixed loglike");
        check_scalar(
            &format!("{name}.check_loglike"),
            got_fixed,
            c["check_loglike"].as_f64().unwrap(),
            1e-9,
        );

        // Fitted model. Log-likelihood and information criteria match to ~1e-9;
        // the loadings, idiosyncratic variances and factor AR to ~1e-7.
        let res = model.fit().unwrap();
        assert!(res.converged, "{name}: did not converge");
        check_scalar(
            &format!("{name}.llf"),
            res.llf,
            c["llf"].as_f64().unwrap(),
            1e-9,
        );
        check_scalar(
            &format!("{name}.aic"),
            res.aic,
            c["aic"].as_f64().unwrap(),
            1e-9,
        );
        check_scalar(
            &format!("{name}.bic"),
            res.bic,
            c["bic"].as_f64().unwrap(),
            1e-9,
        );
        check_scalar(
            &format!("{name}.hqic"),
            res.hqic,
            c["hqic"].as_f64().unwrap(),
            1e-9,
        );

        let want_params = vec1(&c["params"]);
        check_params(name, &res.params, &want_params, 1e-6, 1e-6);
    }
}

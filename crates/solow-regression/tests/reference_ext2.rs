//! Cross-validation of the further extended regression estimators (GLSAR,
//! SlicedInverseReg) against golden reference values frozen in
//! `tests/fixtures/regression_ext2.json`.
//!
//! Tolerances follow the crate discipline: GLSAR coefficients, standard errors,
//! and the AR coefficients `rho` are matched to <=1e-6 (the iterative two-step
//! fixed point is reproduced on both sides to convergence; the observed errors
//! are far smaller, around 1e-12). SIR eigenvalues are matched to <=1e-7 and the
//! EDR directions to <=1e-6 up to a per-column sign flip (eigenvectors are only
//! determined up to sign).

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_regression::{Glsar, SlicedInverseReg};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/regression_ext2.json"
    );
    let s = fs::read_to_string(p)
        .expect("fixture present (run tools/reference/gen_regression_ext2.py)");
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

fn check_scalar(label: &str, got: f64, want: f64, key: &str, tol: f64) {
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{label}.{key}: rel-err {e:.3e} (got {got}, want {want})"
    );
}

#[test]
fn glsar_matches_reference() {
    let fx = load();
    for c in fx["glsar"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        let order = c["order"].as_u64().unwrap() as usize;
        let maxiter = c["maxiter"].as_u64().unwrap() as usize;
        let rtol = c["rtol"].as_f64().unwrap();
        let res = Glsar::new(y, x, order)
            .unwrap()
            .iterative_fit(maxiter, rtol)
            .unwrap();

        // Regression coefficients and their standard errors: the inner fit is an
        // exact OLS on the whitened (reduced) data, so once `rho` matches the
        // reference fixed point these match to ~1e-15 (observed). Assertions keep
        // a large safety margin; both far exceed the spec floors (params 1e-6,
        // bse 1e-6).
        check_vec(name, &res.params, c, "params", 1e-9);
        check_vec(name, &res.bse, c, "bse", 1e-9);
        // Estimated AR coefficients from the iterative Yule-Walker updates
        // (observed ~1e-16; spec floor 1e-6).
        check_vec(name, &res.rho, c, "rho", 1e-9);

        // Auxiliary fit quantities (reduced sample size, scale, iteration count).
        check_scalar(
            name,
            res.ols.nobs,
            c["nobs"].as_f64().unwrap(),
            "nobs",
            1e-12,
        );
        check_scalar(
            name,
            res.ols.df_resid,
            c["df_resid"].as_f64().unwrap(),
            "df_resid",
            1e-12,
        );
        check_scalar(
            name,
            res.ols.scale,
            c["scale"].as_f64().unwrap(),
            "scale",
            1e-8,
        );
        assert_eq!(
            res.converged,
            c["converged"].as_bool().unwrap(),
            "{name}: converged flag"
        );
        assert_eq!(
            res.iterations as u64,
            c["iter"].as_u64().unwrap(),
            "{name}: iteration count"
        );
    }
}

#[test]
fn sir_matches_reference() {
    let fx = load();
    for c in fx["sir"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        let slice_n = c["slice_n"].as_u64().unwrap() as usize;
        let res = SlicedInverseReg::new(y, x).unwrap().fit(slice_n).unwrap();

        // Eigenvalues of the between-slice covariance: a deterministic
        // eigendecomposition, observed ~1e-16, matched tightly (spec floor 1e-7).
        check_vec(name, &res.eigenvalues, c, "eigenvalues", 1e-8);

        // EDR directions (params columns), each determined only up to sign.
        let want = mat(&c["params"]);
        assert_eq!(res.params.dim(), want.dim(), "{name}: params shape");
        let (p, ncol) = res.params.dim();
        for j in 0..ncol {
            // Choose the sign that aligns the columns (sign-free comparison).
            let mut dot = 0.0;
            for i in 0..p {
                dot += res.params[[i, j]] * want[[i, j]];
            }
            let sign = if dot < 0.0 { -1.0 } else { 1.0 };
            for i in 0..p {
                let g = sign * res.params[[i, j]];
                let w = want[[i, j]];
                let e = rel(g, w);
                assert!(
                    e <= 1e-6,
                    "{name}.params[{i}][{j}]: rel-err {e:.3e} (got {g}, want {w})"
                );
            }
        }
    }
}

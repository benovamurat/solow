//! Cross-validation of the VAR estimator against golden reference values
//! frozen in `tests/fixtures/var.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_var::Var;
use std::fs;

fn load() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/var.json");
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_var.py)");
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

/// 3-D array `(p, K, K)` from nested JSON, returned as a vector of `K x K`
/// matrices.
fn cube(v: &Value) -> Vec<Array2<f64>> {
    v.as_array().unwrap().iter().map(mat).collect()
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

fn check_vec(label: &str, got: &Array1<f64>, want: &Array1<f64>, tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for i in 0..got.len() {
        let e = rel(got[i], want[i]);
        assert!(
            e <= tol,
            "{label}[{i}]: rel-err {e:.3e} (got {}, want {})",
            got[i],
            want[i]
        );
    }
}

fn check_mat(label: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) {
    assert_eq!(got.dim(), want.dim(), "{label}: shape");
    let (m, n) = got.dim();
    for i in 0..m {
        for j in 0..n {
            let e = rel(got[[i, j]], want[[i, j]]);
            assert!(
                e <= tol,
                "{label}[{i}][{j}]: rel-err {e:.3e} (got {}, want {})",
                got[[i, j]],
                want[[i, j]]
            );
        }
    }
}

#[test]
fn var_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let y = mat(&c["endog"]);
        let p = c["p"].as_u64().unwrap() as usize;
        let res = Var::new(y).unwrap().fit(p).unwrap();
        let exp = &c["expected"];

        // Integer/structural quantities.
        assert_eq!(
            res.nobs as u64,
            exp["nobs"].as_u64().unwrap(),
            "{name}: nobs"
        );
        assert_eq!(
            res.df_model as u64,
            exp["df_model"].as_u64().unwrap(),
            "{name}: df_model"
        );
        assert_eq!(
            res.df_resid as u64,
            exp["df_resid"].as_u64().unwrap(),
            "{name}: df_resid"
        );
        assert_eq!(
            res.neqs as u64,
            exp["neqs"].as_u64().unwrap(),
            "{name}: neqs"
        );
        assert_eq!(
            res.k_ar as u64,
            exp["k_ar"].as_u64().unwrap(),
            "{name}: k_ar"
        );

        // Closed-form OLS quantities: machine precision (<= 1e-8).
        check_mat(
            &format!("{name}.params"),
            &res.params,
            &mat(&exp["params"]),
            1e-8,
        );
        check_vec(
            &format!("{name}.intercept"),
            &res.intercept,
            &vec1(&exp["intercept"]),
            1e-8,
        );
        check_mat(
            &format!("{name}.fittedvalues"),
            &res.fittedvalues,
            &mat(&exp["fittedvalues"]),
            1e-8,
        );
        check_mat(
            &format!("{name}.resid"),
            &res.resid,
            &mat(&exp["resid"]),
            1e-8,
        );
        check_mat(
            &format!("{name}.sigma_u"),
            &res.sigma_u,
            &mat(&exp["sigma_u"]),
            1e-8,
        );
        check_mat(
            &format!("{name}.sigma_u_mle"),
            &res.sigma_u_mle,
            &mat(&exp["sigma_u_mle"]),
            1e-8,
        );

        // Coefficient matrices A_1, ..., A_p.
        let want_coefs = cube(&exp["coefs"]);
        assert_eq!(
            res.coefs.len(),
            want_coefs.len(),
            "{name}: number of coef matrices"
        );
        for (i, (g, w)) in res.coefs.iter().zip(want_coefs.iter()).enumerate() {
            check_mat(&format!("{name}.coefs[{i}]"), g, w, 1e-8);
        }

        // Standard errors and derived inference.
        check_mat(&format!("{name}.bse"), &res.bse, &mat(&exp["bse"]), 1e-8);
        check_mat(
            &format!("{name}.tvalues"),
            &res.tvalues,
            &mat(&exp["tvalues"]),
            1e-8,
        );
        check_mat(
            &format!("{name}.pvalues"),
            &res.pvalues,
            &mat(&exp["pvalues"]),
            1e-6,
        );

        // Likelihood and information criteria.
        check_scalar(
            &format!("{name}.llf"),
            res.llf,
            exp["llf"].as_f64().unwrap(),
            1e-8,
        );
        check_scalar(
            &format!("{name}.aic"),
            res.aic,
            exp["aic"].as_f64().unwrap(),
            1e-8,
        );
        check_scalar(
            &format!("{name}.bic"),
            res.bic,
            exp["bic"].as_f64().unwrap(),
            1e-8,
        );
        check_scalar(
            &format!("{name}.hqic"),
            res.hqic,
            exp["hqic"].as_f64().unwrap(),
            1e-8,
        );
        check_scalar(
            &format!("{name}.fpe"),
            res.fpe,
            exp["fpe"].as_f64().unwrap(),
            1e-8,
        );
    }
}

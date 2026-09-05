//! Cross-validation of the extended regression estimators (QuantReg,
//! RollingOLS, RecursiveLS) against golden reference values frozen in
//! `tests/fixtures/regression_ext.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_regression::{QuantReg, RecursiveLS, RollingOLS};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/regression_ext.json"
    );
    let s =
        fs::read_to_string(p).expect("fixture present (run tools/reference/gen_regression_ext.py)");
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

fn check_mat(label: &str, got: &Array2<f64>, want: &Array2<f64>, key: &str, tol: f64) {
    assert_eq!(got.dim(), want.dim(), "{label}.{key}: shape mismatch");
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

fn check_scalar(label: &str, got: f64, want: f64, key: &str, tol: f64) {
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{label}.{key}: rel-err {e:.3e} (got {got}, want {want})"
    );
}

#[test]
fn quantreg_matches_reference() {
    let fx = load();
    for c in fx["quantreg"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        let model = QuantReg::new(y, x).unwrap();
        for f in c["fits"].as_array().unwrap() {
            let q = f["q"].as_f64().unwrap();
            let res = model.fit(q).unwrap();
            let label = format!("{name}@q={q}");
            // The IRLS recursion reproduces the reference bit-for-bit, so every
            // quantity matches to near machine precision. Assertions carry a
            // safety margin above the observed ~1e-15 (see crate notes); they are
            // far tighter than the spec floors (params 1e-6, bse 1e-5).
            check_vec(&label, &res.params, f, "params", 1e-9);
            // Robust sparsity/kernel sandwich standard errors.
            check_vec(&label, &res.bse, f, "bse", 1e-9);
            // Auxiliary quantities (sparsity = 1/f̂₀ and the kernel bandwidth).
            check_scalar(
                &label,
                res.sparsity,
                f["sparsity"].as_f64().unwrap(),
                "sparsity",
                1e-9,
            );
            check_scalar(
                &label,
                res.bandwidth,
                f["bandwidth"].as_f64().unwrap(),
                "bandwidth",
                1e-10,
            );
            assert!((res.q - q).abs() < 1e-15, "{label}: q stored");
        }
    }
}

#[test]
fn rolling_ols_matches_reference() {
    let fx = load();
    for c in fx["rolling"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        let window = c["window"].as_u64().unwrap() as usize;
        let res = RollingOLS::new(y, x, window).unwrap().fit().unwrap();
        let want = mat(&c["params"]);
        // Per-window OLS coefficients: closed-form, matched to ~1e-15.
        check_mat(name, &res.params, &want, "params", 1e-12);
        // Window-end alignment.
        let ends = c["window_ends"].as_array().unwrap();
        assert_eq!(res.window_ends.len(), ends.len(), "{name}: n_windows");
        for (i, e) in ends.iter().enumerate() {
            assert_eq!(
                res.window_ends[i],
                e.as_u64().unwrap() as usize,
                "{name}: end[{i}]"
            );
        }
    }
}

#[test]
fn recursive_ls_matches_reference() {
    let fx = load();
    for c in fx["recursive"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let y = vec1(&c["endog"]);
        let x = mat(&c["exog"]);
        let res = RecursiveLS::new(y, x).unwrap().fit().unwrap();

        // The exact-diffuse Kalman recursion reproduces the reference to ~1e-15;
        // assertions keep a safety margin (spec floor for params/llf is 1e-7).
        // Final and recursive coefficients.
        check_vec(name, &res.params, c, "params", 1e-10);
        let rec_want = mat(&c["recursive_coefficients"]);
        check_mat(
            name,
            &res.recursive_coefficients,
            &rec_want,
            "recursive_coefficients",
            1e-10,
        );

        // Recursive residuals and structural-stability statistics.
        check_vec(name, &res.resid_recursive, c, "resid_recursive", 1e-10);
        check_vec(name, &res.cusum, c, "cusum", 1e-10);
        check_vec(name, &res.cusum_squares, c, "cusum_squares", 1e-10);

        // Concentrated log-likelihood and scale.
        check_scalar(name, res.llf, c["llf"].as_f64().unwrap(), "llf", 1e-10);
        check_vec(name, &res.llf_obs, c, "llf_obs", 1e-10);
        check_scalar(
            name,
            res.scale,
            c["scale"].as_f64().unwrap(),
            "scale",
            1e-10,
        );
        check_scalar(
            name,
            res.nobs_diffuse as f64,
            c["nobs_diffuse"].as_f64().unwrap(),
            "nobs_diffuse",
            1e-12,
        );
    }
}

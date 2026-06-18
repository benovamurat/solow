//! Cross-validation of the regression models against golden reference values
//! frozen in `tests/fixtures/models.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_regression::{LinearModel, LinearResults};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/models.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_models.py)");
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

fn verify(label: &str, res: &LinearResults, exp: &Value) {
    check_vec(label, &res.params, exp, "params", 1e-8);
    check_vec(label, &res.bse, exp, "bse", 1e-8);
    check_vec(label, &res.tvalues, exp, "tvalues", 1e-8);
    check_vec(label, &res.pvalues, exp, "pvalues", 1e-7);
    check_vec(label, &res.fittedvalues, exp, "fittedvalues", 1e-8);
    check_vec(label, &res.resid, exp, "resid", 1e-8);

    check_scalar(label, res.rsquared, exp, "rsquared", 1e-9);
    check_scalar(label, res.rsquared_adj, exp, "rsquared_adj", 1e-9);
    check_scalar(label, res.fvalue, exp, "fvalue", 1e-8);
    check_scalar(label, res.f_pvalue, exp, "f_pvalue", 1e-7);
    check_scalar(label, res.llf, exp, "llf", 1e-8);
    check_scalar(label, res.aic, exp, "aic", 1e-8);
    check_scalar(label, res.bic, exp, "bic", 1e-8);
    check_scalar(label, res.scale, exp, "scale", 1e-9);
    check_scalar(label, res.ssr, exp, "ssr", 1e-9);
    check_scalar(label, res.ess, exp, "ess", 1e-9);
    check_scalar(label, res.centered_tss, exp, "centered_tss", 1e-9);
    check_scalar(label, res.uncentered_tss, exp, "uncentered_tss", 1e-9);
    check_scalar(label, res.df_model, exp, "df_model", 1e-12);
    check_scalar(label, res.df_resid, exp, "df_resid", 1e-12);
    check_scalar(label, res.mse_model, exp, "mse_model", 1e-9);
    check_scalar(label, res.mse_resid, exp, "mse_resid", 1e-9);
    check_scalar(label, res.mse_total, exp, "mse_total", 1e-9);
    check_scalar(label, res.nobs, exp, "nobs", 1e-12);

    // Confidence interval (alpha = 0.05).
    let ci = res.conf_int(0.05);
    let want = mat(&exp["conf_int"]);
    for i in 0..res.params.len() {
        for j in 0..2 {
            let e = rel(ci[[i, j]], want[[i, j]]);
            assert!(e <= 1e-7, "{label}.conf_int[{i}][{j}]: rel-err {e:.3e}");
        }
    }
}

#[test]
fn regression_matches_reference() {
    let fx = load();
    for f in fx["fixtures"].as_array().unwrap() {
        let model = f["model"].as_str().unwrap();
        let y = vec1(&f["endog"]);
        let x = mat(&f["exog"]);
        let label = model.to_string();
        let res = match model {
            "ols" => LinearModel::ols(y, x).unwrap().fit().unwrap(),
            "wls" => {
                let w = vec1(&f["weights"]);
                LinearModel::wls(y, x, w).unwrap().fit().unwrap()
            }
            "gls" => {
                let sigma = mat(&f["sigma"]);
                LinearModel::gls(y, x, &sigma).unwrap().fit().unwrap()
            }
            other => panic!("unknown model {other}"),
        };
        verify(&label, &res, &f["expected"]);
    }
}

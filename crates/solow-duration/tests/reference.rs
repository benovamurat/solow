//! Cross-validation of the survival estimators against golden reference values
//! frozen in `tests/fixtures/duration.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_duration::{PHReg, SurvfuncRight};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/duration.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_duration.py)");
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

/// Read a numeric value that may be encoded as the sentinel string "nan".
fn num(v: &Value) -> f64 {
    match v {
        Value::String(s) => match s.as_str() {
            "nan" => f64::NAN,
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            other => other.parse().unwrap(),
        },
        _ => v.as_f64().unwrap(),
    }
}

fn slice(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / (1.0 + want.abs())
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

/// Compare a vector that may contain NaN entries (Greenwood SE at exhausted
/// risk sets). NaN must align in both vectors; finite entries match tightly.
fn check_vec_nan(label: &str, got: &Array1<f64>, want: &[f64], key: &str, tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}.{key}: length");
    for i in 0..got.len() {
        if want[i].is_nan() {
            assert!(
                got[i].is_nan(),
                "{label}.{key}[{i}]: expected NaN, got {}",
                got[i]
            );
        } else {
            assert!(
                !got[i].is_nan(),
                "{label}.{key}[{i}]: got NaN, want {}",
                want[i]
            );
            let e = rel(got[i], want[i]);
            assert!(
                e <= tol,
                "{label}.{key}[{i}]: rel-err {e:.3e} (got {}, want {})",
                got[i],
                want[i]
            );
        }
    }
}

#[test]
fn km_matches_reference() {
    let fx = load();
    for c in fx["km"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let time = slice(&c["time"]);
        let status = slice(&c["status"]);
        let s = SurvfuncRight::new(&time, &status).unwrap();
        let exp = &c["expected"];

        // Survival times are exact (selected unique input times).
        check_vec(name, &s.surv_times, exp, "surv_times", 1e-12);
        check_vec(name, &s.n_risk, exp, "n_risk", 1e-12);
        check_vec(name, &s.n_events, exp, "n_events", 1e-12);
        // Product-limit probabilities and Greenwood SE.
        check_vec(name, &s.surv_prob, exp, "surv_prob", 1e-10);
        let se_want = slice(&exp["surv_prob_se"]);
        check_vec_nan(name, &s.surv_prob_se, &se_want, "surv_prob_se", 1e-10);
    }
}

#[test]
fn cox_matches_reference() {
    let fx = load();
    for c in fx["cox"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let time = slice(&c["time"]);
        let status = slice(&c["status"]);
        let exog = mat(&c["exog"]);
        let model = PHReg::new(&time, &exog, &status).unwrap();
        let res = model.fit().unwrap();
        assert!(res.converged, "{name}: did not converge");
        let exp = &c["expected"];

        // Coefficients and partial log-likelihood: tight (no distribution
        // inverse in the loop).
        check_vec(name, &res.params, exp, "params", 1e-7);
        let llf_want = exp["llf"].as_f64().unwrap();
        assert!(
            rel(res.llf, llf_want) <= 1e-7,
            "{name}.llf: got {}, want {}",
            res.llf,
            llf_want
        );

        // Standard errors / z-statistics derive from a matrix inverse.
        check_vec(name, &res.bse, exp, "bse", 1e-6);
        check_vec(name, &res.tvalues, exp, "tvalues", 1e-6);
        // p-values use the normal inverse-CDF tail.
        check_vec(name, &res.pvalues, exp, "pvalues", 1e-6);
    }
}

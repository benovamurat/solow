//! Cross-validation of the duration-crate extensions (survdiff log-rank tests,
//! Efron-tie Cox PH regression, and competing-risks cumulative incidence)
//! against golden reference values frozen in `tests/fixtures/duration_ext.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_duration::{survdiff, CumIncidenceRight, PHRegTies, Ties, WeightType};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/duration_ext.json"
    );
    let s =
        fs::read_to_string(p).expect("fixture present (run tools/reference/gen_duration_ext.py)");
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

/// Compare a vector that may contain NaN entries. NaN must align in both
/// vectors; finite entries match tightly.
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
fn survdiff_matches_reference() {
    let fx = load();
    let specs: [(&str, WeightType); 5] = [
        ("logrank", WeightType::LogRank),
        ("gb", WeightType::GehanBreslow),
        ("tw", WeightType::TaroneWare),
        ("fh1", WeightType::FlemingHarrington(1.0)),
        ("fh05", WeightType::FlemingHarrington(0.5)),
    ];
    for c in fx["survdiff"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let time = slice(&c["time"]);
        let status = slice(&c["status"]);
        let group = slice(&c["group"]);
        let exp = &c["expected"];
        for (label, wt) in specs {
            let res = survdiff(&time, &status, &group, wt).unwrap();
            let want_chisq = exp[label]["chisq"].as_f64().unwrap();
            let want_pv = exp[label]["pvalue"].as_f64().unwrap();
            // Log-rank chisq is closed-form; assert at 1e-7.
            assert!(
                rel(res.chisq, want_chisq) <= 1e-7,
                "{name}.{label}.chisq: got {}, want {}",
                res.chisq,
                want_chisq
            );
            // p-value goes through the chi-square CDF; assert at 1e-6.
            assert!(
                rel(res.pvalue, want_pv) <= 1e-6,
                "{name}.{label}.pvalue: got {}, want {}",
                res.pvalue,
                want_pv
            );
        }
    }
}

#[test]
fn cox_efron_matches_reference() {
    let fx = load();
    for c in fx["cox_efron"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let time = slice(&c["time"]);
        let status = slice(&c["status"]);
        let exog = mat(&c["exog"]);
        let model = PHRegTies::new(&time, &exog, &status, Ties::Efron).unwrap();
        let res = model.fit().unwrap();
        assert!(res.converged, "{name}: did not converge");
        let exp = &c["expected"];

        // MLE parameters and partial log-likelihood: tight (1e-6).
        check_vec(name, &res.params, exp, "params", 1e-6);
        let llf_want = exp["llf"].as_f64().unwrap();
        assert!(
            rel(res.llf, llf_want) <= 1e-6,
            "{name}.llf: got {}, want {}",
            res.llf,
            llf_want
        );

        // Standard errors / z-statistics from a matrix inverse.
        check_vec(name, &res.bse, exp, "bse", 1e-6);
        check_vec(name, &res.tvalues, exp, "tvalues", 1e-6);
        // p-values use the normal tail inverse.
        check_vec(name, &res.pvalues, exp, "pvalues", 1e-6);
    }
}

#[test]
fn cuminc_matches_reference() {
    let fx = load();
    for c in fx["cuminc"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let time = slice(&c["time"]);
        let status = slice(&c["status"]);
        let ci = CumIncidenceRight::new(&time, &status).unwrap();
        let exp = &c["expected"];

        // Reported times are exact (selected unique input times).
        check_vec(name, &ci.times, exp, "times", 1e-12);

        let cinc = exp["cinc"].as_array().unwrap();
        let cinc_se = exp["cinc_se"].as_array().unwrap();
        assert_eq!(ci.cinc.len(), cinc.len(), "{name}: number of causes");
        for j in 0..ci.cinc.len() {
            // Cumulative incidence probabilities: closed-form, 1e-7.
            let cinc_want = vec1(&cinc[j]);
            assert_eq!(
                ci.cinc[j].len(),
                cinc_want.len(),
                "{name}.cinc[{j}]: length"
            );
            for i in 0..ci.cinc[j].len() {
                let e = rel(ci.cinc[j][i], cinc_want[i]);
                assert!(
                    e <= 1e-7,
                    "{name}.cinc[{j}][{i}]: rel-err {e:.3e} (got {}, want {})",
                    ci.cinc[j][i],
                    cinc_want[i]
                );
            }
            // Standard errors may contain NaN once the risk set is exhausted.
            let se_want = slice(&cinc_se[j]);
            check_vec_nan(
                name,
                &ci.cinc_se[j],
                &se_want,
                &format!("cinc_se[{j}]"),
                1e-7,
            );
        }
    }
}

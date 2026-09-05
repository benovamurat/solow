//! Cross-validation of the time-series-analysis extensions (KPSS, theoretical
//! ARMA acf/pacf, seasonal decomposition and Holt-Winters exponential
//! smoothing) against golden reference values frozen in
//! `tests/fixtures/tsa_ext.json`.

use ndarray::Array1;
use serde_json::Value;
use solow_tsa::{
    arma_acf, arma_acovf, arma_pacf, kpss, seasonal_decompose, ExponentialSmoothing, Holt,
    KpssLags, KpssRegression, Seasonal, SeasonalModel, SimpleExpSmoothing,
};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/tsa_ext.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_tsa_ext.py)");
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

fn check(label: &str, got: f64, want: f64, tol: f64) {
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{label}: rel-err {e:.3e} (got {got}, want {want})"
    );
}

/// Compare a got-vector against a fixture array that may contain `null` (for
/// NaN entries). `null` entries require the got value to be NaN.
fn check_opt_vec(label: &str, got: &Array1<f64>, exp: &Value, tol: f64) {
    let arr = exp.as_array().unwrap();
    assert_eq!(got.len(), arr.len(), "{label}: length");
    for (i, e) in arr.iter().enumerate() {
        if e.is_null() {
            assert!(
                got[i].is_nan(),
                "{label}[{i}]: expected NaN, got {}",
                got[i]
            );
        } else {
            let want = e.as_f64().unwrap();
            assert!(got[i].is_finite(), "{label}[{i}]: expected finite, got NaN");
            check(&format!("{label}[{i}]"), got[i], want, tol);
        }
    }
}

fn check_vec(label: &str, got: &Array1<f64>, exp: &Value, tol: f64) {
    let want = vec1(exp);
    assert_eq!(got.len(), want.len(), "{label}: length");
    for i in 0..got.len() {
        check(&format!("{label}[{i}]"), got[i], want[i], tol);
    }
}

#[test]
fn kpss_matches_reference() {
    let fx = load();
    for c in fx["kpss"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x = vec1(&c["x"]);
        let reg = KpssRegression::parse(c["regression"].as_str().unwrap()).unwrap();
        let nlags = match &c["nlags"] {
            Value::String(s) if s == "auto" => KpssLags::Auto,
            Value::String(s) if s == "legacy" => KpssLags::Legacy,
            v => KpssLags::Fixed(v.as_u64().unwrap() as usize),
        };
        let r = kpss(&x, reg, nlags).unwrap();
        // KPSS statistic: closed-form, tight (matches to ~1e-13).
        check(
            &format!("kpss[{name}].stat"),
            r.stat,
            c["stat"].as_f64().unwrap(),
            1e-12,
        );
        // Truncation lag must match exactly.
        assert_eq!(
            r.lags as u64,
            c["lags"].as_u64().unwrap(),
            "kpss[{name}].lags"
        );
        // p-value comes from a table interpolation -> tight.
        check(
            &format!("kpss[{name}].pvalue"),
            r.pvalue,
            c["pvalue"].as_f64().unwrap(),
            1e-8,
        );
        // Critical values are constants.
        let crit = c["crit"].as_array().unwrap();
        for (j, cv) in crit.iter().enumerate().take(4) {
            check(
                &format!("kpss[{name}].crit[{j}]"),
                r.crit_values[j],
                cv.as_f64().unwrap(),
                1e-12,
            );
        }
    }
}

#[test]
fn arma_acf_pacf_matches_reference() {
    let fx = load();
    for c in fx["arma"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let ar: Vec<f64> = vec1(&c["ar"]).to_vec();
        let ma: Vec<f64> = vec1(&c["ma"]).to_vec();
        let lags = c["lags"].as_u64().unwrap() as usize;
        let acovf = arma_acovf(&ar, &ma, lags, 1.0).unwrap();
        let acf = arma_acf(&ar, &ma, lags).unwrap();
        let pacf = arma_pacf(&ar, &ma, lags).unwrap();
        check_vec(&format!("arma[{name}].acovf"), &acovf, &c["acovf"], 1e-10);
        check_vec(&format!("arma[{name}].acf"), &acf, &c["acf"], 1e-10);
        check_vec(&format!("arma[{name}].pacf"), &pacf, &c["pacf"], 1e-10);
    }
}

#[test]
fn seasonal_decompose_matches_reference() {
    let fx = load();
    for c in fx["seasonal"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x = vec1(&c["x"]);
        let period = c["period"].as_u64().unwrap() as usize;
        let model = SeasonalModel::parse(c["model"].as_str().unwrap()).unwrap();
        let r = seasonal_decompose(&x, period, model).unwrap();
        check_opt_vec(
            &format!("seasonal[{name}].trend"),
            &r.trend,
            &c["trend"],
            1e-10,
        );
        check_opt_vec(
            &format!("seasonal[{name}].seasonal"),
            &r.seasonal,
            &c["seasonal"],
            1e-10,
        );
        check_opt_vec(
            &format!("seasonal[{name}].resid"),
            &r.resid,
            &c["resid"],
            1e-10,
        );
    }
}

#[test]
fn holtwinters_matches_reference() {
    let fx = load();
    // The smoothing parameters are found by a separate bound-constrained
    // optimizer (BFGS over a reparameterised feasible region) seeded by a grid
    // search. Both implementations reach the same optimum; tiny optimizer
    // differences leave the worst-case relative error at ~4e-7, so we assert at
    // 1e-6 (params, sse, fitted, forecast all clear this).
    let param_tol = 1e-6;
    let value_tol = 1e-6;
    for c in fx["holtwinters"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let kind = c["kind"].as_str().unwrap();
        let x = vec1(&c["x"]);
        let period = c["period"].as_u64().unwrap() as usize;
        let horizon = c["horizon"].as_u64().unwrap() as usize;

        let res = match kind {
            "ses" => SimpleExpSmoothing::new(x.clone()).fit().unwrap(),
            "holt" => Holt::new(x.clone()).fit().unwrap(),
            "hw_add" => {
                ExponentialSmoothing::new(x.clone(), true, Some(Seasonal::Additive), period)
                    .unwrap()
                    .fit()
                    .unwrap()
            }
            "hw_mul" => {
                ExponentialSmoothing::new(x.clone(), true, Some(Seasonal::Multiplicative), period)
                    .unwrap()
                    .fit()
                    .unwrap()
            }
            other => panic!("unknown kind {other}"),
        };

        // Initial states are fixed by the deterministic heuristic -> tight.
        check(
            &format!("hw[{name}].initial_level"),
            res.initial_level,
            c["initial_level"].as_f64().unwrap(),
            1e-8,
        );
        if c["initial_trend"].as_f64().unwrap().abs() > 0.0 || res.beta != 0.0 {
            check(
                &format!("hw[{name}].initial_trend"),
                res.initial_trend,
                c["initial_trend"].as_f64().unwrap(),
                1e-8,
            );
        }
        if !c["initial_seasons"].as_array().unwrap().is_empty() {
            check_vec(
                &format!("hw[{name}].initial_seasons"),
                &res.initial_seasons,
                &c["initial_seasons"],
                1e-8,
            );
        }

        // Smoothing parameters and SSE: optimizer-limited.
        check(
            &format!("hw[{name}].alpha"),
            res.alpha,
            c["alpha"].as_f64().unwrap(),
            param_tol,
        );
        check(
            &format!("hw[{name}].beta"),
            res.beta,
            c["beta"].as_f64().unwrap(),
            param_tol,
        );
        check(
            &format!("hw[{name}].gamma"),
            res.gamma,
            c["gamma"].as_f64().unwrap(),
            param_tol,
        );
        check(
            &format!("hw[{name}].sse"),
            res.sse,
            c["sse"].as_f64().unwrap(),
            value_tol,
        );

        // Fitted values and forecasts.
        check_vec(
            &format!("hw[{name}].fittedvalues"),
            &res.fittedvalues,
            &c["fittedvalues"],
            value_tol,
        );
        let fc = res.forecast(horizon);
        check_vec(
            &format!("hw[{name}].forecast"),
            &fc,
            &c["forecast"],
            value_tol,
        );
    }
}

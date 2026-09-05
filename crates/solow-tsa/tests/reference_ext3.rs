//! Cross-validation of the time-series-analysis extensions (set 3): the
//! Hodrick-Prescott, Baxter-King and Christiano-Fitzgerald filters,
//! autoregressive order selection, deterministic-term construction, and the
//! innovations algorithm. Golden values are frozen in
//! `tests/fixtures/tsa_ext3.json`.
#![allow(clippy::needless_range_loop)]

use ndarray::Array1;
use serde_json::Value;
use solow_tsa::{
    ar_select_order, bkfilter, cffilter, hpfilter, innovations_algo, ArIc, DeterministicProcess,
    Trend,
};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/tsa_ext3.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_tsa_ext3.py)");
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

fn check_vec(label: &str, got: &Array1<f64>, want: &Array1<f64>, tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for i in 0..got.len() {
        check(&format!("{label}[{i}]"), got[i], want[i], tol);
    }
}

fn parse_trend(s: &str) -> Trend {
    match s {
        "n" => Trend::N,
        "c" => Trend::C,
        "t" => Trend::T,
        "ct" => Trend::Ct,
        "ctt" => Trend::Ctt,
        other => panic!("unknown trend {other}"),
    }
}

#[test]
fn hpfilter_matches_reference() {
    // The HP filter is the exact ridge solution trend = (I + lamb K'K)^{-1} x.
    // Our dense symmetric solve reproduces the reference sparse solve to well
    // under 1e-7 (the achieved error is ~1e-10 or better); we assert at 1e-7.
    let fx = load();
    for c in fx["hp"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x = vec1(&c["x"]);
        let lamb = c["lamb"].as_f64().unwrap();
        let (cycle, trend) = hpfilter(&x, lamb).unwrap();
        check_vec(
            &format!("hp[{name}].cycle"),
            &cycle,
            &vec1(&c["cycle"]),
            1e-7,
        );
        check_vec(
            &format!("hp[{name}].trend"),
            &trend,
            &vec1(&c["trend"]),
            1e-7,
        );
    }
}

#[test]
fn bkfilter_matches_reference() {
    // Closed-form weighted moving average; matches to ~1e-12 in practice. We
    // assert the band-pass cycle at 1e-8.
    let fx = load();
    for c in fx["bk"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x = vec1(&c["x"]);
        let low = c["low"].as_f64().unwrap();
        let high = c["high"].as_f64().unwrap();
        let k = c["K"].as_u64().unwrap() as usize;
        let cycle = bkfilter(&x, low, high, k).unwrap();
        check_vec(
            &format!("bk[{name}].cycle"),
            &cycle,
            &vec1(&c["cycle"]),
            1e-8,
        );
    }
}

#[test]
fn cffilter_matches_reference() {
    // Closed-form asymmetric filter; matches to ~1e-12 in practice. We assert
    // both cycle and trend at 1e-8.
    let fx = load();
    for c in fx["cf"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x = vec1(&c["x"]);
        let low = c["low"].as_f64().unwrap();
        let high = c["high"].as_f64().unwrap();
        let drift = c["drift"].as_bool().unwrap();
        let (cycle, trend) = cffilter(&x, low, high, drift).unwrap();
        check_vec(
            &format!("cf[{name}].cycle"),
            &cycle,
            &vec1(&c["cycle"]),
            1e-8,
        );
        check_vec(
            &format!("cf[{name}].trend"),
            &trend,
            &vec1(&c["trend"]),
            1e-8,
        );
    }
}

#[test]
fn ar_select_order_matches_reference() {
    // The IC path is a closed-form transform of the OLS residual sum of
    // squares, so each value matches to ~1e-10. We assert at 1e-8 and require
    // the chosen order to be exact.
    let fx = load();
    for c in fx["ar_select"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let y = vec1(&c["y"]);
        let maxlag = c["maxlag"].as_u64().unwrap() as usize;
        let ic = ArIc::parse(c["ic"].as_str().unwrap()).unwrap();
        let trend = parse_trend(c["trend"].as_str().unwrap());
        let res = ar_select_order(&y, maxlag, ic, trend).unwrap();

        let want_path = vec1(&c["ic_path"]);
        check_vec(
            &format!("ar[{name}].ic_path"),
            &res.ic_path,
            &want_path,
            1e-8,
        );

        let want_order = c["selected_order"].as_u64().unwrap() as usize;
        assert_eq!(
            res.selected_order, want_order,
            "ar[{name}].selected_order: got {} want {}",
            res.selected_order, want_order
        );
    }
}

#[test]
fn deterministic_process_matches_reference() {
    // Deterministic terms are integers / exact products; assert at 1e-10.
    let fx = load();
    for c in fx["deterministic"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let steps = c["steps"].as_u64().unwrap() as usize;
        let constant = c["constant"].as_bool().unwrap();
        let order = c["order"].as_u64().unwrap() as usize;
        let seasonal = c["seasonal"].as_bool().unwrap();
        let period = c["period"].as_u64().unwrap() as usize;
        let dp = DeterministicProcess::new(steps, constant, order, seasonal, period).unwrap();
        let terms = dp.in_sample();

        let exp_rows = c["terms"].as_array().unwrap();
        assert_eq!(terms.nrows(), exp_rows.len(), "dp[{name}] rows");
        for (i, row) in exp_rows.iter().enumerate() {
            let want = vec1(row);
            assert_eq!(terms.ncols(), want.len(), "dp[{name}] row {i} cols");
            for j in 0..want.len() {
                check(
                    &format!("dp[{name}].terms[{i}][{j}]"),
                    terms[[i, j]],
                    want[j],
                    1e-10,
                );
            }
        }
    }
}

#[test]
fn innovations_algo_matches_reference() {
    // The innovations recursion is a deterministic linear recursion that
    // reproduces the reference theta / sigma2 to machine precision; assert at
    // 1e-8.
    let fx = load();
    for c in fx["innovations"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let acovf = vec1(&c["acovf"]);
        let nobs = c["nobs"].as_u64().unwrap() as usize;
        let res = innovations_algo(&acovf, nobs).unwrap();

        // sigma2 (v) sequence.
        check_vec(
            &format!("innov[{name}].sigma2"),
            &res.sigma2,
            &vec1(&c["sigma2"]),
            1e-8,
        );

        // theta matrix, row by row.
        let exp_rows = c["theta"].as_array().unwrap();
        assert_eq!(
            res.theta.nrows(),
            exp_rows.len(),
            "innov[{name}] theta rows"
        );
        for (i, row) in exp_rows.iter().enumerate() {
            let want = vec1(row);
            assert_eq!(
                res.theta.ncols(),
                want.len(),
                "innov[{name}] theta row {i} cols"
            );
            for j in 0..want.len() {
                check(
                    &format!("innov[{name}].theta[{i}][{j}]"),
                    res.theta[[i, j]],
                    want[j],
                    1e-8,
                );
            }
        }
    }
}

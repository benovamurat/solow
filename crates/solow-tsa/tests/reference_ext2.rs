//! Cross-validation of the time-series-analysis extensions (set 2): Granger
//! causality tests, the Engle-Granger cointegration test, STL seasonal-trend
//! decomposition by loess, and ARMA order selection by information criterion.
//! Golden values are frozen in `tests/fixtures/tsa_ext2.json`.
#![allow(clippy::needless_range_loop)]

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_tsa::{
    arma_order_select_ic, coint, grangercausalitytests, AutoLag, CointTrend, InfoCriterion, Stl,
};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/tsa_ext2.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_tsa_ext2.py)");
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

#[test]
fn granger_matches_reference() {
    let fx = load();
    for c in fx["granger"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let cols = c["data"].as_array().unwrap();
        let y = vec1(&cols[0]);
        let x = vec1(&cols[1]);
        let n = y.len();
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = y[i];
            data[[i, 1]] = x[i];
        }
        let maxlag = c["maxlag"].as_u64().unwrap() as usize;
        let res = grangercausalitytests(&data, maxlag).unwrap();
        let exp_lags = c["lags"].as_array().unwrap();
        assert_eq!(res.len(), exp_lags.len(), "{name}: number of lags");
        for (r, e) in res.iter().zip(exp_lags) {
            let lab = format!("granger[{name}].lag{}", r.lag);
            assert_eq!(r.lag as u64, e["lag"].as_u64().unwrap(), "{lab}.lag");
            // ssr F test: F closed-form (1e-7), p via F-dist inverse (1e-7).
            let ef = e["ssr_ftest"].as_array().unwrap();
            check(
                &format!("{lab}.ssr_F"),
                r.ssr_ftest.0,
                ef[0].as_f64().unwrap(),
                1e-7,
            );
            check(
                &format!("{lab}.ssr_Fp"),
                r.ssr_ftest.1,
                ef[1].as_f64().unwrap(),
                1e-7,
            );
            assert_eq!(
                r.ssr_ftest.2 as u64,
                ef[2].as_u64().unwrap(),
                "{lab}.df_num"
            );
            assert_eq!(
                r.ssr_ftest.3 as u64,
                ef[3].as_u64().unwrap(),
                "{lab}.df_den"
            );
            // ssr chi2 test.
            let ec = e["ssr_chi2test"].as_array().unwrap();
            check(
                &format!("{lab}.chi2"),
                r.ssr_chi2test.0,
                ec[0].as_f64().unwrap(),
                1e-7,
            );
            check(
                &format!("{lab}.chi2p"),
                r.ssr_chi2test.1,
                ec[1].as_f64().unwrap(),
                1e-7,
            );
            assert_eq!(
                r.ssr_chi2test.2 as u64,
                ec[2].as_u64().unwrap(),
                "{lab}.chi2_df"
            );
            // lr test.
            let el = e["lrtest"].as_array().unwrap();
            check(
                &format!("{lab}.lr"),
                r.lrtest.0,
                el[0].as_f64().unwrap(),
                1e-7,
            );
            check(
                &format!("{lab}.lrp"),
                r.lrtest.1,
                el[1].as_f64().unwrap(),
                1e-7,
            );
            assert_eq!(r.lrtest.2 as u64, el[2].as_u64().unwrap(), "{lab}.lr_df");
        }
    }
}

#[test]
fn coint_matches_reference() {
    let fx = load();
    for c in fx["coint"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let y0 = vec1(&c["y0"]);
        let y1 = vec1(&c["y1"]);
        let trend = match c["trend"].as_str().unwrap() {
            "n" => CointTrend::N,
            "c" => CointTrend::C,
            "ct" => CointTrend::Ct,
            "ctt" => CointTrend::Ctt,
            other => panic!("unknown trend {other}"),
        };
        let maxlag = c["maxlag"].as_u64().unwrap() as usize;
        let r = coint(&y0, &y1, trend, maxlag, AutoLag::Aic).unwrap();
        // Statistic: ADF t-stat on residuals (1e-6).
        check(
            &format!("coint[{name}].stat"),
            r.stat,
            c["stat"].as_f64().unwrap(),
            1e-6,
        );
        // MacKinnon p-value (1e-4 per spec).
        check(
            &format!("coint[{name}].pvalue"),
            r.pvalue,
            c["pvalue"].as_f64().unwrap(),
            1e-4,
        );
        // Critical values (constants given nobs).
        let crit = c["crit"].as_array().unwrap();
        for (j, cv) in crit.iter().enumerate() {
            if cv.is_null() {
                assert!(
                    r.crit_values[j].is_nan(),
                    "coint[{name}].crit[{j}] expected NaN"
                );
            } else {
                check(
                    &format!("coint[{name}].crit[{j}]"),
                    r.crit_values[j],
                    cv.as_f64().unwrap(),
                    1e-8,
                );
            }
        }
    }
}

#[test]
fn stl_matches_reference() {
    // STL is computed by iterated loess. The port reproduces the reference
    // inner loop (cycle-subseries loess, the three moving-average low-pass
    // filter, and the trend loess) deterministically: every component matches
    // to ~1e-14 absolute in practice. We assert relative error at 1e-10, which
    // leaves a comfortable margin over the achieved accuracy.
    let fx = load();
    for c in fx["stl"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x = vec1(&c["x"]);
        let period = c["period"].as_u64().unwrap() as usize;
        let stl = Stl::new(x.clone(), period).unwrap();
        let r = stl.fit().unwrap();
        let exp_trend = vec1(&c["trend"]);
        let exp_season = vec1(&c["seasonal"]);
        let exp_resid = vec1(&c["resid"]);
        assert_eq!(r.trend.len(), exp_trend.len(), "stl[{name}] length");
        for i in 0..r.trend.len() {
            check(
                &format!("stl[{name}].trend[{i}]"),
                r.trend[i],
                exp_trend[i],
                1e-10,
            );
            check(
                &format!("stl[{name}].seasonal[{i}]"),
                r.seasonal[i],
                exp_season[i],
                1e-10,
            );
            check(
                &format!("stl[{name}].resid[{i}]"),
                r.resid[i],
                exp_resid[i],
                1e-10,
            );
        }
    }
}

#[test]
fn arma_order_select_ic_matches_reference() {
    // The IC grids come from the exact Gaussian (Kalman) ARMA likelihood, which
    // we reproduce bit-for-bit at fixed parameters. The remaining difference is
    // the optimiser's converged point; at the (flat) optimum the AIC/BIC agree
    // with the reference to ~1e-6 or better, so we assert at 1e-6.
    let fx = load();
    for c in fx["arma_order"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let y = vec1(&c["y"]);
        let max_ar = c["max_ar"].as_u64().unwrap() as usize;
        let max_ma = c["max_ma"].as_u64().unwrap() as usize;
        let res = arma_order_select_ic(
            &y,
            max_ar,
            max_ma,
            &[InfoCriterion::Aic, InfoCriterion::Bic],
        )
        .unwrap();

        let exp_aic = c["aic"].as_array().unwrap();
        let exp_bic = c["bic"].as_array().unwrap();
        for r in &res {
            let (exp_grid, kind) = match r.ic {
                InfoCriterion::Aic => (exp_aic, "aic"),
                InfoCriterion::Bic => (exp_bic, "bic"),
            };
            for p in 0..=max_ar {
                let row = exp_grid[p].as_array().unwrap();
                for q in 0..=max_ma {
                    let want = row[q].as_f64().unwrap();
                    check(
                        &format!("arma[{name}].{kind}[{p}][{q}]"),
                        r.grid[[p, q]],
                        want,
                        1e-6,
                    );
                }
            }
        }
        // argmin orders.
        let exp_aic_min = c["aic_min_order"].as_array().unwrap();
        let exp_bic_min = c["bic_min_order"].as_array().unwrap();
        for r in &res {
            let exp = match r.ic {
                InfoCriterion::Aic => exp_aic_min,
                InfoCriterion::Bic => exp_bic_min,
            };
            assert_eq!(
                (r.min_order.0 as u64, r.min_order.1 as u64),
                (exp[0].as_u64().unwrap(), exp[1].as_u64().unwrap()),
                "arma[{name}].{:?}_min_order",
                r.ic
            );
        }
    }
}

//! Cross-validation of the time-series-analysis primitives and the AutoReg
//! estimator against golden reference values frozen in
//! `tests/fixtures/tsa.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_tsa::{
    acf, acf_qstat, acovf, adfuller, ccf, pacf, q_stat, AdfRegression, AutoLag, AutoReg, Original,
    PacfMethod, Trend, Trim,
};
use std::fs;

fn load() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/tsa.json");
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_tsa.py)");
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
    let (m, n) = (rows.len(), if rows.is_empty() { 0 } else { rows[0].len() });
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

fn check_vec(label: &str, got: &Array1<f64>, want: &Array1<f64>, tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
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
    assert_eq!(got.dim(), want.dim(), "{label}: shape mismatch");
    for i in 0..got.nrows() {
        for j in 0..got.ncols() {
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
fn primitives_match_reference() {
    let fx = load();
    let prim = &fx["primitives"];
    let x = vec1(&prim["x"]);
    let y = vec1(&prim["y"]);
    let nobs = prim["nobs"].as_u64().unwrap() as usize;

    // acovf variants.
    for c in prim["acovf"].as_array().unwrap() {
        let adjusted = c["adjusted"].as_bool().unwrap();
        let demean = c["demean"].as_bool().unwrap();
        let nlag = c["nlag"].as_u64().unwrap() as usize;
        let got = acovf(&x, adjusted, demean, nlag).unwrap();
        let label = format!("acovf(adj={adjusted},demean={demean})");
        check_vec(&label, &got, &vec1(&c["values"]), 1e-10);
    }

    // acf + Ljung-Box qstat/pvalues.
    for c in prim["acf"].as_array().unwrap() {
        let adjusted = c["adjusted"].as_bool().unwrap();
        let nlags = c["nlags"].as_u64().unwrap() as usize;
        let (a, q, p) = acf_qstat(&x, nlags, adjusted).unwrap();
        let label = format!("acf(adj={adjusted})");
        check_vec(&label, &a, &vec1(&c["acf"]), 1e-10);
        check_vec(&format!("{label}.qstat"), &q, &vec1(&c["qstat"]), 1e-9);
        // p-values traverse the chi-squared survival function.
        check_vec(&format!("{label}.pvalues"), &p, &vec1(&c["pvalues"]), 1e-6);

        // acf alone agrees with the acf component.
        let a2 = acf(&x, nlags, adjusted).unwrap();
        check_vec(&format!("{label}.acf-only"), &a2, &vec1(&c["acf"]), 1e-10);
    }

    // q_stat directly.
    {
        let qc = &prim["q_stat"];
        let acf_in = vec1(&qc["acf_in"]);
        let (q, p) = q_stat(&acf_in, nobs);
        check_vec("q_stat.qstat", &q, &vec1(&qc["qstat"]), 1e-9);
        check_vec("q_stat.pvalues", &p, &vec1(&qc["pvalues"]), 1e-6);
    }

    // pacf yw and ols.
    for c in prim["pacf"].as_array().unwrap() {
        let method = match c["method"].as_str().unwrap() {
            "yw" => PacfMethod::YuleWalker,
            "ols" => PacfMethod::Ols,
            other => panic!("unknown pacf method {other}"),
        };
        let nlags = c["nlags"].as_u64().unwrap() as usize;
        let got = pacf(&x, nlags, method).unwrap();
        let label = format!("pacf({:?})", method);
        check_vec(&label, &got, &vec1(&c["values"]), 1e-9);
    }

    // ccf.
    for c in prim["ccf"].as_array().unwrap() {
        let adjusted = c["adjusted"].as_bool().unwrap();
        let got = ccf(&x, &y, adjusted).unwrap();
        let label = format!("ccf(adj={adjusted})");
        check_vec(&label, &got, &vec1(&c["values"]), 1e-10);
    }
}

#[test]
fn lagmat_matches_reference() {
    let fx = load();
    let lm = &fx["lagmat"];
    let x = vec1(&lm["x"]);
    let xcol = x.view().insert_axis(ndarray::Axis(1)).to_owned();
    for c in lm["cases"].as_array().unwrap() {
        let maxlag = c["maxlag"].as_u64().unwrap() as usize;
        let trim = match c["trim"].as_str().unwrap() {
            "forward" => Trim::Forward,
            "backward" => Trim::Backward,
            "both" => Trim::Both,
            "none" => Trim::None,
            other => panic!("unknown trim {other}"),
        };
        let original = match c["original"].as_str().unwrap() {
            "ex" => Original::Ex,
            "sep" => Original::Sep,
            "in" => Original::In,
            other => panic!("unknown original {other}"),
        };
        let (lags, _leads) = solow_tsa::lagmat(&xcol, maxlag, trim, original).unwrap();
        let label = format!("lagmat(maxlag={maxlag},trim={:?})", trim);
        check_mat(&label, &lags, &mat(&c["values"]), 1e-12);
    }
}

#[test]
fn add_trend_matches_reference() {
    let fx = load();
    let at = &fx["add_trend"];
    let x = mat(&at["x"]);
    for c in at["cases"].as_array().unwrap() {
        let trend = Trend::parse(c["trend"].as_str().unwrap()).unwrap();
        let prepend = c["prepend"].as_bool().unwrap();
        let got = solow_tsa::add_trend(&x, trend, prepend);
        let label = format!("add_trend({:?},prepend={prepend})", trend);
        check_mat(&label, &got, &mat(&c["values"]), 1e-12);
    }
}

#[test]
fn adfuller_matches_reference() {
    let fx = load();
    for c in fx["adfuller"].as_array().unwrap() {
        let x = vec1(&c["x"]);
        let maxlag = c["maxlag"].as_u64().unwrap() as usize;
        let regression = AdfRegression::parse(c["regression"].as_str().unwrap()).unwrap();
        let autolag = match c["autolag"].as_str().unwrap() {
            "none" => AutoLag::None,
            "AIC" => AutoLag::Aic,
            "BIC" => AutoLag::Bic,
            other => panic!("unknown autolag {other}"),
        };
        let label = format!(
            "adfuller({},reg={},autolag={})",
            c["series"].as_str().unwrap(),
            c["regression"].as_str().unwrap(),
            c["autolag"].as_str().unwrap()
        );
        let res = adfuller(&x, maxlag, regression, autolag).unwrap();

        assert_eq!(
            res.usedlag,
            c["usedlag"].as_u64().unwrap() as usize,
            "{label}: usedlag"
        );
        assert_eq!(
            res.nobs,
            c["nobs"].as_u64().unwrap() as usize,
            "{label}: nobs"
        );
        // adf statistic: tight.
        let want_stat = c["adfstat"].as_f64().unwrap();
        assert!(
            rel(res.adfstat, want_stat) <= 1e-7,
            "{label}: adfstat rel-err {:.3e} (got {}, want {want_stat})",
            rel(res.adfstat, want_stat),
            res.adfstat
        );
        // critical values: closed-form polynomial in 1/nobs.
        let crit = [
            c["crit_1"].as_f64().unwrap(),
            c["crit_5"].as_f64().unwrap(),
            c["crit_10"].as_f64().unwrap(),
        ];
        for (i, &want) in crit.iter().enumerate() {
            assert!(rel(res.crit_values[i], want) <= 1e-9, "{label}: crit[{i}]");
        }
        // p-value: passes the test statistic through a polynomial and the
        // normal CDF; tight at 1e-6.
        let want_p = c["pvalue"].as_f64().unwrap();
        assert!(
            rel(res.pvalue, want_p) <= 1e-6,
            "{label}: pvalue rel-err {:.3e} (got {}, want {want_p})",
            rel(res.pvalue, want_p),
            res.pvalue
        );
        // information criterion (when an autolag was selected).
        if let Some(ic) = c["icbest"].as_f64() {
            let got_ic = res.icbest.expect("icbest present");
            assert!(rel(got_ic, ic) <= 1e-7, "{label}: icbest");
        }
    }
}

#[test]
fn autoreg_matches_reference() {
    let fx = load();
    for c in fx["autoreg"].as_array().unwrap() {
        let x = vec1(&c["x"]);
        let lags = c["lags"].as_u64().unwrap() as usize;
        let trend = Trend::parse(c["trend"].as_str().unwrap()).unwrap();
        let label = format!(
            "AutoReg({},lags={lags},trend={})",
            c["series"].as_str().unwrap(),
            c["trend"].as_str().unwrap()
        );
        let res = AutoReg::new(x, lags, trend).unwrap().fit().unwrap();

        check_vec(
            &format!("{label}.params"),
            &res.params,
            &vec1(&c["params"]),
            1e-8,
        );
        check_vec(&format!("{label}.bse"), &res.bse, &vec1(&c["bse"]), 1e-8);
        check_vec(
            &format!("{label}.tvalues"),
            &res.tvalues,
            &vec1(&c["tvalues"]),
            1e-8,
        );
        // p-values route through the normal survival function.
        check_vec(
            &format!("{label}.pvalues"),
            &res.pvalues,
            &vec1(&c["pvalues"]),
            1e-6,
        );
        check_vec(
            &format!("{label}.fittedvalues"),
            &res.fittedvalues,
            &vec1(&c["fittedvalues"]),
            1e-8,
        );
        check_vec(
            &format!("{label}.resid"),
            &res.resid,
            &vec1(&c["resid"]),
            1e-8,
        );

        let approx = |got: f64, key: &str, tol: f64| {
            let want = c[key].as_f64().unwrap();
            assert!(
                rel(got, want) <= tol,
                "{label}.{key}: rel-err {:.3e} (got {got}, want {want})",
                rel(got, want)
            );
        };
        approx(res.sigma2, "sigma2", 1e-8);
        approx(res.llf, "llf", 1e-8);
        approx(res.aic, "aic", 1e-8);
        approx(res.bic, "bic", 1e-8);
        approx(res.hqic, "hqic", 1e-8);
        assert_eq!(
            res.nobs,
            c["nobs"].as_u64().unwrap() as usize,
            "{label}: nobs"
        );
        assert_eq!(
            res.df_model,
            c["df_model"].as_u64().unwrap() as usize,
            "{label}: df_model"
        );
    }
}

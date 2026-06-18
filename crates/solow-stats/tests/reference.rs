//! Cross-validation of the statistical tests against golden reference values
//! frozen in `tests/fixtures/stats.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_stats::{
    acorr_ljungbox, durbin_watson, het_breuschpagan, het_white, jarque_bera, multipletests,
    omni_normtest, ttest_ind, ztest, Alternative, DescrStatsW, MultiTestMethod, UseVar,
};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/stats.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_stats.py)");
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

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / (1.0 + want.abs())
}

fn close(label: &str, got: f64, want: f64, tol: f64) {
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{label}: rel-err {e:.3e} (got {got}, want {want})"
    );
}

fn alt_of(s: &str) -> Alternative {
    match s {
        "two-sided" => Alternative::TwoSided,
        "larger" => Alternative::Larger,
        "smaller" => Alternative::Smaller,
        other => panic!("unknown alternative {other}"),
    }
}

#[test]
fn durbin_watson_matches_reference() {
    let fx = load();
    let resid = vec1(&fx["resid"]);
    let want = fx["durbin_watson"].as_f64().unwrap();
    close("durbin_watson", durbin_watson(&resid), want, 1e-12);
}

#[test]
fn jarque_bera_matches_reference() {
    let fx = load();
    let resid = vec1(&fx["resid"]);
    let jb = jarque_bera(&resid);
    let exp = &fx["jarque_bera"];
    close("jb.skew", jb.skew, exp["skew"].as_f64().unwrap(), 1e-12);
    close(
        "jb.kurtosis",
        jb.kurtosis,
        exp["kurtosis"].as_f64().unwrap(),
        1e-12,
    );
    close(
        "jb.statistic",
        jb.statistic,
        exp["statistic"].as_f64().unwrap(),
        1e-10,
    );
    close(
        "jb.pvalue",
        jb.pvalue,
        exp["pvalue"].as_f64().unwrap(),
        1e-9,
    );
}

#[test]
fn omni_normtest_matches_reference() {
    let fx = load();
    let resid = vec1(&fx["resid"]);
    let (stat, pval) = omni_normtest(&resid);
    let exp = &fx["omni_normtest"];
    close(
        "omni.statistic",
        stat,
        exp["statistic"].as_f64().unwrap(),
        1e-10,
    );
    close("omni.pvalue", pval, exp["pvalue"].as_f64().unwrap(), 1e-9);
}

#[test]
fn descrstatsw_matches_reference() {
    let fx = load();
    let d = &fx["descrstatsw"];
    let data = vec1(&d["data"]);
    let weights = vec1(&d["weights"]);
    for c in d["cases"].as_array().unwrap() {
        let ddof = c["ddof"].as_f64().unwrap();
        let dsw = DescrStatsW::new(data.clone(), Some(weights.clone()), ddof);
        close(
            "sum_weights",
            dsw.sum_weights(),
            c["sum_weights"].as_f64().unwrap(),
            1e-12,
        );
        close("nobs", dsw.nobs(), c["nobs"].as_f64().unwrap(), 1e-12);
        close("sum", dsw.sum(), c["sum"].as_f64().unwrap(), 1e-12);
        close("mean", dsw.mean(), c["mean"].as_f64().unwrap(), 1e-12);
        close(
            "sumsquares",
            dsw.sumsquares(),
            c["sumsquares"].as_f64().unwrap(),
            1e-12,
        );
        close("var", dsw.var(), c["var"].as_f64().unwrap(), 1e-12);
        close("std", dsw.std(), c["std"].as_f64().unwrap(), 1e-12);
        close(
            "std_mean",
            dsw.std_mean(),
            c["std_mean"].as_f64().unwrap(),
            1e-12,
        );

        let value = c["ttest_value"].as_f64().unwrap();
        let r = dsw.ttest_mean(value, Alternative::TwoSided);
        close(
            "ttest.statistic",
            r.statistic,
            c["ttest_statistic"].as_f64().unwrap(),
            1e-10,
        );
        close(
            "ttest.pvalue",
            r.pvalue,
            c["ttest_pvalue"].as_f64().unwrap(),
            1e-9,
        );
        close("ttest.df", r.df, c["ttest_df"].as_f64().unwrap(), 1e-12);
        let rl = dsw.ttest_mean(value, Alternative::Larger);
        close(
            "ttest.pvalue_larger",
            rl.pvalue,
            c["ttest_pvalue_larger"].as_f64().unwrap(),
            1e-9,
        );
        let rs = dsw.ttest_mean(value, Alternative::Smaller);
        close(
            "ttest.pvalue_smaller",
            rs.pvalue,
            c["ttest_pvalue_smaller"].as_f64().unwrap(),
            1e-9,
        );
    }
}

#[test]
fn ttest_ind_matches_reference() {
    let fx = load();
    let t = &fx["ttest_ind"];
    let x1 = vec1(&t["x1"]);
    let x2 = vec1(&t["x2"]);
    for c in t["cases"].as_array().unwrap() {
        let usevar = match c["usevar"].as_str().unwrap() {
            "pooled" => UseVar::Pooled,
            "unequal" => UseVar::Unequal,
            o => panic!("unknown usevar {o}"),
        };
        let alt = alt_of(c["alternative"].as_str().unwrap());
        let value = c["value"].as_f64().unwrap();
        let r = ttest_ind(&x1, &x2, alt, usevar, value);
        let label = format!("ttest_ind[{:?},{:?}]", usevar, alt);
        close(
            &format!("{label}.statistic"),
            r.statistic,
            c["statistic"].as_f64().unwrap(),
            1e-10,
        );
        close(
            &format!("{label}.pvalue"),
            r.pvalue,
            c["pvalue"].as_f64().unwrap(),
            1e-9,
        );
        close(
            &format!("{label}.df"),
            r.df,
            c["df"].as_f64().unwrap(),
            1e-10,
        );
    }
}

#[test]
fn ztest_matches_reference() {
    let fx = load();
    let z = &fx["ztest"];
    let x1 = vec1(&z["x1"]);
    for c in z["cases"].as_array().unwrap() {
        let ddof = c["ddof"].as_f64().unwrap();
        let alt = alt_of(c["alternative"].as_str().unwrap());
        let value = c["value"].as_f64().unwrap();
        let r = ztest(&x1, value, alt, ddof);
        let label = format!("ztest[ddof={ddof},{:?}]", alt);
        close(
            &format!("{label}.statistic"),
            r.statistic,
            c["statistic"].as_f64().unwrap(),
            1e-10,
        );
        close(
            &format!("{label}.pvalue"),
            r.pvalue,
            c["pvalue"].as_f64().unwrap(),
            1e-9,
        );
    }
}

#[test]
fn het_breuschpagan_matches_reference() {
    let fx = load();
    let h = &fx["het_breuschpagan"];
    let exog = mat(&h["exog"]);
    let resid = vec1(&h["resid"]);
    let (lm, lmpv, f, fpv) = het_breuschpagan(&resid, &exog).unwrap();
    close("bp.lm", lm, h["lm"].as_f64().unwrap(), 1e-9);
    close("bp.lm_pvalue", lmpv, h["lm_pvalue"].as_f64().unwrap(), 1e-9);
    close("bp.fvalue", f, h["fvalue"].as_f64().unwrap(), 1e-9);
    close("bp.f_pvalue", fpv, h["f_pvalue"].as_f64().unwrap(), 1e-9);
}

#[test]
fn het_white_matches_reference() {
    let fx = load();
    let h = &fx["het_white"];
    let exog = mat(&h["exog"]);
    let resid = vec1(&h["resid"]);
    let (lm, lmpv, f, fpv) = het_white(&resid, &exog).unwrap();
    close("white.lm", lm, h["lm"].as_f64().unwrap(), 1e-9);
    close(
        "white.lm_pvalue",
        lmpv,
        h["lm_pvalue"].as_f64().unwrap(),
        1e-9,
    );
    close("white.fvalue", f, h["fvalue"].as_f64().unwrap(), 1e-9);
    close("white.f_pvalue", fpv, h["f_pvalue"].as_f64().unwrap(), 1e-9);
}

#[test]
fn acorr_ljungbox_matches_reference() {
    let fx = load();
    let l = &fx["acorr_ljungbox"];
    let series = vec1(&l["series"]);
    let lags = l["lags"].as_u64().unwrap() as usize;
    let got = acorr_ljungbox(&series, lags);
    let rows = l["rows"].as_array().unwrap();
    assert_eq!(got.len(), rows.len());
    for (g, exp) in got.iter().zip(rows.iter()) {
        assert_eq!(g.lag, exp["lag"].as_u64().unwrap() as usize);
        close(
            &format!("ljungbox[{}].lb_stat", g.lag),
            g.lb_stat,
            exp["lb_stat"].as_f64().unwrap(),
            1e-10,
        );
        close(
            &format!("ljungbox[{}].lb_pvalue", g.lag),
            g.lb_pvalue,
            exp["lb_pvalue"].as_f64().unwrap(),
            1e-9,
        );
    }
}

#[test]
fn multipletests_matches_reference() {
    let fx = load();
    let m = &fx["multipletests"];
    let pvals = vec1(&m["pvals"]);
    let pvals: Vec<f64> = pvals.to_vec();
    let alpha = m["alpha"].as_f64().unwrap();
    for c in m["cases"].as_array().unwrap() {
        let method = match c["method"].as_str().unwrap() {
            "bonferroni" => MultiTestMethod::Bonferroni,
            "fdr_bh" => MultiTestMethod::FdrBh,
            "holm" => MultiTestMethod::Holm,
            o => panic!("unknown method {o}"),
        };
        let res = multipletests(&pvals, alpha, method);
        let exp_reject: Vec<bool> = c["reject"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_bool().unwrap())
            .collect();
        assert_eq!(res.reject, exp_reject, "{:?}.reject", method);
        let exp_pvc = vec1(&c["pvals_corrected"]);
        for (i, (&g, w)) in res.pvals_corrected.iter().zip(exp_pvc.iter()).enumerate() {
            close(&format!("{:?}.pvals_corrected[{i}]", method), g, *w, 1e-10);
        }
    }
}

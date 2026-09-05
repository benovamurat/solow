//! Cross-validation of the time-series-analysis extensions (set 4): the
//! Zivot-Andrews structural-break unit-root test, the range unit-root (RUR)
//! test, and the break-variance heteroskedasticity test. Golden values are
//! frozen in `tests/fixtures/tsa_ext4.json`.
#![allow(clippy::needless_range_loop)]

use ndarray::Array1;
use serde_json::Value;
use solow_tsa::{
    breakvar_heteroskedasticity_test, range_unit_root_test, zivot_andrews, AutoLag,
    BreakvarAlternative, SubsetLength, ZaRegression,
};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/tsa_ext4.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_tsa_ext4.py)");
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

fn za_reg(s: &str) -> ZaRegression {
    match s {
        "c" => ZaRegression::C,
        "t" => ZaRegression::T,
        "ct" => ZaRegression::Ct,
        other => panic!("unknown regression {other}"),
    }
}

fn za_autolag(v: &Value) -> Option<AutoLag> {
    if v.is_null() {
        return Some(AutoLag::None);
    }
    match v.as_str().unwrap() {
        "AIC" => Some(AutoLag::Aic),
        "BIC" => Some(AutoLag::Bic),
        "t-stat" => Some(AutoLag::TStat),
        other => panic!("unknown autolag {other}"),
    }
}

#[test]
fn zivot_andrews_matches_reference() {
    // The Zivot-Andrews statistic is the minimum OLS t-statistic over candidate
    // breakpoints; each t-statistic is a closed-form OLS quantity, so the final
    // statistic and critical values match to ~1e-10. We assert the statistic at
    // 1e-6, the break index exactly, and baselag exactly. The p-value is a
    // linear interpolation of the simulated reference table; it reproduces the
    // reference `numpy.interp` exactly, so we assert it at 1e-8.
    let fx = load();
    for c in fx["zivot_andrews"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x = vec1(&c["x"]);
        let trim = c["trim"].as_f64().unwrap();
        let maxlag = c["maxlag"].as_u64().map(|v| v as usize);
        let reg = za_reg(c["regression"].as_str().unwrap());
        let autolag = za_autolag(&c["autolag"]);

        let res = zivot_andrews(&x, trim, maxlag, reg, autolag).unwrap();

        check(
            &format!("za[{name}].stat"),
            res.stat,
            c["stat"].as_f64().unwrap(),
            1e-6,
        );
        check(
            &format!("za[{name}].pvalue"),
            res.pvalue,
            c["pvalue"].as_f64().unwrap(),
            1e-8,
        );
        check(
            &format!("za[{name}].crit_1"),
            res.crit_values[0],
            c["crit_1"].as_f64().unwrap(),
            1e-8,
        );
        check(
            &format!("za[{name}].crit_5"),
            res.crit_values[1],
            c["crit_5"].as_f64().unwrap(),
            1e-8,
        );
        check(
            &format!("za[{name}].crit_10"),
            res.crit_values[2],
            c["crit_10"].as_f64().unwrap(),
            1e-8,
        );
        assert_eq!(
            res.baselag,
            c["baselag"].as_u64().unwrap() as usize,
            "za[{name}].baselag"
        );
        assert_eq!(
            res.breakidx,
            c["breakidx"].as_u64().unwrap() as usize,
            "za[{name}].breakidx"
        );
    }
}

#[test]
fn range_unit_root_matches_reference() {
    // The RUR statistic is an exact integer count divided by sqrt(nobs), so it
    // is reproduced to machine precision (assert 1e-10). The p-value is a
    // table-bucket lookup (assert exact), and the critical values are an exact
    // linear interpolation of the table (assert 1e-8).
    let fx = load();
    for c in fx["range_unit_root"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x = vec1(&c["x"]);
        let res = range_unit_root_test(&x).unwrap();

        check(
            &format!("rur[{name}].stat"),
            res.stat,
            c["stat"].as_f64().unwrap(),
            1e-10,
        );
        // p-value is selected from a discrete table; require exact equality.
        let want_p = c["pvalue"].as_f64().unwrap();
        assert!(
            (res.pvalue - want_p).abs() < 1e-12,
            "rur[{name}].pvalue: got {} want {want_p}",
            res.pvalue
        );
        check(
            &format!("rur[{name}].crit_10"),
            res.crit_values[0],
            c["crit_10"].as_f64().unwrap(),
            1e-8,
        );
        check(
            &format!("rur[{name}].crit_5"),
            res.crit_values[1],
            c["crit_5"].as_f64().unwrap(),
            1e-8,
        );
        check(
            &format!("rur[{name}].crit_2_5"),
            res.crit_values[2],
            c["crit_2_5"].as_f64().unwrap(),
            1e-8,
        );
        check(
            &format!("rur[{name}].crit_1"),
            res.crit_values[3],
            c["crit_1"].as_f64().unwrap(),
            1e-8,
        );
    }
}

fn bv_alt(s: &str) -> BreakvarAlternative {
    match s {
        "increasing" => BreakvarAlternative::Increasing,
        "decreasing" => BreakvarAlternative::Decreasing,
        "two-sided" => BreakvarAlternative::TwoSided,
        other => panic!("unknown alternative {other}"),
    }
}

#[test]
fn breakvar_matches_reference() {
    // The statistic is a ratio of sums-of-squares (closed form, ~1e-12); the
    // p-value comes from an F or chi-squared distribution evaluated via the
    // reference's incomplete-beta / -gamma routines. We assert the statistic at
    // 1e-10 and the p-value at 1e-6 (distribution-CDF accuracy).
    let fx = load();
    for c in fx["breakvar"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let resid = vec1(&c["resid"]);
        let subset = if c["subset_is_int"].as_bool().unwrap() {
            SubsetLength::Fixed(c["subset_length"].as_u64().unwrap() as usize)
        } else {
            SubsetLength::Fraction(c["subset_length"].as_f64().unwrap())
        };
        let alt = bv_alt(c["alternative"].as_str().unwrap());
        let use_f = c["use_f"].as_bool().unwrap();

        let (stat, pvalue) = breakvar_heteroskedasticity_test(&resid, subset, alt, use_f).unwrap();

        check(
            &format!("bv[{name}].stat"),
            stat,
            c["stat"].as_f64().unwrap(),
            1e-10,
        );
        check(
            &format!("bv[{name}].pvalue"),
            pvalue,
            c["pvalue"].as_f64().unwrap(),
            1e-6,
        );
    }
}

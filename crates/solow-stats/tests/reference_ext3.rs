//! Cross-validation of the OLS-residual diagnostics extension batch
//! (Breusch-Godfrey serial-correlation, Ramsey RESET functional-form, Engle
//! ARCH, the generic autocorrelation LM test, and nested-model LR / F
//! comparisons) against golden reference values frozen in
//! `tests/fixtures/stats_ext3.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_regression::LinearModel;
use solow_stats::{
    acorr_breusch_godfrey, acorr_lm, compare_f_test, compare_lr_test, het_arch, linear_reset,
    ResetAug,
};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/stats_ext3.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_stats_ext3.py)");
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

fn assert_rel(label: &str, got: f64, want: f64, tol: f64) {
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{label}: rel-err {e:.3e} (got {got}, want {want})"
    );
}

/// `nlags` is encoded as JSON `null` for the reference default rule.
fn nlags_of(v: &Value) -> Option<usize> {
    v.as_u64().map(|x| x as usize)
}

// ---------------------------------------------------------------------------
// OLS-residual diagnostics. The LM statistic is `nobs · R²` (closed form from
// the auxiliary OLS, asserted 1e-8); the F statistic comes from the auxiliary
// regression's overall F (BG: from a Wald restriction). The p-values pass
// through the chi-squared / F survival functions and are asserted 1e-6, well
// inside the requirement. The RESET Wald statistic is a closed-form quadratic
// form (asserted 1e-8) with a chi-squared / F p-value (1e-6).
// ---------------------------------------------------------------------------
#[test]
fn ols_diagnostics_match_reference() {
    let fx = load();
    for case in fx["diagnostics"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let endog = vec1(&case["endog"]);
        let exog = mat(&case["exog"]);
        let res = LinearModel::ols(endog.clone(), exog.clone())
            .unwrap()
            .fit()
            .unwrap();
        let resid = res.resid.clone();

        // Sanity: our refit reproduces the reference residuals tightly.
        let ref_resid = vec1(&case["resid"]);
        for i in 0..resid.len() {
            assert_rel(&format!("{name}.resid[{i}]"), resid[i], ref_resid[i], 1e-8);
        }

        // Breusch-Godfrey.
        for (bi, b) in case["bg"].as_array().unwrap().iter().enumerate() {
            let nl = nlags_of(&b["nlags"]);
            let (lm, lmp, fv, fp) = acorr_breusch_godfrey(&resid, &exog, nl).unwrap();
            let lab = format!("{name}.bg[{bi}]");
            assert_rel(&format!("{lab}.lm"), lm, b["lm"].as_f64().unwrap(), 1e-8);
            assert_rel(
                &format!("{lab}.lm_pvalue"),
                lmp,
                b["lm_pvalue"].as_f64().unwrap(),
                1e-6,
            );
            assert_rel(
                &format!("{lab}.fvalue"),
                fv,
                b["fvalue"].as_f64().unwrap(),
                1e-8,
            );
            assert_rel(
                &format!("{lab}.f_pvalue"),
                fp,
                b["f_pvalue"].as_f64().unwrap(),
                1e-6,
            );
        }

        // Generic autocorrelation LM.
        for (li, b) in case["acorr_lm"].as_array().unwrap().iter().enumerate() {
            let nl = nlags_of(&b["nlags"]);
            let (lm, lmp, fv, fp) = acorr_lm(&resid, nl, 0).unwrap();
            let lab = format!("{name}.acorr_lm[{li}]");
            assert_rel(&format!("{lab}.lm"), lm, b["lm"].as_f64().unwrap(), 1e-8);
            assert_rel(
                &format!("{lab}.lm_pvalue"),
                lmp,
                b["lm_pvalue"].as_f64().unwrap(),
                1e-6,
            );
            assert_rel(
                &format!("{lab}.fvalue"),
                fv,
                b["fvalue"].as_f64().unwrap(),
                1e-8,
            );
            assert_rel(
                &format!("{lab}.f_pvalue"),
                fp,
                b["f_pvalue"].as_f64().unwrap(),
                1e-6,
            );
        }

        // Engle ARCH-LM.
        for (ai, b) in case["het_arch"].as_array().unwrap().iter().enumerate() {
            let nl = nlags_of(&b["nlags"]);
            let (lm, lmp, fv, fp) = het_arch(&resid, nl, 0).unwrap();
            let lab = format!("{name}.het_arch[{ai}]");
            assert_rel(&format!("{lab}.lm"), lm, b["lm"].as_f64().unwrap(), 1e-8);
            assert_rel(
                &format!("{lab}.lm_pvalue"),
                lmp,
                b["lm_pvalue"].as_f64().unwrap(),
                1e-6,
            );
            assert_rel(
                &format!("{lab}.fvalue"),
                fv,
                b["fvalue"].as_f64().unwrap(),
                1e-8,
            );
            assert_rel(
                &format!("{lab}.f_pvalue"),
                fp,
                b["f_pvalue"].as_f64().unwrap(),
                1e-6,
            );
        }

        // Ramsey RESET.
        for (ri, b) in case["reset"].as_array().unwrap().iter().enumerate() {
            let power = b["power"].as_u64().unwrap() as usize;
            let ttype = match b["test_type"].as_str().unwrap() {
                "fitted" => ResetAug::Fitted,
                "exog" => ResetAug::Exog,
                other => panic!("unknown reset test_type {other}"),
            };
            let use_f = b["use_f"].as_bool().unwrap();
            let (stat, pv) = linear_reset(&endog, &exog, power, ttype, use_f).unwrap();
            let lab = format!("{name}.reset[{ri}]");
            assert_rel(
                &format!("{lab}.statistic"),
                stat,
                b["statistic"].as_f64().unwrap(),
                1e-8,
            );
            assert_rel(
                &format!("{lab}.pvalue"),
                pv,
                b["pvalue"].as_f64().unwrap(),
                1e-6,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Nested-model comparisons. Both the LR statistic (−2 Δllf, closed form from the
// two OLS log-likelihoods) and the F statistic (ratio of SSR differences) are
// closed form and asserted 1e-8; their chi-squared / F p-values are asserted
// 1e-6. Degrees of freedom are exact integers.
// ---------------------------------------------------------------------------
#[test]
fn nested_comparisons_match_reference() {
    let fx = load();
    for case in fx["compare"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let endog = vec1(&case["endog"]);
        let exog_full = mat(&case["exog_full"]);
        let exog_restr = mat(&case["exog_restricted"]);
        let full = LinearModel::ols(endog.clone(), exog_full)
            .unwrap()
            .fit()
            .unwrap();
        let restr = LinearModel::ols(endog, exog_restr).unwrap().fit().unwrap();

        let (lr, lrp, lrdf) = compare_lr_test(&full, &restr);
        assert_rel(
            &format!("{name}.lr_stat"),
            lr,
            case["lr_stat"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.lr_pvalue"),
            lrp,
            case["lr_pvalue"].as_f64().unwrap(),
            1e-6,
        );
        assert_rel(
            &format!("{name}.lr_df"),
            lrdf,
            case["lr_df"].as_f64().unwrap(),
            1e-12,
        );

        let (f, fp, fdf) = compare_f_test(&full, &restr);
        assert_rel(
            &format!("{name}.f_value"),
            f,
            case["f_value"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.f_pvalue"),
            fp,
            case["f_pvalue"].as_f64().unwrap(),
            1e-6,
        );
        assert_rel(
            &format!("{name}.f_df"),
            fdf,
            case["f_df"].as_f64().unwrap(),
            1e-12,
        );
    }
}

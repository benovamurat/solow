//! Cross-validation of the statistical extensions (ANOVA, one-way ANOVA,
//! proportions, Tukey HSD, power, contingency tables) against golden reference
//! values frozen in `tests/fixtures/stats_ext.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_stats::{
    anova_lm, f_oneway, pairwise_tukeyhsd, proportion_confint, proportions_ztest, AnovaType,
    ConfintMethod, NormalIndPower, OnewayUseVar, TTestPower, Table, Term,
};
use solow_stats::{anova_oneway, Alternative};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/stats_ext.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_stats_ext.py)");
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

fn alt_of(s: &str) -> Alternative {
    match s {
        "two-sided" => Alternative::TwoSided,
        "larger" => Alternative::Larger,
        "smaller" => Alternative::Smaller,
        other => panic!("unknown alternative {other}"),
    }
}

// ---------------------------------------------------------------------------
// ANOVA (types I/II/III). Closed-form algebra: tolerance 1e-8.
// ---------------------------------------------------------------------------
#[test]
fn anova_matches_reference() {
    let fx = load();
    let a = &fx["anova"];
    let endog = vec1(&a["endog"]);
    let exog = mat(&a["exog"]);

    // Build the term list from term_names + term_cols.
    let term_names: Vec<String> = a["term_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect();
    let term_cols = &a["term_cols"];
    let make_terms = || -> Vec<Term> {
        term_names
            .iter()
            .map(|name| {
                let pair = term_cols[name].as_array().unwrap();
                let start = pair[0].as_u64().unwrap() as usize;
                let stop = pair[1].as_u64().unwrap() as usize;
                Term::new(name.clone(), start, stop)
            })
            .collect()
    };

    for (key, typ) in [
        ("1", AnovaType::I),
        ("2", AnovaType::II),
        ("3", AnovaType::III),
    ] {
        let terms = make_terms();
        let tab = anova_lm(&endog, &exog, &terms, typ).expect("anova");
        let exp = &fx["anova"]["tables"][key];
        let index: Vec<String> = exp["index"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        let rows = &exp["rows"];

        assert_eq!(tab.rows.len(), index.len(), "type {key}: row count");
        for name in &index {
            let got = tab
                .row(name)
                .unwrap_or_else(|| panic!("type {key}: missing row {name}"));
            let want = &rows[name];
            let lab = format!("anova[{key}].{name}");
            assert_rel(
                &format!("{lab}.df"),
                got.df,
                want["df"].as_f64().unwrap(),
                1e-12,
            );
            assert_rel(
                &format!("{lab}.sum_sq"),
                got.sum_sq,
                want["sum_sq"].as_f64().unwrap(),
                1e-8,
            );
            assert_rel(
                &format!("{lab}.mean_sq"),
                got.mean_sq,
                want["mean_sq"].as_f64().unwrap(),
                1e-8,
            );
            if let Some(f) = want["F"].as_f64() {
                assert_rel(&format!("{lab}.F"), got.f.unwrap(), f, 1e-8);
            } else {
                assert!(got.f.is_none(), "{lab}: expected no F");
            }
            if let Some(pr) = want["PR"].as_f64() {
                assert_rel(&format!("{lab}.PR"), got.pr.unwrap(), pr, 1e-8);
            } else {
                assert!(got.pr.is_none(), "{lab}: expected no PR");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// One-way ANOVA: classic / Welch / Brown-Forsythe. Tolerance 1e-8.
// ---------------------------------------------------------------------------
#[test]
fn oneway_matches_reference() {
    let fx = load();
    let o = &fx["oneway"];
    let groups: Vec<Vec<f64>> = o["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| {
            g.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();

    for case in o["cases"].as_array().unwrap() {
        let use_var = match case["use_var"].as_str().unwrap() {
            "equal" => OnewayUseVar::Equal,
            "unequal" => OnewayUseVar::Unequal,
            "bf" => OnewayUseVar::BrownForsythe,
            other => panic!("unknown use_var {other}"),
        };
        let res = anova_oneway(&groups, use_var, true);
        let lab = format!("oneway[{}]", case["use_var"].as_str().unwrap());
        assert_rel(
            &format!("{lab}.statistic"),
            res.statistic,
            case["statistic"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.pvalue"),
            res.pvalue,
            case["pvalue"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.df_num"),
            res.df_num,
            case["df_num"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.df_denom"),
            res.df_denom,
            case["df_denom"].as_f64().unwrap(),
            1e-8,
        );
    }

    // scipy f_oneway equivalent.
    let (f, p) = f_oneway(&groups);
    assert_rel(
        "f_oneway.statistic",
        f,
        o["f_oneway"]["statistic"].as_f64().unwrap(),
        1e-8,
    );
    assert_rel(
        "f_oneway.pvalue",
        p,
        o["f_oneway"]["pvalue"].as_f64().unwrap(),
        1e-8,
    );
}

// ---------------------------------------------------------------------------
// Proportions z-test and confidence intervals. Tolerance 1e-8 (closed form).
// ---------------------------------------------------------------------------
#[test]
fn proportion_matches_reference() {
    let fx = load();
    let p = &fx["proportion"];

    for case in p["ztest"].as_array().unwrap() {
        let count: Vec<f64> = case["count"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let nobs: Vec<f64> = case["nobs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let value = case["value"].as_f64().unwrap();
        let alt = alt_of(case["alternative"].as_str().unwrap());
        let (z, pv) = proportions_ztest(&count, &nobs, value, alt);
        let lab = format!(
            "ztest[{}/{}]",
            case["kind"].as_str().unwrap(),
            case["alternative"].as_str().unwrap()
        );
        assert_rel(
            &format!("{lab}.zstat"),
            z,
            case["zstat"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.pvalue"),
            pv,
            case["pvalue"].as_f64().unwrap(),
            1e-8,
        );
    }

    for case in p["confint"].as_array().unwrap() {
        let count = case["count"].as_f64().unwrap();
        let nobs = case["nobs"].as_f64().unwrap();
        let alpha = case["alpha"].as_f64().unwrap();
        let method = match case["method"].as_str().unwrap() {
            "normal" => ConfintMethod::Normal,
            "agresti_coull" => ConfintMethod::AgrestiCoull,
            "wilson" => ConfintMethod::Wilson,
            "beta" => ConfintMethod::Beta,
            "jeffreys" => ConfintMethod::Jeffreys,
            other => panic!("unknown method {other}"),
        };
        let (lo, hi) = proportion_confint(count, nobs, alpha, method);
        let lab = format!(
            "confint[{} {}/{}]",
            case["method"].as_str().unwrap(),
            count,
            nobs
        );
        // All methods (including beta/jeffreys through the Beta inverse) agree to
        // near machine precision; 1e-8 leaves headroom.
        assert_rel(
            &format!("{lab}.lower"),
            lo,
            case["lower"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.upper"),
            hi,
            case["upper"].as_f64().unwrap(),
            1e-8,
        );
    }
}

// ---------------------------------------------------------------------------
// Tukey HSD. Studentized-range quantities; the spec asks only for >= 1e-4 but
// the Gauss-Legendre quadrature reproduces the reference to ~1e-13, so we
// assert 1e-8 on the studentized-range pieces (q_crit, confint, p-values).
// ---------------------------------------------------------------------------
#[test]
fn tukey_matches_reference() {
    let fx = load();
    let t = &fx["tukey"];
    let data: Vec<f64> = t["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let groups: Vec<usize> = t["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let alpha = t["alpha"].as_f64().unwrap();
    let res = pairwise_tukeyhsd(&data, &groups, alpha);

    let meandiffs = vec1(&t["meandiffs"]);
    let std_pairs = vec1(&t["std_pairs"]);
    let confint = mat(&t["confint"]);
    let pvalues = vec1(&t["pvalues"]);
    let reject: Vec<bool> = t["reject"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_bool().unwrap())
        .collect();

    assert_eq!(res.pairs.len(), meandiffs.len());
    // Closed-form pieces: meandiffs and std_pairs.
    for i in 0..res.pairs.len() {
        assert_rel(
            &format!("tukey.meandiff[{i}]"),
            res.meandiffs[i],
            meandiffs[i],
            1e-10,
        );
        assert_rel(
            &format!("tukey.std_pairs[{i}]"),
            res.std_pairs[i],
            std_pairs[i],
            1e-10,
        );
    }
    assert_rel(
        "tukey.df_total",
        res.df_total,
        t["df_total"].as_f64().unwrap(),
        1e-12,
    );
    assert_rel(
        "tukey.variance",
        res.variance,
        t["variance"].as_f64().unwrap(),
        1e-10,
    );
    // Studentized-range pieces: q_crit, confint, p-values (achieved ~1e-13).
    assert_rel(
        "tukey.q_crit",
        res.q_crit,
        t["q_crit"].as_f64().unwrap(),
        1e-8,
    );
    for i in 0..res.pairs.len() {
        assert_rel(
            &format!("tukey.confint.lo[{i}]"),
            res.confint[i].0,
            confint[[i, 0]],
            1e-8,
        );
        assert_rel(
            &format!("tukey.confint.hi[{i}]"),
            res.confint[i].1,
            confint[[i, 1]],
            1e-8,
        );
        assert_rel(
            &format!("tukey.pvalue[{i}]"),
            res.pvalues[i],
            pvalues[i],
            1e-8,
        );
        assert_eq!(res.reject[i], reject[i], "tukey.reject[{i}]");
    }
}

// ---------------------------------------------------------------------------
// Power (one-sample t and two-sample normal). Power values via the noncentral
// t / normal agree to ~1e-15 (asserted at 1e-8). Solved nobs: the normal case
// matches to ~1e-9, but the t case differs by ~6.5e-8 relative because the
// reference's own brentq root-finder stops at power=0.80000003 rather than
// exactly 0.8 -- so the t solve_nobs is asserted at 1e-6 (our root is the
// tighter one; the residual gap is the reference's tolerance, not our error).
// ---------------------------------------------------------------------------
#[test]
fn power_matches_reference() {
    let fx = load();
    let pw = &fx["power"];

    let tt = TTestPower;
    for case in pw["ttest"].as_array().unwrap() {
        let es = case["effect_size"].as_f64().unwrap();
        let nobs = case["nobs"].as_f64().unwrap();
        let alpha = case["alpha"].as_f64().unwrap();
        let alt = alt_of(case["alternative"].as_str().unwrap());
        let p = tt.power(es, nobs, alpha, None, alt);
        let lab = format!(
            "ttest_power[es={es},alt={}]",
            case["alternative"].as_str().unwrap()
        );
        assert_rel(
            &format!("{lab}.power"),
            p,
            case["power"].as_f64().unwrap(),
            1e-8,
        );
        let n = tt.solve_power(es, alpha, 0.8, alt);
        // See module note: residual ~6.5e-8 is the reference's own root tolerance.
        assert_rel(
            &format!("{lab}.solve_nobs"),
            n,
            case["nobs_for_power_0_8"].as_f64().unwrap(),
            1e-6,
        );
    }

    let nip = NormalIndPower;
    for case in pw["normal_ind"].as_array().unwrap() {
        let es = case["effect_size"].as_f64().unwrap();
        let nobs1 = case["nobs1"].as_f64().unwrap();
        let alpha = case["alpha"].as_f64().unwrap();
        let ratio = case["ratio"].as_f64().unwrap();
        let alt = alt_of(case["alternative"].as_str().unwrap());
        let p = nip.power(es, nobs1, alpha, ratio, alt);
        let lab = format!(
            "normal_power[es={es},alt={}]",
            case["alternative"].as_str().unwrap()
        );
        assert_rel(
            &format!("{lab}.power"),
            p,
            case["power"].as_f64().unwrap(),
            1e-8,
        );
        let n = nip.solve_power(es, alpha, 0.8, ratio, alt);
        assert_rel(
            &format!("{lab}.solve_nobs"),
            n,
            case["nobs1_for_power_0_8"].as_f64().unwrap(),
            1e-6,
        );
    }
}

// ---------------------------------------------------------------------------
// Contingency chi-squared. Closed-form: tolerance 1e-8.
// ---------------------------------------------------------------------------
#[test]
fn contingency_matches_reference() {
    let fx = load();
    for (ci, case) in fx["contingency"]["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let table = mat(&case["table"]);
        let t = Table::new(table);
        let res = t.test_nominal_association();
        let lab = format!("contingency[{ci}]");
        assert_rel(
            &format!("{lab}.statistic"),
            res.statistic,
            case["statistic"].as_f64().unwrap(),
            1e-8,
        );
        assert_eq!(res.df, case["df"].as_u64().unwrap() as usize, "{lab}.df");
        assert_rel(
            &format!("{lab}.pvalue"),
            res.pvalue,
            case["pvalue"].as_f64().unwrap(),
            1e-8,
        );
        let expected = mat(&case["expected"]);
        for i in 0..expected.nrows() {
            for j in 0..expected.ncols() {
                assert_rel(
                    &format!("{lab}.expected[{i}][{j}]"),
                    res.expected[[i, j]],
                    expected[[i, j]],
                    1e-8,
                );
            }
        }
    }
}

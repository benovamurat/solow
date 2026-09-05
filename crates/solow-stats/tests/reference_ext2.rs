//! Cross-validation of the second statistical-extension batch (inter-rater
//! agreement, two-sample Poisson rate tests, descriptive statistics and linear
//! mediation point estimates) against golden reference values frozen in
//! `tests/fixtures/stats_ext2.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_stats::Alternative;
use solow_stats::{
    aggregate_raters, cohens_kappa, describe, fleiss_kappa, test_poisson_2indep, Compare,
    FleissMethod, Mediation, PoissonMethod,
};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/stats_ext2.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_stats_ext2.py)");
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
// Inter-rater agreement. Cohen's kappa is closed-form (1e-8); its zero-test
// p-values pass through the normal distribution (asserted 1e-8, well inside the
// 1e-6 requirement). Fleiss'/Randolph's kappa is closed-form (1e-8).
// ---------------------------------------------------------------------------
#[test]
fn inter_rater_matches_reference() {
    let fx = load();
    let ir = &fx["inter_rater"];

    for (ci, case) in ir["cohens"].as_array().unwrap().iter().enumerate() {
        let table = mat(&case["table"]);
        let alpha = case["alpha"].as_f64().unwrap();
        let r = cohens_kappa(&table, alpha);
        let lab = format!("cohens[{ci}]");
        assert_rel(
            &format!("{lab}.kappa"),
            r.kappa,
            case["kappa"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.kappa_max"),
            r.kappa_max,
            case["kappa_max"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.var_kappa"),
            r.var_kappa,
            case["var_kappa"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.var_kappa0"),
            r.var_kappa0,
            case["var_kappa0"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.std_kappa"),
            r.std_kappa,
            case["std_kappa"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.std_kappa0"),
            r.std_kappa0,
            case["std_kappa0"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.z_value"),
            r.z_value,
            case["z_value"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.pvalue_one_sided"),
            r.pvalue_one_sided,
            case["pvalue_one_sided"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.pvalue_two_sided"),
            r.pvalue_two_sided,
            case["pvalue_two_sided"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.kappa_low"),
            r.kappa_low,
            case["kappa_low"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.kappa_upp"),
            r.kappa_upp,
            case["kappa_upp"].as_f64().unwrap(),
            1e-8,
        );
    }

    // aggregate_raters: integer-count table and sorted category labels.
    for (ci, case) in ir["aggregate"].as_array().unwrap().iter().enumerate() {
        let data = mat(&case["data"]);
        let (counts, cats) = aggregate_raters(&data);
        let exp_counts = mat(&case["counts"]);
        assert_eq!(counts.dim(), exp_counts.dim(), "aggregate[{ci}].shape");
        for i in 0..counts.nrows() {
            for j in 0..counts.ncols() {
                assert_eq!(
                    counts[[i, j]],
                    exp_counts[[i, j]],
                    "aggregate[{ci}].counts[{i}][{j}]"
                );
            }
        }
        let exp_cats = vec1(&case["categories"]);
        assert_eq!(cats.len(), exp_cats.len(), "aggregate[{ci}].categories.len");
        for (k, &c) in cats.iter().enumerate() {
            assert_eq!(c, exp_cats[k], "aggregate[{ci}].categories[{k}]");
        }
    }

    for (ci, case) in ir["fleiss"].as_array().unwrap().iter().enumerate() {
        let counts = mat(&case["counts"]);
        let method = match case["method"].as_str().unwrap() {
            "fleiss" => FleissMethod::Fleiss,
            "randolph" => FleissMethod::Randolph,
            other => panic!("unknown fleiss method {other}"),
        };
        let k = fleiss_kappa(&counts, method);
        assert_rel(
            &format!("fleiss[{ci}]"),
            k,
            case["kappa"].as_f64().unwrap(),
            1e-8,
        );
    }
}

// ---------------------------------------------------------------------------
// Two-sample Poisson rate test. The normal-based statistics are exact algebra,
// and their p-values pass only through the normal CDF/SF (asserted 1e-10). The
// exact-cond / cond-midp p-values pass through the binomial CDF/PMF (via the
// regularised incomplete beta) and the scipy "minlike" two-sided algorithm,
// reproduced to ~1e-12 (asserted 1e-8, inside the 1e-6 requirement).
// ---------------------------------------------------------------------------
#[test]
fn poisson_2indep_matches_reference() {
    let fx = load();
    for (ci, case) in fx["poisson"]["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let c1 = case["count1"].as_f64().unwrap();
        let e1 = case["exposure1"].as_f64().unwrap();
        let c2 = case["count2"].as_f64().unwrap();
        let e2 = case["exposure2"].as_f64().unwrap();
        let compare = match case["compare"].as_str().unwrap() {
            "ratio" => Compare::Ratio,
            "diff" => Compare::Diff,
            other => panic!("unknown compare {other}"),
        };
        let method = match case["method"].as_str().unwrap() {
            "score" => PoissonMethod::Score,
            "wald" => PoissonMethod::Wald,
            "waldccv" => PoissonMethod::WaldCcv,
            "score-log" => PoissonMethod::ScoreLog,
            "wald-log" => PoissonMethod::WaldLog,
            "sqrt" => PoissonMethod::Sqrt,
            "exact-cond" => PoissonMethod::ExactCond,
            "cond-midp" => PoissonMethod::CondMidp,
            other => panic!("unknown method {other}"),
        };
        let alt = alt_of(case["alternative"].as_str().unwrap());
        let r = test_poisson_2indep(c1, e1, c2, e2, None, method, compare, alt);
        let lab = format!(
            "poisson[{ci}/{}/{}/{}]",
            case["compare"].as_str().unwrap(),
            case["method"].as_str().unwrap(),
            case["alternative"].as_str().unwrap()
        );

        // statistic may be null for the binomial-based tests.
        if let Some(stat) = case["statistic"].as_f64() {
            assert_rel(&format!("{lab}.statistic"), r.statistic, stat, 1e-10);
        } else {
            assert!(r.statistic.is_nan(), "{lab}: expected NaN statistic");
        }
        assert_rel(
            &format!("{lab}.pvalue"),
            r.pvalue,
            case["pvalue"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.rate1"),
            r.rate1,
            case["rate1"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{lab}.rate2"),
            r.rate2,
            case["rate2"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{lab}.ratio"),
            r.ratio,
            case["ratio"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{lab}.diff"),
            r.diff,
            case["diff"].as_f64().unwrap(),
            1e-10,
        );
    }
}

// ---------------------------------------------------------------------------
// Descriptive statistics. All quantities are closed-form (1e-8); the JB p-value
// passes through the chi-squared survival function (also asserted 1e-8).
// ---------------------------------------------------------------------------
#[test]
fn describe_matches_reference() {
    let fx = load();
    for case in fx["describe"]["cases"].as_array().unwrap() {
        let data = vec1(&case["data"]);
        let alpha = case["alpha"].as_f64().unwrap();
        let use_t = case["use_t"].as_bool().unwrap();
        let d = describe(&data, alpha, use_t);
        let lab = format!("describe[{}]", case["name"].as_str().unwrap());

        assert_rel(
            &format!("{lab}.nobs"),
            d.nobs,
            case["nobs"].as_f64().unwrap(),
            1e-12,
        );
        assert_rel(
            &format!("{lab}.missing"),
            d.missing,
            case["missing"].as_f64().unwrap(),
            1e-12,
        );
        assert_rel(
            &format!("{lab}.mean"),
            d.mean,
            case["mean"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.std_err"),
            d.std_err,
            case["std_err"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.upper_ci"),
            d.upper_ci,
            case["upper_ci"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.lower_ci"),
            d.lower_ci,
            case["lower_ci"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.std"),
            d.std,
            case["std"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.iqr"),
            d.iqr,
            case["iqr"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.iqr_normal"),
            d.iqr_normal,
            case["iqr_normal"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.mad"),
            d.mad,
            case["mad"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.mad_normal"),
            d.mad_normal,
            case["mad_normal"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.coef_var"),
            d.coef_var,
            case["coef_var"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.range"),
            d.range,
            case["range"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.max"),
            d.max,
            case["max"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.min"),
            d.min,
            case["min"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.skew"),
            d.skew,
            case["skew"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.kurtosis"),
            d.kurtosis,
            case["kurtosis"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.jarque_bera"),
            d.jarque_bera,
            case["jarque_bera"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.jarque_bera_pval"),
            d.jarque_bera_pval,
            case["jarque_bera_pval"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.mode"),
            d.mode,
            case["mode"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.mode_freq"),
            d.mode_freq,
            case["mode_freq"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{lab}.median"),
            d.median,
            case["median"].as_f64().unwrap(),
            1e-8,
        );

        let exp_perc = vec1(&case["percentiles"]);
        assert_eq!(d.percentiles.len(), exp_perc.len(), "{lab}.percentiles.len");
        for (k, &p) in d.percentiles.iter().enumerate() {
            assert_rel(&format!("{lab}.percentiles[{k}]"), p, exp_perc[k], 1e-8);
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic linear mediation point estimates. Closed-form (Baron-Kenny);
// asserted at 1e-8. (Simulation-based CIs and p-values are out of scope.)
// ---------------------------------------------------------------------------
#[test]
fn mediation_matches_reference() {
    let fx = load();
    let m = &fx["mediation"];
    let med = Mediation {
        outcome_endog: vec1(&m["outcome_endog"]),
        outcome_exog: mat(&m["outcome_exog"]),
        mediator_endog: vec1(&m["mediator_endog"]),
        mediator_exog: mat(&m["mediator_exog"]),
        exp_pos_outcome: m["exp_pos_outcome"].as_u64().unwrap() as usize,
        exp_pos_mediator: m["exp_pos_mediator"].as_u64().unwrap() as usize,
        med_pos_outcome: m["med_pos_outcome"].as_u64().unwrap() as usize,
    };
    let r = med.fit();
    assert_rel(
        "mediation.acme_ctrl",
        r.acme_ctrl,
        m["acme_ctrl"].as_f64().unwrap(),
        1e-8,
    );
    assert_rel(
        "mediation.acme_tx",
        r.acme_tx,
        m["acme_tx"].as_f64().unwrap(),
        1e-8,
    );
    assert_rel(
        "mediation.ade_ctrl",
        r.ade_ctrl,
        m["ade_ctrl"].as_f64().unwrap(),
        1e-8,
    );
    assert_rel(
        "mediation.ade_tx",
        r.ade_tx,
        m["ade_tx"].as_f64().unwrap(),
        1e-8,
    );
    assert_rel(
        "mediation.total_effect",
        r.total_effect,
        m["total_effect"].as_f64().unwrap(),
        1e-8,
    );
    assert_rel(
        "mediation.acme_avg",
        r.acme_avg,
        m["acme_avg"].as_f64().unwrap(),
        1e-8,
    );
    assert_rel(
        "mediation.ade_avg",
        r.ade_avg,
        m["ade_avg"].as_f64().unwrap(),
        1e-8,
    );
    assert_rel(
        "mediation.prop_med_ctrl",
        r.prop_med_ctrl,
        m["prop_med_ctrl"].as_f64().unwrap(),
        1e-8,
    );
    assert_rel(
        "mediation.prop_med_tx",
        r.prop_med_tx,
        m["prop_med_tx"].as_f64().unwrap(),
        1e-8,
    );
    assert_rel(
        "mediation.prop_med_avg",
        r.prop_med_avg,
        m["prop_med_avg"].as_f64().unwrap(),
        1e-8,
    );
}

//! Cross-validation of the fifth stats-extension batch against golden reference
//! values frozen in `tests/fixtures/stats_ext5.json`.
//!
//! Covered:
//!   * the Blinder-Oaxaca two-fold (every weighting scheme) and three-fold
//!     decompositions (`stats.oaxaca.OaxacaBlinder`);
//!   * the distance covariance / correlation statistics and the asymptotic dCov
//!     test (`stats.dist_dependence_measures`);
//!   * the numeric descriptive-statistics summary
//!     (`stats.descriptivestats.describe`).

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_stats::{
    as_column, describe, distance_correlation, distance_covariance, distance_covariance_test,
    distance_statistics, distance_variance, OaxacaBlinder, TwoFoldType,
};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/stats_ext5.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_stats_ext5.py)");
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

fn assert_vec(label: &str, got: &Array1<f64>, want: &Array1<f64>, tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for i in 0..got.len() {
        assert_rel(&format!("{label}[{i}]"), got[i], want[i], tol);
    }
}

// Read either a 1-D vector (when dim == 1) or a 2-D matrix into an n x dim array.
fn read_design(v: &Value, dim: usize) -> Array2<f64> {
    if dim == 1 {
        as_column(&vec1(v))
    } else {
        mat(v)
    }
}

// ---------------------------------------------------------------------------
// Blinder-Oaxaca decomposition.
//
// Every quantity is a closed-form linear combination of OLS coefficients and
// covariate means, so the effects reproduce the reference to machine precision
// (asserted 1e-8). We first confirm the intermediate quantities (gap, group
// sizes, covariate means, the two group coefficient vectors) so a mismatch is
// localised, then check the two-fold (all five weighting schemes) and three-fold
// effects.
// ---------------------------------------------------------------------------
#[test]
fn oaxaca_matches_reference() {
    let fx = load();
    for case in fx["oaxaca"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let endog = vec1(&case["endog"]);
        let exog = mat(&case["exog"]);
        let bifurcate = case["bifurcate"].as_u64().unwrap() as usize;

        let m = OaxacaBlinder::new(endog, exog, bifurcate, true).unwrap();

        // Intermediate quantities.
        assert_rel(
            &format!("{name}.gap"),
            m.gap(),
            case["gap"].as_f64().unwrap(),
            1e-8,
        );
        assert_vec(
            &format!("{name}.exog_f_mean"),
            m.exog_f_mean(),
            &vec1(&case["exog_f_mean"]),
            1e-8,
        );
        assert_vec(
            &format!("{name}.exog_s_mean"),
            m.exog_s_mean(),
            &vec1(&case["exog_s_mean"]),
            1e-8,
        );
        assert_vec(
            &format!("{name}.f_params"),
            m.f_params(),
            &vec1(&case["f_params"]),
            1e-8,
        );
        assert_vec(
            &format!("{name}.s_params"),
            m.s_params(),
            &vec1(&case["s_params"]),
            1e-8,
        );

        // Two-fold: every weighting scheme.
        let two = &case["two_fold"];
        let schemes = [
            ("pooled", TwoFoldType::Pooled),
            ("nuemark", TwoFoldType::Neumark),
            ("cotton", TwoFoldType::Cotton),
            ("reimers", TwoFoldType::Reimers),
        ];
        for (key, kind) in schemes {
            let r = m.two_fold(kind).unwrap();
            let exp = &two[key];
            assert_rel(
                &format!("{name}.two_fold.{key}.unexplained"),
                r.unexplained,
                exp["unexplained"].as_f64().unwrap(),
                1e-8,
            );
            assert_rel(
                &format!("{name}.two_fold.{key}.explained"),
                r.explained,
                exp["explained"].as_f64().unwrap(),
                1e-8,
            );
            assert_rel(
                &format!("{name}.two_fold.{key}.gap"),
                r.gap,
                exp["gap"].as_f64().unwrap(),
                1e-8,
            );
            // Effects partition the gap exactly.
            assert!((r.unexplained + r.explained - r.gap).abs() < 1e-8);
        }

        // self_submitted with the recorded weight on the larger-mean group.
        let ss = &two["self_submitted"];
        let w = ss["weight"].as_f64().unwrap();
        let r = m.two_fold(TwoFoldType::SelfSubmitted(w)).unwrap();
        assert_rel(
            &format!("{name}.two_fold.self_submitted.unexplained"),
            r.unexplained,
            ss["unexplained"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.two_fold.self_submitted.explained"),
            r.explained,
            ss["explained"].as_f64().unwrap(),
            1e-8,
        );

        // Three-fold.
        let tf = m.three_fold();
        let exp = &case["three_fold"];
        assert_rel(
            &format!("{name}.three_fold.endowments"),
            tf.endowments,
            exp["endowments"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.three_fold.coefficients"),
            tf.coefficients,
            exp["coefficients"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.three_fold.interaction"),
            tf.interaction,
            exp["interaction"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.three_fold.gap"),
            tf.gap,
            exp["gap"].as_f64().unwrap(),
            1e-8,
        );
        // Three-fold partitions the gap exactly.
        assert!((tf.endowments + tf.coefficients + tf.interaction - tf.gap).abs() < 1e-8);
    }
}

// ---------------------------------------------------------------------------
// Distance covariance / correlation and the asymptotic dCov test.
//
// All `distance_statistics` quantities are closed-form functions of the
// pairwise euclidean-distance matrices (asserted 1e-8). The asymptotic test
// statistic is closed form (1e-8); its p-value passes through the standard
// normal cdf and is asserted 1e-6.
// ---------------------------------------------------------------------------
#[test]
fn dist_dependence_matches_reference() {
    let fx = load();
    for case in fx["dist"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let xd = case["x_dim"].as_u64().unwrap() as usize;
        let yd = case["y_dim"].as_u64().unwrap() as usize;
        let x = read_design(&case["x"], xd);
        let y = read_design(&case["y"], yd);

        let st = distance_statistics(&x, &y).unwrap();
        assert_rel(
            &format!("{name}.test_statistic"),
            st.test_statistic,
            case["test_statistic"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.distance_correlation"),
            st.distance_correlation,
            case["distance_correlation"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.distance_covariance"),
            st.distance_covariance,
            case["distance_covariance"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.dvar_x"),
            st.dvar_x,
            case["dvar_x"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.dvar_y"),
            st.dvar_y,
            case["dvar_y"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.S"),
            st.s,
            case["S"].as_f64().unwrap(),
            1e-8,
        );

        // Standalone entry points agree with the bundle.
        let dcov = distance_covariance(&x, &y).unwrap();
        assert_rel(
            &format!("{name}.dcov_fn"),
            dcov,
            case["dcov"].as_f64().unwrap(),
            1e-8,
        );
        let dcor = distance_correlation(&x, &y).unwrap();
        assert_rel(
            &format!("{name}.dcor_fn"),
            dcor,
            case["dcor"].as_f64().unwrap(),
            1e-8,
        );
        let dvarx = distance_variance(&x).unwrap();
        assert_rel(
            &format!("{name}.dvar_self_x"),
            dvarx,
            case["dvar_self_x"].as_f64().unwrap(),
            1e-8,
        );

        // Asymptotic dCov test.
        let t = distance_covariance_test(&x, &y).unwrap();
        assert_rel(
            &format!("{name}.asym_test_statistic"),
            t.statistic,
            case["asym_test_statistic"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.asym_pval"),
            t.pvalue,
            case["asym_pval"].as_f64().unwrap(),
            1e-6,
        );
    }
}

// ---------------------------------------------------------------------------
// Descriptive-statistics summary. Every reported quantity is closed-form (mean,
// std with ddof=1, percentiles by linear interpolation, skew/kurtosis biased
// estimators, MAD/IQR, CI bounds), so they match the reference at 1e-10.
// ---------------------------------------------------------------------------
#[test]
fn describe_matches_reference() {
    let fx = load();
    for case in fx["describe"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let data = vec1(&case["data"]);
        let alpha = case["alpha"].as_f64().unwrap();
        let use_t = case["use_t"].as_bool().unwrap();
        let d = describe(&data, alpha, use_t);

        assert_rel(
            &format!("{name}.nobs"),
            d.nobs,
            case["nobs"].as_f64().unwrap(),
            1e-12,
        );
        assert_rel(
            &format!("{name}.mean"),
            d.mean,
            case["mean"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{name}.std"),
            d.std,
            case["std"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{name}.std_err"),
            d.std_err,
            case["std_err"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{name}.upper_ci"),
            d.upper_ci,
            case["upper_ci"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{name}.lower_ci"),
            d.lower_ci,
            case["lower_ci"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{name}.iqr"),
            d.iqr,
            case["iqr"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{name}.mad"),
            d.mad,
            case["mad"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{name}.coef_var"),
            d.coef_var,
            case["coef_var"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{name}.range"),
            d.range,
            case["range"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{name}.min"),
            d.min,
            case["min"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{name}.max"),
            d.max,
            case["max"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{name}.skew"),
            d.skew,
            case["skew"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{name}.kurtosis"),
            d.kurtosis,
            case["kurtosis"].as_f64().unwrap(),
            1e-10,
        );
        assert_rel(
            &format!("{name}.median"),
            d.median,
            case["median"].as_f64().unwrap(),
            1e-10,
        );

        let exp_perc = vec1(&case["percentiles"]);
        assert_eq!(
            d.percentiles.len(),
            exp_perc.len(),
            "{name}.percentiles.len"
        );
        for (k, &p) in d.percentiles.iter().enumerate() {
            assert_rel(&format!("{name}.percentiles[{k}]"), p, exp_perc[k], 1e-10);
        }
    }
}

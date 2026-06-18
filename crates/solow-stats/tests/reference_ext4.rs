//! Cross-validation of the robust-covariance and correlation-tools extension
//! batch against golden reference values frozen in
//! `tests/fixtures/stats_ext4.json`.
//!
//! Covered: the heteroskedasticity-consistent OLS covariances (HC0..HC3), the
//! Newey-West HAC covariance with Bartlett weights, the one-way clustered
//! covariance, variance-inflation factors, the Lilliefors / KS normality test,
//! the nearest-PSD correlation/covariance tools, cov2corr/corr2cov, and the two
//! one-sided equivalence test (ttost_ind).

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_regression::LinearModel;
use solow_stats::{
    corr2cov, corr_clipped, corr_nearest, cov2corr, cov2corr_std, cov_cluster, cov_hac, cov_hc0,
    cov_hc1, cov_hc2, cov_hc3, cov_nearest, hat_diag, kstest_normal, lilliefors,
    variance_inflation_factor, LillieforsDist, NearestMethod, UseVar,
};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/stats_ext4.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_stats_ext4.py)");
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

fn assert_mat(label: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) {
    assert_eq!(got.dim(), want.dim(), "{label}: shape");
    for i in 0..got.nrows() {
        for j in 0..got.ncols() {
            assert_rel(&format!("{label}[{i},{j}]"), got[[i, j]], want[[i, j]], tol);
        }
    }
}

fn assert_vec(label: &str, got: &Array1<f64>, want: &Array1<f64>, tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for i in 0..got.len() {
        assert_rel(&format!("{label}[{i}]"), got[i], want[i], tol);
    }
}

fn bse_of(cov: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter((0..cov.nrows()).map(|i| cov[[i, i]].sqrt()))
}

// ---------------------------------------------------------------------------
// Robust sandwich covariances. The HC family and the cluster/HAC matrices are
// closed-form quadratic forms in the residuals, so both the covariance entries
// and the derived robust bse are asserted at 1e-8 (closed form). We refit the
// OLS ourselves and first confirm the residuals reproduce the reference.
// ---------------------------------------------------------------------------
#[test]
fn sandwich_covariances_match_reference() {
    let fx = load();
    for case in fx["sandwich"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let endog = vec1(&case["endog"]);
        let exog = mat(&case["exog"]);
        let res = LinearModel::ols(endog, exog.clone())
            .unwrap()
            .fit()
            .unwrap();
        let resid = res.resid.clone();

        // Sanity: residuals reproduced.
        assert_vec(
            &format!("{name}.resid"),
            &resid,
            &vec1(&case["resid"]),
            1e-8,
        );
        // Leverages used by HC2/HC3.
        let h = hat_diag(&exog).unwrap();
        assert_vec(
            &format!("{name}.hat_diag"),
            &h,
            &vec1(&case["hat_diag"]),
            1e-9,
        );

        let maxlags = case["maxlags"].as_u64().unwrap() as usize;
        let groups: Vec<i64> = case["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();

        // HC0..HC3 covariance + bse (closed form, 1e-8).
        let hc0 = cov_hc0(&exog, &resid).unwrap();
        assert_mat(
            &format!("{name}.cov_hc0"),
            &hc0,
            &mat(&case["cov_hc0"]),
            1e-8,
        );
        assert_vec(
            &format!("{name}.bse_hc0"),
            &bse_of(&hc0),
            &vec1(&case["bse_hc0"]),
            1e-8,
        );

        let hc1 = cov_hc1(&exog, &resid).unwrap();
        assert_mat(
            &format!("{name}.cov_hc1"),
            &hc1,
            &mat(&case["cov_hc1"]),
            1e-8,
        );
        assert_vec(
            &format!("{name}.bse_hc1"),
            &bse_of(&hc1),
            &vec1(&case["bse_hc1"]),
            1e-8,
        );

        let hc2 = cov_hc2(&exog, &resid).unwrap();
        assert_mat(
            &format!("{name}.cov_hc2"),
            &hc2,
            &mat(&case["cov_hc2"]),
            1e-8,
        );
        assert_vec(
            &format!("{name}.bse_hc2"),
            &bse_of(&hc2),
            &vec1(&case["bse_hc2"]),
            1e-8,
        );

        let hc3 = cov_hc3(&exog, &resid).unwrap();
        assert_mat(
            &format!("{name}.cov_hc3"),
            &hc3,
            &mat(&case["cov_hc3"]),
            1e-8,
        );
        assert_vec(
            &format!("{name}.bse_hc3"),
            &bse_of(&hc3),
            &vec1(&case["bse_hc3"]),
            1e-8,
        );

        // HAC (Newey-West, Bartlett), with and without the n/(n-k) correction.
        let hac_c = cov_hac(&exog, &resid, maxlags, true).unwrap();
        assert_mat(
            &format!("{name}.cov_hac_corr"),
            &hac_c,
            &mat(&case["cov_hac_corr"]),
            1e-8,
        );
        assert_vec(
            &format!("{name}.bse_hac_corr"),
            &bse_of(&hac_c),
            &vec1(&case["bse_hac_corr"]),
            1e-8,
        );
        let hac_n = cov_hac(&exog, &resid, maxlags, false).unwrap();
        assert_mat(
            &format!("{name}.cov_hac_nocorr"),
            &hac_n,
            &mat(&case["cov_hac_nocorr"]),
            1e-8,
        );
        assert_vec(
            &format!("{name}.bse_hac_nocorr"),
            &bse_of(&hac_n),
            &vec1(&case["bse_hac_nocorr"]),
            1e-8,
        );

        // One-way clustered, with and without the small-sample correction.
        let clu_c = cov_cluster(&exog, &resid, &groups, true).unwrap();
        assert_mat(
            &format!("{name}.cov_cluster_corr"),
            &clu_c,
            &mat(&case["cov_cluster_corr"]),
            1e-8,
        );
        assert_vec(
            &format!("{name}.bse_cluster_corr"),
            &bse_of(&clu_c),
            &vec1(&case["bse_cluster_corr"]),
            1e-8,
        );
        let clu_n = cov_cluster(&exog, &resid, &groups, false).unwrap();
        assert_mat(
            &format!("{name}.cov_cluster_nocorr"),
            &clu_n,
            &mat(&case["cov_cluster_nocorr"]),
            1e-8,
        );
        assert_vec(
            &format!("{name}.bse_cluster_nocorr"),
            &bse_of(&clu_n),
            &vec1(&case["bse_cluster_nocorr"]),
            1e-8,
        );
    }
}

// ---------------------------------------------------------------------------
// Variance-inflation factors: closed-form R² of an auxiliary OLS, asserted 1e-9.
// ---------------------------------------------------------------------------
#[test]
fn vif_matches_reference() {
    let fx = load();
    for case in fx["vif"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let exog = mat(&case["exog"]);
        let want = vec1(&case["vif"]);
        for j in 0..exog.ncols() {
            let v = variance_inflation_factor(&exog, j).unwrap();
            assert_rel(&format!("{name}.vif[{j}]"), v, want[j], 1e-9);
        }
    }
}

// ---------------------------------------------------------------------------
// Lilliefors / KS normality test. The KS statistic is closed form (asserted
// 1e-7). The Dalal-Wilkinson approximate p-value is closed form and asserted
// 1e-6, but only when the reference actually returned that closed-form value
// (i.e. it is below ~0.1); otherwise the reference substitutes a simulation
// table value we do not reproduce, and we assert only against the raw
// closed-form `pval_lf` (which our implementation computes exactly).
// ---------------------------------------------------------------------------
#[test]
fn lilliefors_matches_reference() {
    let fx = load();
    for case in fx["lilliefors"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let x = vec1(&case["x"]);
        let (stat, pval) = lilliefors(&x, LillieforsDist::Norm).unwrap();
        assert_rel(
            &format!("{name}.stat"),
            stat,
            case["stat"].as_f64().unwrap(),
            1e-7,
        );

        // Our closed-form p-value always equals the reference's raw pval_lf.
        assert_rel(
            &format!("{name}.pval_lf"),
            pval,
            case["pval_lf"].as_f64().unwrap(),
            1e-6,
        );

        // When the reference used the closed-form branch, its returned approx
        // p-value also matches tight.
        if case["closed_form"].as_bool().unwrap() {
            assert_rel(
                &format!("{name}.pval_approx"),
                pval,
                case["pval_approx"].as_f64().unwrap(),
                1e-6,
            );
        }

        // kstest_normal is an alias.
        let (statk, pvalk) = kstest_normal(&x).unwrap();
        assert_rel(
            &format!("{name}.kstest_stat"),
            statk,
            case["kstest_stat"].as_f64().unwrap(),
            1e-7,
        );
        assert!((statk - stat).abs() < 1e-15);
        assert!((pvalk - pval).abs() < 1e-15);
    }
}

// ---------------------------------------------------------------------------
// Correlation/covariance helpers and nearest-PSD projection.
//
// cov2corr / corr2cov are exact elementwise transforms (1e-12). corr_clipped is
// a single eigenvalue clip + rescale (asserted 1e-8, limited only by the
// eigensolver). corr_nearest / cov_nearest("nearest") are *iterative*
// alternating-projection algorithms that hit the iteration cap for genuinely
// indefinite inputs; even so, the reference and this implementation converge to
// the same fixed point — the measured max relative error is ~1e-15 — so they are
// asserted at 1e-9. cov_nearest("clipped") inherits the single-clip accuracy.
// ---------------------------------------------------------------------------
#[test]
fn correlation_tools_match_reference() {
    let fx = load();
    let cases = fx["corrtools"].as_array().unwrap();

    // Round-trip: cov2corr / corr2cov.
    let rt = cases.iter().find(|c| c["name"] == "roundtrip").unwrap();
    let cov = mat(&rt["cov"]);
    let (corr, std) = cov2corr_std(&cov);
    assert_mat("roundtrip.corr", &corr, &mat(&rt["corr"]), 1e-12);
    assert_mat(
        "roundtrip.corr_plain",
        &cov2corr(&cov),
        &mat(&rt["corr"]),
        1e-12,
    );
    assert_vec("roundtrip.std", &std, &vec1(&rt["std"]), 1e-12);
    let cov_back = corr2cov(&corr, &std);
    assert_mat(
        "roundtrip.cov_back",
        &cov_back,
        &mat(&rt["cov_back"]),
        1e-12,
    );

    // Nearest-PSD projections.
    let nr = cases.iter().find(|c| c["name"] == "nearest").unwrap();
    let bad = mat(&nr["bad_corr"]);
    let threshold = nr["threshold"].as_f64().unwrap();

    let clipped = corr_clipped(&bad, threshold).unwrap();
    assert_mat(
        "nearest.corr_clipped",
        &clipped,
        &mat(&nr["corr_clipped"]),
        1e-8,
    );

    let near = corr_nearest(&bad, threshold, 100).unwrap();
    // Iterative alternating projection. Despite both the reference and this
    // implementation hitting the iteration cap, the two converge to the same
    // fixed point; the measured max relative error is ~1e-15, so we assert 1e-9.
    assert_mat(
        "nearest.corr_nearest",
        &near,
        &mat(&nr["corr_nearest"]),
        1e-9,
    );

    let badcov = mat(&nr["bad_cov"]);
    let cn_clip = cov_nearest(&badcov, NearestMethod::Clipped, threshold, 100).unwrap();
    assert_mat(
        "nearest.cov_nearest_clipped",
        &cn_clip,
        &mat(&nr["cov_nearest_clipped"]),
        1e-8,
    );
    let cn_near = cov_nearest(&badcov, NearestMethod::Nearest, threshold, 100).unwrap();
    // Same as corr_nearest: converges to the reference fixed point (~1e-15).
    assert_mat(
        "nearest.cov_nearest_nearest",
        &cn_near,
        &mat(&nr["cov_nearest_nearest"]),
        1e-9,
    );
}

// ---------------------------------------------------------------------------
// Two one-sided equivalence test. Each one-sided leg is a two-sample t-test
// (closed-form statistic, t-distribution p-value), asserted 1e-8 for statistics
// and degrees of freedom and 1e-6 for p-values.
// ---------------------------------------------------------------------------
#[test]
fn ttost_ind_matches_reference() {
    let fx = load();
    for case in fx["tost"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let x1 = vec1(&case["x1"]);
        let x2 = vec1(&case["x2"]);
        let low = case["low"].as_f64().unwrap();
        let upp = case["upp"].as_f64().unwrap();
        let usevar = match case["usevar"].as_str().unwrap() {
            "pooled" => UseVar::Pooled,
            "unequal" => UseVar::Unequal,
            other => panic!("unknown usevar {other}"),
        };
        let r = solow_stats::ttost_ind(&x1, &x2, low, upp, usevar).unwrap();
        assert_rel(
            &format!("{name}.t1"),
            r.t1,
            case["t1"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.pv1"),
            r.pv1,
            case["pv1"].as_f64().unwrap(),
            1e-6,
        );
        assert_rel(
            &format!("{name}.t2"),
            r.t2,
            case["t2"].as_f64().unwrap(),
            1e-8,
        );
        assert_rel(
            &format!("{name}.pv2"),
            r.pv2,
            case["pv2"].as_f64().unwrap(),
            1e-6,
        );
        assert_rel(
            &format!("{name}.pvalue"),
            r.pvalue,
            case["pvalue"].as_f64().unwrap(),
            1e-6,
        );
    }
}

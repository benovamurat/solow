//! Cross-validation of the factor-analysis, MANOVA and canonical-correlation
//! estimators against golden reference values frozen in
//! `tests/fixtures/multivariate_ext.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_multivariate::{CanCorr, Factor, Manova};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/multivariate_ext.json"
    );
    let s = fs::read_to_string(p)
        .expect("fixture present (run tools/reference/gen_multivariate_ext.py)");
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
    let (m, n) = (rows.len(), rows[0].len());
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

fn check_scalar(label: &str, got: f64, want: f64, tol: f64) {
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{label}: rel-err {e:.3e} (got {got}, want {want})"
    );
}

fn check_vec(label: &str, got: &Array1<f64>, want: &Array1<f64>, tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
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

/// Per-column sign flips that align `ours` onto `theirs` (largest-magnitude
/// reference entry decides the sign of each column).
fn column_signs(ours: &Array2<f64>, theirs: &Array2<f64>) -> Vec<f64> {
    let nc = ours.ncols();
    let mut signs = vec![1.0; nc];
    for c in 0..nc {
        let mut best = 0usize;
        let mut best_mag = -1.0;
        for r in 0..theirs.nrows() {
            let m = theirs[[r, c]].abs();
            if m > best_mag {
                best_mag = m;
                best = r;
            }
        }
        let prod = ours[[best, c]] * theirs[[best, c]];
        signs[c] = if prod < 0.0 { -1.0 } else { 1.0 };
    }
    signs
}

// ---------------------------------------------------------------------------
// Factor analysis.
// ---------------------------------------------------------------------------
#[test]
fn factor_matches_reference() {
    let fx = load();
    for c in fx["factor"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let corr = mat(&c["corr"]);
        let n_factor = c["n_factor"].as_u64().unwrap() as usize;
        let smc = c["smc"].as_bool().unwrap();
        let res = Factor::from_corr(corr, n_factor, smc)
            .fit(500, 1e-10)
            .unwrap();
        let exp = &c["expected"];

        // Eigenvalues / communalities / uniquenesses are sign-invariant.
        check_vec(
            &format!("factor[{name}].eigenvals"),
            &res.eigenvals,
            &vec1(&exp["eigenvals"]),
            1e-7,
        );
        check_vec(
            &format!("factor[{name}].communality"),
            &res.communality,
            &vec1(&exp["communality"]),
            1e-7,
        );
        check_vec(
            &format!("factor[{name}].uniqueness"),
            &res.uniqueness,
            &vec1(&exp["uniqueness"]),
            1e-7,
        );

        // Loadings match up to a per-column sign.
        let want = mat(&exp["loadings"]);
        let signs = column_signs(&res.loadings, &want);
        assert_eq!(
            res.loadings.dim(),
            want.dim(),
            "factor[{name}].loadings shape"
        );
        for i in 0..want.nrows() {
            for j in 0..want.ncols() {
                let g = res.loadings[[i, j]] * signs[j];
                let e = rel(g, want[[i, j]]);
                assert!(
                    e <= 1e-5,
                    "factor[{name}].loadings[{i}][{j}]: rel-err {e:.3e} (got {g}, want {})",
                    want[[i, j]]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MANOVA.
// ---------------------------------------------------------------------------
fn check_stat(label: &str, got: &solow_multivariate::MvStat, exp: &Value) {
    check_scalar(
        &format!("{label}.value"),
        got.value,
        exp["value"].as_f64().unwrap(),
        1e-6,
    );
    check_scalar(
        &format!("{label}.num_df"),
        got.num_df,
        exp["num_df"].as_f64().unwrap(),
        1e-9,
    );
    check_scalar(
        &format!("{label}.den_df"),
        got.den_df,
        exp["den_df"].as_f64().unwrap(),
        1e-6,
    );
    check_scalar(
        &format!("{label}.f_value"),
        got.f_value,
        exp["f_value"].as_f64().unwrap(),
        1e-6,
    );
    check_scalar(
        &format!("{label}.pr_f"),
        got.pr_f,
        exp["pr_f"].as_f64().unwrap(),
        1e-6,
    );
}

#[test]
fn manova_matches_reference() {
    let fx = load();
    for c in fx["manova"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let endog = mat(&c["endog"]);
        let exog = mat(&c["exog"]);
        let m = Manova::new(endog, exog).unwrap();
        let tests = m.mv_test().unwrap();
        let exp = &c["expected"];
        for t in &tests {
            let e = &exp[&t.name];
            let label = format!("manova[{name}].{}", t.name);
            check_stat(
                &format!("{label}.wilks_lambda"),
                &t.stats.wilks_lambda,
                &e["wilks_lambda"],
            );
            check_stat(
                &format!("{label}.pillai_trace"),
                &t.stats.pillai_trace,
                &e["pillai_trace"],
            );
            check_stat(
                &format!("{label}.hotelling_lawley"),
                &t.stats.hotelling_lawley,
                &e["hotelling_lawley"],
            );
            check_stat(
                &format!("{label}.roy_greatest_root"),
                &t.stats.roy_greatest_root,
                &e["roy_greatest_root"],
            );
        }
    }
}

// ---------------------------------------------------------------------------
// CanCorr.
// ---------------------------------------------------------------------------
#[test]
fn cancorr_matches_reference() {
    let fx = load();
    for c in fx["cancorr"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let endog = mat(&c["endog"]);
        let exog = mat(&c["exog"]);
        let cc = CanCorr::new(&endog, &exog).unwrap();
        let exp = &c["expected"];

        check_vec(
            &format!("cancorr[{name}].cancorr"),
            &cc.cancorr,
            &vec1(&exp["cancorr"]),
            1e-7,
        );

        let rows = cc.corr_test();
        let want_rows = exp["corr_test"].as_array().unwrap();
        assert_eq!(
            rows.len(),
            want_rows.len(),
            "cancorr[{name}].corr_test length"
        );
        for (row, w) in rows.iter().zip(want_rows) {
            let label = format!("cancorr[{name}].corr_test[{}]", row.index);
            check_scalar(
                &format!("{label}.cancorr"),
                row.cancorr,
                w["cancorr"].as_f64().unwrap(),
                1e-7,
            );
            check_scalar(
                &format!("{label}.wilks_lambda"),
                row.wilks_lambda,
                w["wilks_lambda"].as_f64().unwrap(),
                1e-6,
            );
            check_scalar(
                &format!("{label}.num_df"),
                row.num_df,
                w["num_df"].as_f64().unwrap(),
                1e-9,
            );
            check_scalar(
                &format!("{label}.den_df"),
                row.den_df,
                w["den_df"].as_f64().unwrap(),
                1e-6,
            );
            check_scalar(
                &format!("{label}.f_value"),
                row.f_value,
                w["f_value"].as_f64().unwrap(),
                1e-6,
            );
            check_scalar(
                &format!("{label}.pr_f"),
                row.pr_f,
                w["pr_f"].as_f64().unwrap(),
                1e-6,
            );
        }
    }
}

//! Cross-validation of the PCA estimator against golden reference values
//! frozen in `tests/fixtures/multivariate.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use solow_multivariate::{Pca, PcaResults};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/multivariate.json"
    );
    let s =
        fs::read_to_string(p).expect("fixture present (run tools/reference/gen_multivariate.py)");
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

fn check_vec(label: &str, got: &Array1<f64>, exp: &Value, key: &str, tol: f64) {
    let want = vec1(&exp[key]);
    assert_eq!(got.len(), want.len(), "{label}.{key}: length");
    for i in 0..got.len() {
        let e = rel(got[i], want[i]);
        assert!(
            e <= tol,
            "{label}.{key}[{i}]: rel-err {e:.3e} (got {}, want {})",
            got[i],
            want[i]
        );
    }
}

fn check_mat(label: &str, got: &Array2<f64>, want: &Array2<f64>, key: &str, tol: f64) {
    assert_eq!(got.dim(), want.dim(), "{label}.{key}: shape");
    for i in 0..got.nrows() {
        for j in 0..got.ncols() {
            let e = rel(got[[i, j]], want[[i, j]]);
            assert!(
                e <= tol,
                "{label}.{key}[{i}][{j}]: rel-err {e:.3e} (got {}, want {})",
                got[[i, j]],
                want[[i, j]]
            );
        }
    }
}

/// For sign-arbitrary quantities, compute per-component signs that flip our
/// eigenvectors onto the reference, then verify all affected matrices with the
/// same signs applied consistently.
fn column_signs(ours: &Array2<f64>, theirs: &Array2<f64>) -> Vec<f64> {
    let nc = ours.ncols();
    let mut signs = vec![1.0; nc];
    for c in 0..nc {
        // Pick the entry with the largest reference magnitude to decide sign.
        let mut best = 0usize;
        let mut best_mag = -1.0;
        for r in 0..ours.nrows() {
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

fn apply_col_signs(m: &Array2<f64>, signs: &[f64]) -> Array2<f64> {
    let mut out = m.clone();
    for r in 0..out.nrows() {
        for c in 0..out.ncols() {
            out[[r, c]] *= signs[c];
        }
    }
    out
}

fn apply_row_signs(m: &Array2<f64>, signs: &[f64]) -> Array2<f64> {
    // coeff is ncomp x nvar; the per-component sign acts on the rows.
    let mut out = m.clone();
    for r in 0..out.nrows() {
        for c in 0..out.ncols() {
            out[[r, c]] *= signs[r];
        }
    }
    out
}

fn verify(label: &str, res: &PcaResults, exp: &Value) {
    // Sign-invariant quantities: tight (1e-7).
    check_vec(label, &res.eigenvals, exp, "eigenvals", 1e-7);
    check_vec(label, &res.rsquare, exp, "rsquare", 1e-7);
    check_vec(
        label,
        &res.explained_variance_ratio,
        exp,
        "explained_variance_ratio",
        1e-7,
    );
    check_vec(label, &res.mu, exp, "mu", 1e-7);
    check_vec(label, &res.sigma, exp, "sigma", 1e-7);

    // projection_full is the original-space reconstruction from all retained
    // components; sign-invariant, verified via the public `project` method.
    let want_proj = mat(&exp["projection_full"]);
    let got_proj = res.project(res.ncomp).unwrap();
    check_mat(label, &got_proj, &want_proj, "projection_full", 1e-7);

    // Sign-dependent quantities: align per-component sign with the reference.
    let want_eigenvecs = mat(&exp["eigenvecs"]);
    let signs = column_signs(&res.eigenvecs, &want_eigenvecs);

    let got_eigenvecs = apply_col_signs(&res.eigenvecs, &signs);
    check_mat(label, &got_eigenvecs, &want_eigenvecs, "eigenvecs", 1e-7);

    let want_loadings = mat(&exp["loadings"]);
    let got_loadings = apply_col_signs(&res.loadings, &signs);
    check_mat(label, &got_loadings, &want_loadings, "loadings", 1e-7);

    let want_factors = mat(&exp["factors"]);
    let got_factors = apply_col_signs(&res.factors, &signs);
    check_mat(label, &got_factors, &want_factors, "factors", 1e-7);

    let want_scores = mat(&exp["scores"]);
    let got_scores = apply_col_signs(&res.scores, &signs);
    check_mat(label, &got_scores, &want_scores, "scores", 1e-7);

    let want_coeff = mat(&exp["coeff"]);
    let got_coeff = apply_row_signs(&res.coeff, &signs);
    check_mat(label, &got_coeff, &want_coeff, "coeff", 1e-7);
}

fn build(c: &Value) -> PcaResults {
    let data = mat(&c["data"]);
    let ncomp = c["ncomp"].as_u64().unwrap() as usize;
    let standardize = c["standardize"].as_bool().unwrap();
    let demean = c["demean"].as_bool().unwrap();
    let normalize = c["normalize"].as_bool().unwrap();
    Pca::new(data)
        .ncomp(ncomp)
        .standardize(standardize)
        .demean(demean)
        .normalize(normalize)
        .fit()
        .unwrap()
}

#[test]
fn pca_matches_reference() {
    let fx = load();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let res = build(c);
        verify(name, &res, &c["expected"]);
    }
}

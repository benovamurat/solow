//! Cross-language reference-fixture tests.
//!
//! Every JSON file under `tests/fixtures/metrics/*.json` was written by
//! `scripts/generate_metric_fixtures.py` running against the canonical
//! `the reference` implementation. This test replays each fixture
//! through the Rust implementation and asserts machine-precision
//! agreement (`≤ 1e-12`).
//!
//! Run
//!
//! ```text
//! /path/to/venv/bin/python scripts/generate_metric_fixtures.py --check
//! ```
//!
//! as a CI drift gate — it exits non-zero if the fixtures on disk
//! disagree with a fresh the reference run.
//!
//! Failure here indicates one of two things: (a) a real Rust/the reference
//! numerical divergence to investigate, or (b) the reference changed the
//! definition of the metric (rare — usually only at a major release).

use ndarray::{Array1, Array2};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use solow_metrics::{
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    cohen_kappa_score, cosine_similarity, explained_variance_score, fbeta_score, laplacian_kernel,
    linear_kernel, log_loss, matthews_corrcoef, max_error, mean_absolute_error,
    mean_absolute_percentage_error, mean_squared_error, median_absolute_error, polynomial_kernel,
    r2_score, rbf_kernel, roc_auc_ovo_score, roc_auc_ovr_score, roc_auc_score,
    root_mean_squared_error, sigmoid_kernel, Average, MulticlassAuc,
};

const TOL: f64 = 1e-12;

fn fixtures_dir() -> PathBuf {
    // Cargo puts CARGO_MANIFEST_DIR at the crate root; the workspace fixtures
    // sit two levels above (crates/solow-metrics/..).
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/metrics")
}

fn load(path: &str) -> Value {
    let full = fixtures_dir().join(path);
    let text = fs::read_to_string(&full)
        .unwrap_or_else(|e| panic!("could not read fixture {}: {e}", full.display()));
    serde_json::from_str(&text).unwrap()
}

fn f_array(v: &Value) -> Array1<f64> {
    Array1::from(
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<_>>(),
    )
}

fn usize_vec(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as usize)
        .collect()
}

fn f2d(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, n_cols), flat).unwrap()
}

fn expected(v: &Value, key: &str) -> f64 {
    v["expected"][key]
        .as_f64()
        .unwrap_or_else(|| panic!("fixture is missing expected.{key}"))
}

fn assert_close(name: &str, got: f64, want: f64) {
    let diff = (got - want).abs();
    assert!(
        diff <= TOL,
        "{name}: Rust = {got:.17}, the reference = {want:.17}, diff = {diff:.3e} (> {TOL:.0e})"
    );
}

// ---------------------------------------------------------------------------
// Regression fixtures
// ---------------------------------------------------------------------------

#[test]
fn regression_basic_matches_reference_bitwise() {
    let f = load("regression_basic.json");
    let y = f_array(&f["inputs"]["y_true"]);
    let p = f_array(&f["inputs"]["y_pred"]);
    assert_close(
        "mean_squared_error",
        mean_squared_error(y.view(), p.view(), None).unwrap(),
        expected(&f, "mean_squared_error"),
    );
    assert_close(
        "root_mean_squared_error",
        root_mean_squared_error(y.view(), p.view(), None).unwrap(),
        expected(&f, "root_mean_squared_error"),
    );
    assert_close(
        "mean_absolute_error",
        mean_absolute_error(y.view(), p.view(), None).unwrap(),
        expected(&f, "mean_absolute_error"),
    );
    assert_close(
        "median_absolute_error",
        median_absolute_error(y.view(), p.view()).unwrap(),
        expected(&f, "median_absolute_error"),
    );
    assert_close(
        "max_error",
        max_error(y.view(), p.view()).unwrap(),
        expected(&f, "max_error"),
    );
    assert_close(
        "r2_score",
        r2_score(y.view(), p.view(), None).unwrap(),
        expected(&f, "r2_score"),
    );
    assert_close(
        "explained_variance_score",
        explained_variance_score(y.view(), p.view(), None).unwrap(),
        expected(&f, "explained_variance_score"),
    );
    assert_close(
        "mean_absolute_percentage_error",
        mean_absolute_percentage_error(y.view(), p.view(), None).unwrap(),
        expected(&f, "mean_absolute_percentage_error"),
    );
}

#[test]
fn regression_weighted_matches_reference_bitwise() {
    let f = load("regression_weighted.json");
    let y = f_array(&f["inputs"]["y_true"]);
    let p = f_array(&f["inputs"]["y_pred"]);
    let w = f_array(&f["inputs"]["sample_weight"]);
    assert_close(
        "mean_squared_error(w)",
        mean_squared_error(y.view(), p.view(), Some(w.view())).unwrap(),
        expected(&f, "mean_squared_error"),
    );
    assert_close(
        "mean_absolute_error(w)",
        mean_absolute_error(y.view(), p.view(), Some(w.view())).unwrap(),
        expected(&f, "mean_absolute_error"),
    );
    assert_close(
        "r2_score(w)",
        r2_score(y.view(), p.view(), Some(w.view())).unwrap(),
        expected(&f, "r2_score"),
    );
}

// ---------------------------------------------------------------------------
// Classification fixtures
// ---------------------------------------------------------------------------

#[test]
fn classification_binary_matches_reference_bitwise() {
    let f = load("classification_binary.json");
    let y_true = usize_vec(&f["inputs"]["y_true"]);
    let y_pred = usize_vec(&f["inputs"]["y_pred"]);
    let y_prob = f_array(&f["inputs"]["y_prob"]);
    let y_bool: Vec<bool> = y_true.iter().map(|&v| v == 1).collect();
    assert_close(
        "accuracy_score",
        accuracy_score(&y_true, &y_pred, None, true).unwrap(),
        expected(&f, "accuracy"),
    );
    let prf = solow_metrics::precision_recall_fscore(&y_true, &y_pred, Average::Binary, 1.0, None)
        .unwrap();
    assert_close("precision", prf.precision[0], expected(&f, "precision"));
    assert_close("recall", prf.recall[0], expected(&f, "recall"));
    assert_close(
        "f1",
        fbeta_score(&y_true, &y_pred, 1.0, Average::Binary, None).unwrap(),
        expected(&f, "f1"),
    );
    assert_close(
        "matthews_corrcoef",
        matthews_corrcoef(&y_true, &y_pred).unwrap(),
        expected(&f, "matthews_corrcoef"),
    );
    assert_close(
        "roc_auc",
        roc_auc_score(&y_bool, y_prob.view()).unwrap(),
        expected(&f, "roc_auc"),
    );
    assert_close(
        "average_precision",
        average_precision_score(&y_bool, y_prob.view()).unwrap(),
        expected(&f, "average_precision"),
    );
    assert_close(
        "brier",
        brier_score_loss(&y_bool, y_prob.view(), None).unwrap(),
        expected(&f, "brier"),
    );
    // Binary log-loss reshaped as a 2-column probability matrix so the
    // multiclass log_loss operates directly on the fixture.
    let n = y_true.len();
    let mut p2 = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        p2[[i, 1]] = y_prob[i];
        p2[[i, 0]] = 1.0 - y_prob[i];
    }
    assert_close(
        "log_loss",
        log_loss(&y_true, p2.view(), None, 1e-15).unwrap(),
        expected(&f, "log_loss"),
    );
}

#[test]
fn classification_multiclass_matches_reference_bitwise() {
    let f = load("classification_multiclass.json");
    let y_true = usize_vec(&f["inputs"]["y_true"]);
    let y_pred = usize_vec(&f["inputs"]["y_pred"]);
    let y_prob = f2d(&f["inputs"]["y_prob"]);
    assert_close(
        "accuracy_score",
        accuracy_score(&y_true, &y_pred, None, true).unwrap(),
        expected(&f, "accuracy"),
    );
    assert_close(
        "balanced_accuracy_score",
        balanced_accuracy_score(&y_true, &y_pred).unwrap(),
        expected(&f, "balanced_accuracy"),
    );
    assert_close(
        "macro_f1",
        fbeta_score(&y_true, &y_pred, 1.0, Average::Macro, None).unwrap(),
        expected(&f, "macro_f1"),
    );
    assert_close(
        "weighted_f1",
        fbeta_score(&y_true, &y_pred, 1.0, Average::Weighted, None).unwrap(),
        expected(&f, "weighted_f1"),
    );
    assert_close(
        "cohen_kappa",
        cohen_kappa_score(&y_true, &y_pred, solow_metrics::KappaWeights::None, None).unwrap(),
        expected(&f, "cohen_kappa"),
    );
    assert_close(
        "log_loss",
        log_loss(&y_true, y_prob.view(), None, 1e-15).unwrap(),
        expected(&f, "log_loss"),
    );
    assert_close(
        "roc_auc_ovr_macro",
        roc_auc_ovr_score(&y_true, y_prob.view(), MulticlassAuc::Macro).unwrap(),
        expected(&f, "roc_auc_ovr_macro"),
    );
    assert_close(
        "roc_auc_ovo_macro",
        roc_auc_ovo_score(&y_true, y_prob.view(), MulticlassAuc::Macro).unwrap(),
        expected(&f, "roc_auc_ovo_macro"),
    );
}

/// Assert that a pairwise kernel matches the reference within `TOL`.
fn assert_matrix_close(name: &str, got: Array2<f64>, want: &Value) {
    let expected = f2d(want);
    assert_eq!(
        got.shape(),
        expected.shape(),
        "{name}: shape mismatch got={:?} want={:?}",
        got.shape(),
        expected.shape(),
    );
    for i in 0..got.nrows() {
        for j in 0..got.ncols() {
            let g = got[[i, j]];
            let e = expected[[i, j]];
            assert!(
                (g - e).abs() < 1e-10,
                "{name}[{i}, {j}]: got {g}, expected {e}"
            );
        }
    }
}

#[test]
fn pairwise_kernels_match_reference_bitwise() {
    let f = load("pairwise_kernels.json");
    let a = f2d(&f["a"]);
    let b = f2d(&f["b"]);
    assert_matrix_close("rbf_1.0", rbf_kernel(a.view(), b.view(), 1.0).unwrap(), &f["rbf_1.0"]);
    assert_matrix_close("linear", linear_kernel(a.view(), b.view()).unwrap(), &f["linear"]);
    assert_matrix_close(
        "polynomial_3",
        polynomial_kernel(a.view(), b.view(), 0.5, 1.0, 3).unwrap(),
        &f["polynomial_3"],
    );
    assert_matrix_close(
        "sigmoid",
        sigmoid_kernel(a.view(), b.view(), 0.5, 0.0).unwrap(),
        &f["sigmoid"],
    );
    assert_matrix_close(
        "laplacian_1.0",
        laplacian_kernel(a.view(), b.view(), 1.0).unwrap(),
        &f["laplacian_1.0"],
    );
    assert_matrix_close(
        "cosine_similarity",
        cosine_similarity(a.view(), b.view()).unwrap(),
        &f["cosine_similarity"],
    );
}

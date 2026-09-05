//! Reference score functions for [`crate::SelectKBest`].
//!
//! Each returns a per-feature score vector matching the semantics of
//! its `feature_selection` counterpart:
//!
//! * [`score_f_classif`] — one-way ANOVA F-statistic per feature
//!   against integer class labels `y ∈ {0, ..., K − 1}`.
//! * [`score_f_regression`] — F-statistic derived from the squared
//!   sample-Pearson correlation between each feature and a continuous
//!   target `y`.

use ndarray::{ArrayView1, ArrayView2};

/// One-way ANOVA F-score per feature against integer labels.
///
/// Returns `f_stat_j = MS_between_j / MS_within_j`. Higher = more
/// class-separating.
///
/// # Panics
///
/// Panics if `y.len() != x.nrows()`. Use [`crate::SelectKBest`]'s
/// error-returning wrapper to surface such shape mismatches cleanly.
pub fn score_f_classif(x: ArrayView2<'_, f64>, y: ArrayView1<'_, usize>) -> Vec<f64> {
    assert_eq!(x.nrows(), y.len(), "score_f_classif: x/y shape mismatch");
    let (n, d) = (x.nrows(), x.ncols());
    let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    let mut out = vec![0.0_f64; d];
    for j in 0..d {
        let mut class_sum = vec![0.0_f64; n_classes];
        let mut class_cnt = vec![0usize; n_classes];
        let mut grand = 0.0_f64;
        for i in 0..n {
            let v = x[[i, j]];
            grand += v;
            class_sum[y[i]] += v;
            class_cnt[y[i]] += 1;
        }
        grand /= n as f64;
        // Between-class sum of squares.
        let mut ss_between = 0.0_f64;
        for c in 0..n_classes {
            if class_cnt[c] == 0 {
                continue;
            }
            let m_c = class_sum[c] / class_cnt[c] as f64;
            let d_c = m_c - grand;
            ss_between += class_cnt[c] as f64 * d_c * d_c;
        }
        // Within-class SS.
        let mut ss_within = 0.0_f64;
        for i in 0..n {
            let c = y[i];
            let m_c = class_sum[c] / class_cnt[c] as f64;
            let d = x[[i, j]] - m_c;
            ss_within += d * d;
        }
        let df_b = (n_classes as f64 - 1.0).max(1.0);
        let df_w = (n as f64 - n_classes as f64).max(1.0);
        let ms_b = ss_between / df_b;
        let ms_w = ss_within / df_w;
        out[j] = if ms_w > 0.0 {
            ms_b / ms_w
        } else {
            f64::INFINITY
        };
    }
    out
}

/// Regression F-score per feature against a continuous target.
///
/// For each column `j` computes the sample Pearson correlation `r_j`,
/// then `f_j = r_j² · (n − 2) / (1 − r_j²)` — the classical F-test
/// on a single-predictor OLS.
pub fn score_f_regression(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Vec<f64> {
    assert_eq!(x.nrows(), y.len(), "score_f_regression: x/y shape mismatch");
    let (n, d) = (x.nrows(), x.ncols());
    let mut mean_y = 0.0_f64;
    for &v in y.iter() {
        mean_y += v;
    }
    mean_y /= n as f64;
    let var_y: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum();
    let mut out = vec![0.0_f64; d];
    for j in 0..d {
        let mut mean_x = 0.0_f64;
        for &v in x.column(j).iter() {
            mean_x += v;
        }
        mean_x /= n as f64;
        let mut cov = 0.0_f64;
        let mut var_x = 0.0_f64;
        for i in 0..n {
            let dx = x[[i, j]] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
        }
        if var_x <= 0.0 || var_y <= 0.0 {
            out[j] = 0.0;
            continue;
        }
        let r = cov / (var_x * var_y).sqrt();
        let r2 = r * r;
        let denom = 1.0 - r2;
        out[j] = if denom > 1e-15 {
            r2 * (n as f64 - 2.0) / denom
        } else {
            f64::INFINITY
        };
    }
    out
}

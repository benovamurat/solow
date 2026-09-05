//! Variance-threshold and top-k filter selectors.

use ndarray::{Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

// ---------------------------------------------------------------------------
// VarianceThreshold
// ---------------------------------------------------------------------------

/// Drops every column whose sample variance is at or below `threshold`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct VarianceThreshold {
    /// Threshold — columns with `var ≤ threshold` are dropped.
    pub threshold: f64,
    /// Indices of the columns kept, in ascending order.
    pub support: Vec<usize>,
    /// Per-column variance observed at fit time (length = original `d`).
    pub variances: Vec<f64>,
}

impl VarianceThreshold {
    /// Fit on `x`.
    pub fn fit(x: ArrayView2<'_, f64>, threshold: f64) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "VarianceThreshold::fit: x must be non-empty".into(),
            ));
        }
        if !(threshold >= 0.0 && threshold.is_finite()) {
            return Err(Error::Value(format!(
                "VarianceThreshold::fit: threshold must be finite and ≥ 0 (got {threshold})"
            )));
        }
        let d = x.ncols();
        let mut variances = Vec::with_capacity(d);
        let mut support = Vec::new();
        for j in 0..d {
            let (m, v) = mean_and_var(x.column(j));
            variances.push(v);
            if v > threshold {
                support.push(j);
            }
            let _ = m;
        }
        Ok(Self {
            threshold,
            support,
            variances,
        })
    }

    /// Return `x` with only the selected columns.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.variances.len() {
            return Err(Error::Shape(format!(
                "VarianceThreshold::transform: expected {} columns, got {}",
                self.variances.len(),
                x.ncols()
            )));
        }
        let mut out = Array2::<f64>::zeros((x.nrows(), self.support.len()));
        for (ci, &c) in self.support.iter().enumerate() {
            for i in 0..x.nrows() {
                out[[i, ci]] = x[[i, c]];
            }
        }
        Ok(out)
    }

    /// One-call fit + transform.
    pub fn fit_transform(x: ArrayView2<'_, f64>, threshold: f64) -> Result<(Self, Array2<f64>)> {
        let s = Self::fit(x, threshold)?;
        let t = s.transform(x)?;
        Ok((s, t))
    }
}

// ---------------------------------------------------------------------------
// SelectKBest
// ---------------------------------------------------------------------------

/// Keeps the `k` columns with the highest score under a caller-
/// supplied score function.
///
/// The score function takes `(x, y)` and returns a per-feature
/// score vector (higher = more informative). Free score functions
/// [`crate::scores::score_f_classif`] and [`crate::scores::score_f_regression`]
/// implement the ANOVA-F and regression-F scores respectively.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SelectKBest {
    /// Number of top columns retained.
    pub k: usize,
    /// Selected column indices, ascending.
    pub support: Vec<usize>,
    /// Per-column score computed at fit time (length = original `d`).
    pub scores: Vec<f64>,
}

impl SelectKBest {
    /// Fit.
    pub fn fit<F>(x: ArrayView2<'_, f64>, y_scores: F, k: usize) -> Result<Self>
    where
        F: Fn(ArrayView2<'_, f64>) -> Vec<f64>,
    {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value("SelectKBest::fit: x must be non-empty".into()));
        }
        if k == 0 || k > x.ncols() {
            return Err(Error::Value(format!(
                "SelectKBest::fit: k must be in [1, d] (got k={k}, d={})",
                x.ncols()
            )));
        }
        let scores = y_scores(x);
        if scores.len() != x.ncols() {
            return Err(Error::Shape(format!(
                "SelectKBest::fit: score function returned {} values for {} columns",
                scores.len(),
                x.ncols()
            )));
        }
        let mut ranked: Vec<(usize, f64)> = scores
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let mut support: Vec<usize> = ranked.iter().take(k).map(|(i, _)| *i).collect();
        support.sort();
        Ok(Self { k, support, scores })
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.scores.len() {
            return Err(Error::Shape(format!(
                "SelectKBest::transform: expected {} columns, got {}",
                self.scores.len(),
                x.ncols()
            )));
        }
        let mut out = Array2::<f64>::zeros((x.nrows(), self.support.len()));
        for (ci, &c) in self.support.iter().enumerate() {
            for i in 0..x.nrows() {
                out[[i, ci]] = x[[i, c]];
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mean_and_var(col: ArrayView1<'_, f64>) -> (f64, f64) {
    let n = col.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let (mut m, mut m2, mut k) = (0.0_f64, 0.0_f64, 0usize);
    for &v in col.iter() {
        k += 1;
        let delta = v - m;
        m += delta / k as f64;
        let delta2 = v - m;
        m2 += delta * delta2;
    }
    let var = if n > 1 { m2 / (n as f64 - 1.0) } else { 0.0 };
    (m, var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn variance_threshold_drops_constant_column() {
        let x = array![[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]];
        let (vt, t) = VarianceThreshold::fit_transform(x.view(), 0.0).unwrap();
        assert_eq!(vt.support, vec![1]);
        assert_eq!(t.dim(), (3, 1));
    }

    #[test]
    fn selectkbest_picks_top_k_by_score() {
        // Feature 0 constant → score 0. Feature 1 monotone → high score.
        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        // Return column variance as a stand-in "score".
        let sk = SelectKBest::fit(
            x.view(),
            |x| {
                (0..x.ncols())
                    .map(|j| {
                        let (_, v) = mean_and_var(x.column(j));
                        v
                    })
                    .collect()
            },
            1,
        )
        .unwrap();
        assert_eq!(sk.support, vec![1]);
    }
}

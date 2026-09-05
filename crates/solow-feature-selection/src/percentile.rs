//! `SelectPercentile`, `SelectFpr`, `SelectFdr`, `SelectFwe` — the
//! univariate score-threshold selectors from `feature_selection`.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Select the top `percentile`% features by a caller-supplied score.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SelectPercentile {
    /// Kept column indices, sorted ascending.
    pub selected: Vec<usize>,
    /// Percentile used.
    pub percentile: f64,
    /// Per-column score.
    pub scores: Vec<f64>,
}

impl SelectPercentile {
    /// Fit with the caller's `(x, y) → per-column scores` closure and
    /// keep the top `percentile`% features (percentile ∈ [0, 100]).
    pub fn fit<F>(x: ArrayView2<'_, f64>, y_scores: F, percentile: f64) -> Result<Self>
    where
        F: FnOnce(ArrayView2<'_, f64>) -> Result<Vec<f64>>,
    {
        if !(0.0..=100.0).contains(&percentile) {
            return Err(Error::Value("SelectPercentile: percentile must be in [0, 100]".into()));
        }
        let d = x.ncols();
        let scores = y_scores(x)?;
        if scores.len() != d {
            return Err(Error::Shape(
                "SelectPercentile: score vector length ≠ x.ncols()".into(),
            ));
        }
        let k = ((percentile / 100.0) * d as f64).round().max(1.0) as usize;
        let mut idx: Vec<usize> = (0..d).collect();
        idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        idx.truncate(k);
        idx.sort();
        Ok(Self {
            selected: idx,
            percentile,
            scores,
        })
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let mut out = Array2::<f64>::zeros((n, self.selected.len()));
        for i in 0..n {
            for (r, &j) in self.selected.iter().enumerate() {
                out[[i, r]] = x[[i, j]];
            }
        }
        Ok(out)
    }
}

/// Select features with a p-value strictly below the FPR threshold.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SelectFpr {
    /// Kept column indices, sorted ascending.
    pub selected: Vec<usize>,
    /// Per-column p-values.
    pub pvalues: Vec<f64>,
    /// Threshold used.
    pub alpha: f64,
}

impl SelectFpr {
    /// Fit with the caller's `(x) → per-column p-values` closure.
    pub fn fit<F>(x: ArrayView2<'_, f64>, y_pvalues: F, alpha: f64) -> Result<Self>
    where
        F: FnOnce(ArrayView2<'_, f64>) -> Result<Vec<f64>>,
    {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(Error::Value("SelectFpr: alpha must be in [0, 1]".into()));
        }
        let pvalues = y_pvalues(x)?;
        let d = x.ncols();
        if pvalues.len() != d {
            return Err(Error::Shape("SelectFpr: p-value length ≠ x.ncols()".into()));
        }
        let selected: Vec<usize> = (0..d).filter(|&j| pvalues[j] < alpha).collect();
        Ok(Self { selected, pvalues, alpha })
    }
}

/// Select features under a Benjamini-Hochberg false-discovery-rate.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SelectFdr {
    /// Kept column indices, sorted ascending.
    pub selected: Vec<usize>,
    /// Per-column p-values.
    pub pvalues: Vec<f64>,
    /// Threshold used.
    pub alpha: f64,
}

impl SelectFdr {
    /// Fit — same closure API as `SelectFpr`.
    pub fn fit<F>(x: ArrayView2<'_, f64>, y_pvalues: F, alpha: f64) -> Result<Self>
    where
        F: FnOnce(ArrayView2<'_, f64>) -> Result<Vec<f64>>,
    {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(Error::Value("SelectFdr: alpha must be in [0, 1]".into()));
        }
        let pvalues = y_pvalues(x)?;
        let d = x.ncols();
        // Benjamini-Hochberg step-up.
        let mut ranked: Vec<(usize, f64)> =
            pvalues.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut cutoff_rank = 0_usize;
        for (rank, (_i, p)) in ranked.iter().enumerate() {
            let threshold = (rank + 1) as f64 * alpha / d as f64;
            if *p <= threshold {
                cutoff_rank = rank + 1;
            }
        }
        let selected: Vec<usize> = ranked.iter().take(cutoff_rank).map(|(i, _)| *i).collect();
        let mut selected = selected;
        selected.sort();
        Ok(Self { selected, pvalues, alpha })
    }
}

/// Select features under a Bonferroni family-wise error correction.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SelectFwe {
    /// Kept column indices, sorted ascending.
    pub selected: Vec<usize>,
    /// Per-column p-values.
    pub pvalues: Vec<f64>,
    /// Threshold used.
    pub alpha: f64,
}

impl SelectFwe {
    /// Fit — same closure API as `SelectFpr`.
    pub fn fit<F>(x: ArrayView2<'_, f64>, y_pvalues: F, alpha: f64) -> Result<Self>
    where
        F: FnOnce(ArrayView2<'_, f64>) -> Result<Vec<f64>>,
    {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(Error::Value("SelectFwe: alpha must be in [0, 1]".into()));
        }
        let pvalues = y_pvalues(x)?;
        let d = x.ncols();
        let bonf = alpha / d as f64;
        let selected: Vec<usize> = (0..d).filter(|&j| pvalues[j] < bonf).collect();
        Ok(Self { selected, pvalues, alpha })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn select_percentile_keeps_top_50pct() {
        let x = array![[0.0_f64, 1.0, 2.0, 3.0]];
        let sp = SelectPercentile::fit(x.view(), |_| Ok(vec![10.0, 5.0, 20.0, 1.0]), 50.0).unwrap();
        assert_eq!(sp.selected, vec![0, 2]);
    }

    #[test]
    fn select_fpr_drops_columns_with_high_pvalue() {
        let x = array![[0.0_f64, 1.0, 2.0]];
        let sp = SelectFpr::fit(x.view(), |_| Ok(vec![0.001, 0.5, 0.02]), 0.05).unwrap();
        assert_eq!(sp.selected, vec![0, 2]);
    }
}

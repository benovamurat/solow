//! Recursive Feature Elimination (Guyon-Weston-Barnhill-Vapnik 2002).
//!
//! Given an importance-producing model, RFE iteratively eliminates
//! the `step` least-important features at every round until
//! `n_features_to_select` remain.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Recursive-feature-elimination selector.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Rfe {
    /// Number of features to retain.
    pub n_features_to_select: usize,
    /// Features eliminated per round.
    pub step: usize,
    /// Final selected feature indices, ascending.
    pub support: Vec<usize>,
    /// Elimination ranking — feature `j` ranked `ranking[j]` (1 = retained).
    pub ranking: Vec<usize>,
}

impl Rfe {
    /// Fit RFE around a caller-supplied ranker.
    ///
    /// `ranker(x_sub)` returns an importance vector aligned with the
    /// columns of `x_sub` (a submatrix of the original `x` with the
    /// currently-retained columns). Higher = more important.
    pub fn fit<F>(
        x: ArrayView2<'_, f64>,
        n_features_to_select: usize,
        step: usize,
        mut ranker: F,
    ) -> Result<Self>
    where
        F: FnMut(ArrayView2<'_, f64>) -> Vec<f64>,
    {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value("Rfe::fit: x must be non-empty".into()));
        }
        if n_features_to_select == 0 || n_features_to_select > x.ncols() {
            return Err(Error::Value(format!(
                "Rfe::fit: n_features_to_select must be in [1, d] (got {n_features_to_select}, d={})",
                x.ncols()
            )));
        }
        if step == 0 {
            return Err(Error::Value("Rfe::fit: step must be ≥ 1".into()));
        }
        let d = x.ncols();
        let mut kept: Vec<usize> = (0..d).collect();
        let mut ranking = vec![1usize; d];
        let mut elimination_round = d;
        while kept.len() > n_features_to_select {
            // Build the sub-matrix.
            let mut sub = Array2::<f64>::zeros((x.nrows(), kept.len()));
            for (jj, &j) in kept.iter().enumerate() {
                for i in 0..x.nrows() {
                    sub[[i, jj]] = x[[i, j]];
                }
            }
            let scores = ranker(sub.view());
            if scores.len() != kept.len() {
                return Err(Error::Shape(format!(
                    "Rfe::fit: ranker returned {} values for {} kept columns",
                    scores.len(),
                    kept.len()
                )));
            }
            // Sort by score ascending → drop the lowest `step` (but keep at
            // least `n_features_to_select`).
            let mut idx: Vec<usize> = (0..kept.len()).collect();
            idx.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap().then(a.cmp(&b)));
            let drop_count = step.min(kept.len() - n_features_to_select);
            let dropped: Vec<usize> = idx[..drop_count].iter().map(|&ii| kept[ii]).collect();
            for &feat in &dropped {
                ranking[feat] = elimination_round;
            }
            elimination_round = elimination_round.saturating_sub(1).max(2);
            kept.retain(|f| !dropped.contains(f));
        }
        // Retained features get rank 1.
        for &j in &kept {
            ranking[j] = 1;
        }
        kept.sort();
        Ok(Self {
            n_features_to_select,
            step,
            support: kept,
            ranking,
        })
    }

    /// Return only the selected columns of `x`.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.ranking.len() {
            return Err(Error::Shape(format!(
                "Rfe::transform: expected {} columns, got {}",
                self.ranking.len(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn rfe_keeps_the_most_informative_features() {
        // Column 0 varies (informative), column 1 constant (uninformative).
        let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]];
        let rfe = Rfe::fit(x.view(), 1, 1, |sub| {
            (0..sub.ncols())
                .map(|j| {
                    let mut s = 0.0_f64;
                    let mut m = 0.0_f64;
                    for i in 0..sub.nrows() {
                        m += sub[[i, j]];
                    }
                    m /= sub.nrows() as f64;
                    for i in 0..sub.nrows() {
                        s += (sub[[i, j]] - m).powi(2);
                    }
                    s
                })
                .collect()
        })
        .unwrap();
        assert_eq!(rfe.support, vec![0]);
        assert_eq!(rfe.ranking[0], 1);
        assert!(rfe.ranking[1] > 1);
    }
}

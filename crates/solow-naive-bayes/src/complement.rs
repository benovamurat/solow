//! Complement Naive Bayes (Rennie et al. 2003).
//!
//! Instead of estimating `p(x_j | y = c)` from class `c`'s samples,
//! estimates it from the *complement* of `c` — i.e. every class
//! except `c`. The classifier picks the class whose complement is
//! *least* likely to have generated the observation. This addresses
//! the bias multinomial NB shows on imbalanced text corpora.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

use crate::gaussian::{argmax_rows, log_softmax_rows, softmax_rows};

/// Complement Naive Bayes classifier.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct ComplementNB {
    /// Per-class prior.
    pub class_prior: Array1<f64>,
    /// Per-class complement log-weights (matches `the reference`'s `feature_log_prob_`).
    pub feature_log_prob: Array2<f64>,
    /// Distinct class count.
    pub n_classes: usize,
    /// Smoothing constant.
    pub alpha: f64,
    /// Whether to L1-normalise the per-class weights (`norm=True` in
    /// the reference, off by default there and here).
    pub normalize: bool,
}

impl ComplementNB {
    /// Fit with `α = 1`, `normalize = false`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, usize>) -> Result<Self> {
        Self::fit_with(x, y, 1.0, false)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        alpha: f64,
        normalize: bool,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "ComplementNB::fit_with: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        if alpha < 0.0 {
            return Err(Error::Value(format!(
                "ComplementNB::fit_with: alpha must be ≥ 0 (got {alpha})"
            )));
        }
        for &v in x.iter() {
            if v < 0.0 {
                return Err(Error::Value(
                    "ComplementNB::fit_with: features must be non-negative".into(),
                ));
            }
        }
        let (n, d) = (x.nrows(), x.ncols());
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        // Feature totals per class.
        let mut feature_count = Array2::<f64>::zeros((n_classes, d));
        let mut class_count = Array1::<f64>::zeros(n_classes);
        for i in 0..n {
            let c = y[i];
            class_count[c] += 1.0;
            for j in 0..d {
                feature_count[[c, j]] += x[[i, j]];
            }
        }
        // Grand totals across classes.
        let feature_total: Array1<f64> = feature_count.sum_axis(ndarray::Axis(0));
        let all_total: f64 = feature_total.iter().sum();
        // Complement counts.
        let mut feature_log_prob = Array2::<f64>::zeros((n_classes, d));
        for c in 0..n_classes {
            let comp_total = all_total - feature_count.row(c).iter().sum::<f64>();
            let denom = comp_total + alpha * d as f64;
            for j in 0..d {
                let comp = feature_total[j] - feature_count[[c, j]];
                let w = -((comp + alpha) / denom).ln();
                feature_log_prob[[c, j]] = w;
            }
            if normalize {
                let row_norm: f64 = (0..d).map(|j| feature_log_prob[[c, j]].abs()).sum();
                if row_norm > 0.0 {
                    for j in 0..d {
                        feature_log_prob[[c, j]] /= row_norm;
                    }
                }
            }
        }
        let class_prior =
            Array1::from_shape_fn(n_classes, |c| (class_count[c] / n as f64).max(1e-300));
        Ok(Self {
            class_prior,
            feature_log_prob,
            n_classes,
            alpha,
            normalize,
        })
    }

    /// Log-posterior joint.
    pub fn predict_log_joint(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((x.nrows(), self.n_classes));
        for i in 0..x.nrows() {
            for c in 0..self.n_classes {
                let mut s = self.class_prior[c].ln();
                for j in 0..x.ncols() {
                    s += x[[i, j]] * self.feature_log_prob[[c, j]];
                }
                out[[i, c]] = s;
            }
        }
        out
    }

    /// Class posteriors.
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        softmax_rows(&self.predict_log_joint(x))
    }

    /// Log posteriors.
    pub fn predict_log_proba(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        log_softmax_rows(&self.predict_log_joint(x))
    }

    /// Class labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Array1<usize> {
        argmax_rows(&self.predict_log_joint(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn separates_imbalanced_word_distributions() {
        let x = array![
            [4.0, 3.0, 0.0, 0.0],
            [5.0, 4.0, 0.0, 1.0],
            [3.0, 4.0, 1.0, 0.0],
            [4.0, 5.0, 0.0, 0.0],
            [5.0, 3.0, 1.0, 0.0],
            [0.0, 0.0, 4.0, 3.0],
            [1.0, 0.0, 5.0, 4.0],
        ];
        let y = Array1::from(vec![0usize, 0, 0, 0, 0, 1, 1]);
        let m = ComplementNB::fit(x.view(), y.view()).unwrap();
        assert_eq!(m.predict(x.view()), y);
    }
}

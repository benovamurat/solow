//! Bernoulli Naive Bayes for binary features.
//!
//! `log p(y = c | x) ∝ log p(y = c) + Σ_j x_j · log θ_{c,j} + (1 − x_j) · log(1 − θ_{c,j})`
//! where `θ_{c, j} = (Σ_i x_{ij} · [y_i = c] + α) / (N_c + 2α)`. Features
//! that are neither 0 nor 1 are binarised by comparison to `binarize`
//! (default 0.0), matching the reference.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

use crate::gaussian::{argmax_rows, log_softmax_rows, softmax_rows};

/// Bernoulli Naive Bayes classifier.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BernoulliNB {
    /// Per-class prior.
    pub class_prior: Array1<f64>,
    /// `log θ_{c, j}`.
    pub log_theta: Array2<f64>,
    /// `log (1 − θ_{c, j})`.
    pub log_neg_theta: Array2<f64>,
    /// Distinct class count.
    pub n_classes: usize,
    /// Smoothing constant.
    pub alpha: f64,
    /// Binarisation threshold (values `> binarize` map to 1, else 0).
    pub binarize: f64,
}

impl BernoulliNB {
    /// Fit with `α = 1`, `binarize = 0.0`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, usize>) -> Result<Self> {
        Self::fit_with(x, y, 1.0, 0.0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        alpha: f64,
        binarize: f64,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "BernoulliNB::fit_with: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        if alpha < 0.0 {
            return Err(Error::Value(format!(
                "BernoulliNB::fit_with: alpha must be ≥ 0 (got {alpha})"
            )));
        }
        let (n, d) = (x.nrows(), x.ncols());
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        let mut count = Array2::<f64>::zeros((n_classes, d));
        let mut class_count = Array1::<f64>::zeros(n_classes);
        for i in 0..n {
            let c = y[i];
            class_count[c] += 1.0;
            for j in 0..d {
                if x[[i, j]] > binarize {
                    count[[c, j]] += 1.0;
                }
            }
        }
        let mut log_theta = Array2::<f64>::zeros((n_classes, d));
        let mut log_neg_theta = Array2::<f64>::zeros((n_classes, d));
        for c in 0..n_classes {
            let denom = class_count[c] + 2.0 * alpha;
            for j in 0..d {
                let theta = (count[[c, j]] + alpha) / denom;
                log_theta[[c, j]] = theta.max(1e-300).ln();
                log_neg_theta[[c, j]] = (1.0 - theta).max(1e-300).ln();
            }
        }
        let class_prior =
            Array1::from_shape_fn(n_classes, |c| (class_count[c] / n as f64).max(1e-300));
        Ok(Self {
            class_prior,
            log_theta,
            log_neg_theta,
            n_classes,
            alpha,
            binarize,
        })
    }

    /// Log-posterior joint.
    pub fn predict_log_joint(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((x.nrows(), self.n_classes));
        for i in 0..x.nrows() {
            for c in 0..self.n_classes {
                let mut s = self.class_prior[c].ln();
                for j in 0..x.ncols() {
                    if x[[i, j]] > self.binarize {
                        s += self.log_theta[[c, j]];
                    } else {
                        s += self.log_neg_theta[[c, j]];
                    }
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
    fn separates_binary_signal() {
        let x = array![
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0]
        ];
        let y = Array1::from(vec![0usize, 0, 0, 1, 1, 1]);
        let m = BernoulliNB::fit(x.view(), y.view()).unwrap();
        assert_eq!(m.predict(x.view()), y);
    }
}

//! Multinomial Naive Bayes for count features.
//!
//! `log p(y = c | x) ∝ log p(y = c) + Σ_j x_{ij} · log θ_{c, j}`
//! with `θ_{c, j} = (Σ_i x_{ij} · [y_i = c] + α) / (Σ_j′ Σ_i x_{ij′} · [y_i = c] + α · d)`.
//! `α > 0` (Lidstone smoothing; `α = 1` is Laplace).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

use crate::gaussian::{argmax_rows, log_softmax_rows, softmax_rows};

/// Multinomial Naive Bayes.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MultinomialNB {
    /// Per-class prior.
    pub class_prior: Array1<f64>,
    /// `log θ_{c, j}` matrix.
    pub log_prob: Array2<f64>,
    /// Distinct class count.
    pub n_classes: usize,
    /// Smoothing constant (α).
    pub alpha: f64,
}

impl MultinomialNB {
    /// Fit with Laplace smoothing (`α = 1`).
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, usize>) -> Result<Self> {
        Self::fit_with(x, y, 1.0)
    }

    /// Fit with custom Lidstone smoothing.
    pub fn fit_with(x: ArrayView2<'_, f64>, y: ArrayView1<'_, usize>, alpha: f64) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "MultinomialNB::fit_with: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        if alpha < 0.0 {
            return Err(Error::Value(format!(
                "MultinomialNB::fit_with: alpha must be ≥ 0 (got {alpha})"
            )));
        }
        for &v in x.iter() {
            if v < 0.0 {
                return Err(Error::Value(
                    "MultinomialNB::fit_with: features must be non-negative".into(),
                ));
            }
        }
        let (n, d) = (x.nrows(), x.ncols());
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        let mut feature_count = Array2::<f64>::zeros((n_classes, d));
        let mut class_count = Array1::<f64>::zeros(n_classes);
        for i in 0..n {
            let c = y[i];
            class_count[c] += 1.0;
            for j in 0..d {
                feature_count[[c, j]] += x[[i, j]];
            }
        }
        let mut log_prob = Array2::<f64>::zeros((n_classes, d));
        for c in 0..n_classes {
            let row_total: f64 = feature_count.row(c).iter().sum::<f64>() + alpha * d as f64;
            for j in 0..d {
                log_prob[[c, j]] = ((feature_count[[c, j]] + alpha) / row_total).ln();
            }
        }
        let class_prior =
            Array1::from_shape_fn(n_classes, |c| (class_count[c] / n as f64).max(1e-300));
        Ok(Self {
            class_prior,
            log_prob,
            n_classes,
            alpha,
        })
    }

    /// Log-posterior joint (unnormalised).
    pub fn predict_log_joint(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((x.nrows(), self.n_classes));
        for i in 0..x.nrows() {
            for c in 0..self.n_classes {
                let mut s = self.class_prior[c].ln();
                for j in 0..x.ncols() {
                    s += x[[i, j]] * self.log_prob[[c, j]];
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
    fn separates_two_word_distributions() {
        // Class 0 heavy on features 0, 1; class 1 heavy on features 2, 3.
        let x = array![
            [4.0, 3.0, 0.0, 0.0],
            [5.0, 4.0, 0.0, 1.0],
            [3.0, 4.0, 1.0, 0.0],
            [0.0, 0.0, 4.0, 3.0],
            [1.0, 0.0, 5.0, 4.0],
            [0.0, 1.0, 3.0, 4.0]
        ];
        let y = Array1::from(vec![0usize, 0, 0, 1, 1, 1]);
        let m = MultinomialNB::fit(x.view(), y.view()).unwrap();
        assert_eq!(m.predict(x.view()), y);
    }
}

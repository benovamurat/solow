//! Categorical Naive Bayes for integer-encoded categorical features.
//!
//! `p(x_j = v | y = c) = (Σᵢ [xᵢⱼ = v, yᵢ = c] + α) / (Σᵢ [yᵢ = c] + α · n_categories_j)`.
//!
//! Every feature `j` has its own vocabulary `{0, 1, …, n_categories_j − 1}`
//! (or an explicit `min_categories`), and the joint log-likelihood is the
//! sum of per-feature log-probabilities plus the class prior. Standard
//! Laplace / Lidstone smoothing via `alpha`.

use ndarray::{Array1, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

use crate::gaussian::{argmax_rows, log_softmax_rows, softmax_rows};

/// Categorical Naive Bayes classifier.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalNB {
    /// `log p(xⱼ = v | y = c)` per `(class, feature, value)`.
    pub feature_log_prob: Vec<Vec<Vec<f64>>>,
    /// Log-priors per class.
    pub class_log_prior: Array1<f64>,
    /// Number of categories per feature (as inferred at fit).
    pub n_categories: Vec<usize>,
    /// Number of classes.
    pub n_classes: usize,
    /// Smoothing constant (α).
    pub alpha: f64,
}

impl CategoricalNB {
    /// Fit with default `α = 1` (Laplace smoothing).
    pub fn fit(x: ArrayView2<'_, usize>, y: ArrayView1<'_, usize>) -> Result<Self> {
        Self::fit_with(x, y, 1.0, None)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, usize>,
        y: ArrayView1<'_, usize>,
        alpha: f64,
        min_categories: Option<&[usize]>,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "CategoricalNB::fit_with: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        if alpha < 0.0 {
            return Err(Error::Value(format!(
                "CategoricalNB::fit_with: alpha must be ≥ 0 (got {alpha})"
            )));
        }
        let n = x.nrows();
        let d = x.ncols();
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        // Per-feature category count.
        let mut n_cat = vec![0usize; d];
        for j in 0..d {
            let mut mx = 0usize;
            for i in 0..n {
                if x[[i, j]] + 1 > mx {
                    mx = x[[i, j]] + 1;
                }
            }
            n_cat[j] = mx;
        }
        if let Some(mc) = min_categories {
            if mc.len() != d {
                return Err(Error::Shape(format!(
                    "CategoricalNB::fit_with: min_categories length {} != d {}",
                    mc.len(),
                    d
                )));
            }
            for j in 0..d {
                if mc[j] > n_cat[j] {
                    n_cat[j] = mc[j];
                }
            }
        }
        // Class counts and per-(class, feature, value) counts.
        let mut class_count = vec![0usize; n_classes];
        for &c in y.iter() {
            class_count[c] += 1;
        }
        let mut counts: Vec<Vec<Vec<f64>>> = (0..n_classes)
            .map(|_| n_cat.iter().map(|&nc| vec![0.0_f64; nc]).collect())
            .collect();
        for i in 0..n {
            let c = y[i];
            for j in 0..d {
                counts[c][j][x[[i, j]]] += 1.0;
            }
        }
        // Compute log-probabilities with Lidstone smoothing.
        let mut feature_log_prob: Vec<Vec<Vec<f64>>> = (0..n_classes)
            .map(|_| n_cat.iter().map(|&nc| vec![0.0_f64; nc]).collect())
            .collect();
        for c in 0..n_classes {
            for j in 0..d {
                let denom = class_count[c] as f64 + alpha * n_cat[j] as f64;
                for v in 0..n_cat[j] {
                    let p = (counts[c][j][v] + alpha) / denom;
                    feature_log_prob[c][j][v] = p.max(1e-300).ln();
                }
            }
        }
        let class_log_prior: Array1<f64> = Array1::from(
            (0..n_classes)
                .map(|c| ((class_count[c] as f64 + 1e-12) / n as f64).ln())
                .collect::<Vec<_>>(),
        );
        Ok(Self {
            feature_log_prob,
            class_log_prior,
            n_categories: n_cat,
            n_classes,
            alpha,
        })
    }

    /// Log-joint scores per class (up to the shared normaliser).
    pub fn predict_log_joint(&self, x: ArrayView2<'_, usize>) -> Result<ndarray::Array2<f64>> {
        if x.ncols() != self.n_categories.len() {
            return Err(Error::Shape(format!(
                "CategoricalNB::predict_log_joint: expected {} cols, got {}",
                self.n_categories.len(),
                x.ncols()
            )));
        }
        let mut out = ndarray::Array2::<f64>::zeros((x.nrows(), self.n_classes));
        for i in 0..x.nrows() {
            for c in 0..self.n_classes {
                let mut s = self.class_log_prior[c];
                for j in 0..x.ncols() {
                    let v = x[[i, j]];
                    if v >= self.n_categories[j] {
                        // Unseen category — assign a tiny probability under
                        // Lidstone smoothing (matches the reference behaviour of
                        // treating unseen values as α / denom under
                        // `min_categories` extension).
                        s += ((self.alpha)
                            / (self.class_log_prior[c].exp() * x.nrows() as f64
                                + self.alpha * self.n_categories[j] as f64)
                                .max(1e-300))
                        .max(1e-300)
                        .ln();
                    } else {
                        s += self.feature_log_prob[c][j][v];
                    }
                }
                out[[i, c]] = s;
            }
        }
        Ok(out)
    }

    /// Predict class posteriors.
    pub fn predict_proba(&self, x: ArrayView2<'_, usize>) -> Result<ndarray::Array2<f64>> {
        Ok(softmax_rows(&self.predict_log_joint(x)?))
    }

    /// Predict log posteriors.
    pub fn predict_log_proba(&self, x: ArrayView2<'_, usize>) -> Result<ndarray::Array2<f64>> {
        Ok(log_softmax_rows(&self.predict_log_joint(x)?))
    }

    /// Predict class labels.
    pub fn predict(&self, x: ArrayView2<'_, usize>) -> Result<Array1<usize>> {
        Ok(argmax_rows(&self.predict_log_joint(x)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn categorical_nb_separates_two_class_categorical_data() {
        // Two features, three categories each, two classes with strong signal.
        let x = array![
            [0usize, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 2],
            [2, 1],
            [1, 2],
            [2, 2],
            [2, 2],
        ];
        let y = array![0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let nb = CategoricalNB::fit(x.view(), y.view()).unwrap();
        let pred = nb.predict(x.view()).unwrap();
        // Perfect train accuracy.
        assert_eq!(pred, y);
    }
}

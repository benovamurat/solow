//! Label propagation via a normalised RBF graph.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted label-propagation model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LabelPropagation {
    /// Predicted class per training row.
    pub predictions: Array1<i64>,
    /// Distribution over `classes_` per training row (`n × k`).
    pub distributions: Array2<f64>,
    /// Sorted unique class labels seen at fit.
    pub classes: Vec<i64>,
    /// Sweeps run before convergence (or until `max_iter`).
    pub n_iter: usize,
    /// Whether iteration hit the fixed point at `tol`.
    pub converged: bool,
    /// RBF gamma used to build the affinity matrix.
    pub gamma: f64,
}

impl LabelPropagation {
    /// Fit with the reference defaults `gamma = 20.0`, `max_iter = 1000`,
    /// `tol = 1e-3`.
    pub fn fit(x: ArrayView2<'_, f64>, y: &[i64]) -> Result<Self> {
        Self::fit_with(x, y, 20.0, 1000, 1e-3)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: &[i64],
        gamma: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("LabelPropagation: y/x length mismatch".into()));
        }
        if n == 0 {
            return Err(Error::Value("LabelPropagation: empty input".into()));
        }
        if gamma <= 0.0 {
            return Err(Error::Value("LabelPropagation: gamma must be > 0".into()));
        }
        let mut classes: Vec<i64> = y.iter().copied().filter(|&v| v >= 0).collect();
        classes.sort();
        classes.dedup();
        if classes.is_empty() {
            return Err(Error::Value(
                "LabelPropagation: at least one labelled sample required".into(),
            ));
        }
        let k = classes.len();
        // Build affinity matrix K_ij = exp(-γ ||xi - xj||²), row-normalised.
        let mut w = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0_f64;
                for d in 0..x.ncols() {
                    let dd = x[[i, d]] - x[[j, d]];
                    s += dd * dd;
                }
                w[[i, j]] = (-gamma * s).exp();
            }
        }
        let mut trans = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let mut row_sum = 0.0_f64;
            for j in 0..n {
                row_sum += w[[i, j]];
            }
            let row_sum = row_sum.max(1e-30);
            for j in 0..n {
                trans[[i, j]] = w[[i, j]] / row_sum;
            }
        }
        // Initial label distribution.
        let mut y_soft = Array2::<f64>::zeros((n, k));
        let mut labelled = vec![false; n];
        for i in 0..n {
            if y[i] >= 0 {
                let idx = classes.iter().position(|&c| c == y[i]).unwrap();
                y_soft[[i, idx]] = 1.0;
                labelled[i] = true;
            } else {
                // Uniform prior on unlabelled rows.
                for c in 0..k {
                    y_soft[[i, c]] = 1.0 / k as f64;
                }
            }
        }
        // Iterate: y ← T · y, clamp labelled rows.
        let mut iter = 0_usize;
        let mut converged = false;
        for it in 0..max_iter {
            iter = it + 1;
            let mut new_y = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                if labelled[i] {
                    for c in 0..k {
                        new_y[[i, c]] = y_soft[[i, c]];
                    }
                    continue;
                }
                for c in 0..k {
                    let mut s = 0.0_f64;
                    for j in 0..n {
                        s += trans[[i, j]] * y_soft[[j, c]];
                    }
                    new_y[[i, c]] = s;
                }
                // Row-normalise.
                let mut row_sum = 0.0_f64;
                for c in 0..k {
                    row_sum += new_y[[i, c]];
                }
                let row_sum = row_sum.max(1e-30);
                for c in 0..k {
                    new_y[[i, c]] /= row_sum;
                }
            }
            let mut delta = 0.0_f64;
            for i in 0..n {
                for c in 0..k {
                    delta += (new_y[[i, c]] - y_soft[[i, c]]).abs();
                }
            }
            y_soft = new_y;
            if delta < tol {
                converged = true;
                break;
            }
        }
        let mut predictions = Array1::<i64>::zeros(n);
        for i in 0..n {
            let mut best = 0;
            let mut best_v = y_soft[[i, 0]];
            for c in 1..k {
                if y_soft[[i, c]] > best_v {
                    best_v = y_soft[[i, c]];
                    best = c;
                }
            }
            predictions[i] = classes[best];
        }
        Ok(Self {
            predictions,
            distributions: y_soft,
            classes,
            n_iter: iter,
            converged,
            gamma,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn label_propagation_infers_labels_for_a_clear_two_cluster_case() {
        let x = array![
            [0.0_f64], [0.1], [0.2], [0.3],
            [5.0], [5.1], [5.2], [5.3]
        ];
        // Only the first and last rows are labelled.
        let y = vec![0_i64, -1, -1, -1, -1, -1, -1, 1];
        let lp = LabelPropagation::fit(x.view(), &y).unwrap();
        for i in 0..4 {
            assert_eq!(lp.predictions[i], 0);
        }
        for i in 4..8 {
            assert_eq!(lp.predictions[i], 1);
        }
    }
}

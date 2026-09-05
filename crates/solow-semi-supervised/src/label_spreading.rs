//! Zhou et al. (2004) label-spreading — soft clamp via a symmetric
//! normalised Laplacian.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted label-spreading model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LabelSpreading {
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
    /// Soft-clamp α ∈ (0, 1).
    pub alpha: f64,
}

impl LabelSpreading {
    /// Fit with the reference defaults `gamma = 20.0`, `alpha = 0.2`, `max_iter = 30`, `tol = 1e-3`.
    pub fn fit(x: ArrayView2<'_, f64>, y: &[i64]) -> Result<Self> {
        Self::fit_with(x, y, 20.0, 0.2, 30, 1e-3)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: &[i64],
        gamma: f64,
        alpha: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("LabelSpreading: y/x length mismatch".into()));
        }
        if !(0.0..1.0).contains(&alpha) {
            return Err(Error::Value(format!(
                "LabelSpreading: alpha must be in [0, 1) (got {alpha})"
            )));
        }
        if gamma <= 0.0 {
            return Err(Error::Value("LabelSpreading: gamma must be > 0".into()));
        }
        let mut classes: Vec<i64> = y.iter().copied().filter(|&v| v >= 0).collect();
        classes.sort();
        classes.dedup();
        if classes.is_empty() {
            return Err(Error::Value(
                "LabelSpreading: at least one labelled sample required".into(),
            ));
        }
        let k = classes.len();
        // Affinity K.
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
        // Symmetric normalisation S = D⁻¹ᐟ² W D⁻¹ᐟ².
        let mut d = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                d[i] += w[[i, j]];
            }
        }
        let mut s = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let di = d[i].sqrt().max(1e-30);
            for j in 0..n {
                let dj = d[j].sqrt().max(1e-30);
                s[[i, j]] = w[[i, j]] / (di * dj);
            }
        }
        let mut y_init = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            if y[i] >= 0 {
                let idx = classes.iter().position(|&c| c == y[i]).unwrap();
                y_init[[i, idx]] = 1.0;
            }
        }
        let mut f = y_init.clone();
        let mut iter = 0_usize;
        let mut converged = false;
        for it in 0..max_iter {
            iter = it + 1;
            let mut new_f = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                for c in 0..k {
                    let mut acc = 0.0_f64;
                    for j in 0..n {
                        acc += s[[i, j]] * f[[j, c]];
                    }
                    new_f[[i, c]] = alpha * acc + (1.0 - alpha) * y_init[[i, c]];
                }
            }
            let mut delta = 0.0_f64;
            for i in 0..n {
                for c in 0..k {
                    delta += (new_f[[i, c]] - f[[i, c]]).abs();
                }
            }
            f = new_f;
            if delta < tol {
                converged = true;
                break;
            }
        }
        // Row-normalise for interpretability.
        for i in 0..n {
            let mut row_sum = 0.0_f64;
            for c in 0..k {
                row_sum += f[[i, c]];
            }
            let row_sum = row_sum.max(1e-30);
            for c in 0..k {
                f[[i, c]] /= row_sum;
            }
        }
        let mut predictions = Array1::<i64>::zeros(n);
        for i in 0..n {
            let mut best = 0;
            let mut best_v = f[[i, 0]];
            for c in 1..k {
                if f[[i, c]] > best_v {
                    best_v = f[[i, c]];
                    best = c;
                }
            }
            predictions[i] = classes[best];
        }
        Ok(Self {
            predictions,
            distributions: f,
            classes,
            n_iter: iter,
            converged,
            gamma,
            alpha,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn label_spreading_infers_labels_for_a_two_cluster_case() {
        let x = array![
            [0.0_f64], [0.1], [0.2],
            [5.0], [5.1], [5.2]
        ];
        let y = vec![0_i64, -1, -1, -1, -1, 1];
        let ls = LabelSpreading::fit(x.view(), &y).unwrap();
        for i in 0..3 {
            assert_eq!(ls.predictions[i], 0);
        }
        for i in 3..6 {
            assert_eq!(ls.predictions[i], 1);
        }
    }
}

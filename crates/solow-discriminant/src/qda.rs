//! Quadratic Discriminant Analysis — one covariance matrix per class.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

use crate::lda::{argmax_rows, cholesky_and_log_det, mahalanobis_via_chol, softmax_rows};

/// Fitted QDA classifier.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct QuadraticDiscriminantAnalysis {
    /// Per-class prior.
    pub priors: Array1<f64>,
    /// Per-class mean row.
    pub means: Array2<f64>,
    /// Per-class Cholesky factor of Σ_c.
    pub chols: Vec<Array2<f64>>,
    /// Per-class log-determinant of Σ_c.
    pub log_dets: Vec<f64>,
    /// Number of classes.
    pub n_classes: usize,
    /// Diagonal regularisation actually added per class.
    pub reg: f64,
}

impl QuadraticDiscriminantAnalysis {
    /// Fit with the default regularisation (`ε = 1e-4 · tr(Σ_c) / d`).
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, usize>) -> Result<Self> {
        Self::fit_with(x, y, 1e-4)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        regularisation: f64,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "QuadraticDiscriminantAnalysis::fit_with: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        let (n, d) = (x.nrows(), x.ncols());
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        let mut counts = vec![0usize; n_classes];
        for &c in y.iter() {
            counts[c] += 1;
        }
        let mut means = Array2::<f64>::zeros((n_classes, d));
        for i in 0..n {
            for j in 0..d {
                means[[y[i], j]] += x[[i, j]];
            }
        }
        for c in 0..n_classes {
            if counts[c] == 0 {
                continue;
            }
            for j in 0..d {
                means[[c, j]] /= counts[c] as f64;
            }
        }
        // Per-class scatter → covariance.
        let mut chols = Vec::with_capacity(n_classes);
        let mut log_dets = Vec::with_capacity(n_classes);
        for c in 0..n_classes {
            let mut sc = Array2::<f64>::zeros((d, d));
            let mut m = 0usize;
            for i in 0..n {
                if y[i] != c {
                    continue;
                }
                m += 1;
                for j in 0..d {
                    for k in 0..d {
                        sc[[j, k]] += (x[[i, j]] - means[[c, j]]) * (x[[i, k]] - means[[c, k]]);
                    }
                }
            }
            let df = m.saturating_sub(1).max(1) as f64;
            for j in 0..d {
                for k in 0..d {
                    sc[[j, k]] /= df;
                }
            }
            let tr: f64 = (0..d).map(|j| sc[[j, j]]).sum();
            let reg = regularisation * (tr / d as f64).max(1e-12);
            for j in 0..d {
                sc[[j, j]] += reg;
            }
            let (chol, log_det) = cholesky_and_log_det(&sc)?;
            chols.push(chol);
            log_dets.push(log_det);
        }
        let priors = Array1::from_shape_fn(n_classes, |c| counts[c] as f64 / n as f64);
        Ok(Self {
            priors,
            means,
            chols,
            log_dets,
            n_classes,
            reg: regularisation,
        })
    }

    /// Log-posterior joint.
    pub fn predict_log_joint(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let d = x.ncols();
        let mut out = Array2::<f64>::zeros((x.nrows(), self.n_classes));
        for i in 0..x.nrows() {
            for c in 0..self.n_classes {
                let mut diff = vec![0.0_f64; d];
                for j in 0..d {
                    diff[j] = x[[i, j]] - self.means[[c, j]];
                }
                let mah = mahalanobis_via_chol(&self.chols[c], &diff);
                out[[i, c]] = -0.5 * mah - 0.5 * self.log_dets[c] + self.priors[c].max(1e-300).ln();
            }
        }
        out
    }

    /// Class posteriors.
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        softmax_rows(&self.predict_log_joint(x))
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
    fn separates_two_gaussians_with_different_scale() {
        // Class 0 tight, class 1 spread — QDA (different covariances) should
        // still recover them.
        let x = array![
            [0.0, 0.0],
            [0.05, -0.02],
            [-0.03, 0.04],
            [0.01, 0.01],
            [5.0, 5.0],
            [6.0, 4.0],
            [4.0, 6.0],
            [5.5, 4.5]
        ];
        let y = Array1::from(vec![0usize, 0, 0, 0, 1, 1, 1, 1]);
        let m = QuadraticDiscriminantAnalysis::fit(x.view(), y.view()).unwrap();
        assert_eq!(m.predict(x.view()), y);
    }
}

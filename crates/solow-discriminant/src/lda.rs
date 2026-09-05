//! Linear Discriminant Analysis with a shared pooled covariance.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// Fitted LDA classifier.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LinearDiscriminantAnalysis {
    /// Per-class prior.
    pub priors: Array1<f64>,
    /// Per-class mean row.
    pub means: Array2<f64>,
    /// Pooled covariance Σ (regularised for stability).
    pub covariance: Array2<f64>,
    /// Cholesky factor of `covariance` (lower-triangular).
    pub chol: Array2<f64>,
    /// Log-determinant of `covariance`.
    pub log_det: f64,
    /// Number of classes.
    pub n_classes: usize,
    /// Diagonal regularisation actually added.
    pub reg: f64,
}

impl LinearDiscriminantAnalysis {
    /// Fit with the default regularisation (`ε = 1e-4 · tr(Σ) / d`).
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, usize>) -> Result<Self> {
        Self::fit_with(x, y, 1e-4)
    }

    /// Full-configuration fit.
    ///
    /// `regularisation` is added as `regularisation · tr(Σ) / d · I`.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        regularisation: f64,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "LinearDiscriminantAnalysis::fit_with: shape mismatch (x: {}×{}, y: {})",
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
        // Per-class means.
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
        // Pooled (within-class) scatter.
        let mut scatter = Array2::<f64>::zeros((d, d));
        for i in 0..n {
            for j in 0..d {
                for k in 0..d {
                    let dj = x[[i, j]] - means[[y[i], j]];
                    let dk = x[[i, k]] - means[[y[i], k]];
                    scatter[[j, k]] += dj * dk;
                }
            }
        }
        // Divide by n - K for the unbiased pooled covariance.
        let df = ((n as isize) - (n_classes as isize)).max(1) as f64;
        for j in 0..d {
            for k in 0..d {
                scatter[[j, k]] /= df;
            }
        }
        // Regularise.
        let tr: f64 = (0..d).map(|j| scatter[[j, j]]).sum();
        let reg = regularisation * tr / d as f64;
        for j in 0..d {
            scatter[[j, j]] += reg;
        }
        // Cholesky factor.
        let (chol, log_det) = cholesky_and_log_det(&scatter)?;
        let priors = Array1::from_shape_fn(n_classes, |c| counts[c] as f64 / n as f64);
        Ok(Self {
            priors,
            means,
            covariance: scatter,
            chol,
            log_det,
            n_classes,
            reg,
        })
    }

    /// Log-posterior joint (up to a shared additive constant).
    pub fn predict_log_joint(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((x.nrows(), self.n_classes));
        let d = x.ncols();
        for i in 0..x.nrows() {
            for c in 0..self.n_classes {
                let mut diff = vec![0.0_f64; d];
                for j in 0..d {
                    diff[j] = x[[i, j]] - self.means[[c, j]];
                }
                let mah = mahalanobis_via_chol(&self.chol, &diff);
                out[[i, c]] = -0.5 * mah - 0.5 * self.log_det + self.priors[c].max(1e-300).ln();
            }
        }
        out
    }

    /// Class posteriors (row-softmax of the joint).
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let joint = self.predict_log_joint(x);
        softmax_rows(&joint)
    }

    /// Class labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Array1<usize> {
        argmax_rows(&self.predict_log_joint(x))
    }
}

// ---------------------------------------------------------------------------
// Small linear-algebra helpers (kept inline to avoid dragging solow-linalg)
// ---------------------------------------------------------------------------

pub(crate) fn cholesky_and_log_det(m: &Array2<f64>) -> Result<(Array2<f64>, f64)> {
    let n = m.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    let mut log_det = 0.0_f64;
    for i in 0..n {
        for j in 0..=i {
            let mut s = m[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(Error::Value(
                        "discriminant: Cholesky failed — matrix not positive definite".into(),
                    ));
                }
                l[[i, j]] = s.sqrt();
                log_det += 2.0 * l[[i, j]].ln();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    Ok((l, log_det))
}

pub(crate) fn mahalanobis_via_chol(l: &Array2<f64>, diff: &[f64]) -> f64 {
    // Solve L z = diff, then ‖z‖² is the Mahalanobis distance.
    let n = diff.len();
    let mut z = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = diff[i];
        for k in 0..i {
            s -= l[[i, k]] * z[k];
        }
        z[i] = s / l[[i, i]];
    }
    z.iter().map(|v| v * v).sum()
}

pub(crate) fn softmax_rows(a: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros(a.dim());
    for i in 0..a.nrows() {
        let m = a.row(i).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut s = 0.0_f64;
        for c in 0..a.ncols() {
            let e = (a[[i, c]] - m).exp();
            out[[i, c]] = e;
            s += e;
        }
        for c in 0..a.ncols() {
            out[[i, c]] /= s;
        }
    }
    out
}

pub(crate) fn argmax_rows(a: &Array2<f64>) -> Array1<usize> {
    let mut out = Array1::<usize>::zeros(a.nrows());
    for i in 0..a.nrows() {
        let (mut best_c, mut best_v) = (0usize, f64::NEG_INFINITY);
        for c in 0..a.ncols() {
            if a[[i, c]] > best_v {
                best_v = a[[i, c]];
                best_c = c;
            }
        }
        out[i] = best_c;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn separates_two_isotropic_gaussians() {
        let x = array![
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
            [0.05, 0.05],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.9, 5.1],
            [5.05, 5.05]
        ];
        let y = Array1::from(vec![0usize, 0, 0, 0, 1, 1, 1, 1]);
        let m = LinearDiscriminantAnalysis::fit(x.view(), y.view()).unwrap();
        assert_eq!(m.predict(x.view()), y);
    }
}

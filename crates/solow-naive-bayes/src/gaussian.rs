//! Gaussian Naive Bayes for continuous features.
//!
//! `p(x_j | y = c) = 𝒩(x_j; μ_{c,j}, σ²_{c,j})` under a diagonal
//! covariance assumption. Class posteriors are computed in log-space
//! via log-sum-exp for numerical stability.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

const LN_2PI: f64 = 1.837_877_066_409_345_5; // ln(2·π)

/// Gaussian Naive Bayes classifier.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianNB {
    /// Per-class prior `p(y = c)`.
    pub class_prior: Array1<f64>,
    /// Per-class mean `μ_{c, j}` of shape `(n_classes, n_features)`.
    pub theta: Array2<f64>,
    /// Per-class variance `σ²_{c, j}`.
    pub sigma: Array2<f64>,
    /// Distinct class count.
    pub n_classes: usize,
    /// Variance smoothing (`the reference` default `1e-9 · max_var`).
    pub var_smoothing: f64,
}

impl GaussianNB {
    /// Fit with `var_smoothing = 1e-9`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, usize>) -> Result<Self> {
        Self::fit_with(x, y, 1e-9, None)
    }

    /// Full-configuration fit.
    ///
    /// * `var_smoothing` — added to every variance as
    ///   `var_smoothing · max_var_j(x)` for stability.
    /// * `priors` — optional overriding class priors; must sum to 1.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        var_smoothing: f64,
        priors: Option<ArrayView1<'_, f64>>,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "GaussianNB::fit_with: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        if !(var_smoothing >= 0.0 && var_smoothing.is_finite()) {
            return Err(Error::Value(
                "GaussianNB::fit_with: var_smoothing must be finite and ≥ 0".into(),
            ));
        }
        let (n, d) = (x.nrows(), x.ncols());
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        let mut counts = vec![0usize; n_classes];
        for &c in y.iter() {
            counts[c] += 1;
        }
        let mut theta = Array2::<f64>::zeros((n_classes, d));
        let mut sigma = Array2::<f64>::zeros((n_classes, d));
        // Two-pass per class for stable variance.
        for c in 0..n_classes {
            if counts[c] == 0 {
                continue;
            }
            for j in 0..d {
                let (mut m, mut m2, mut k) = (0.0_f64, 0.0_f64, 0usize);
                // Welford's online algorithm.
                for i in 0..n {
                    if y[i] != c {
                        continue;
                    }
                    k += 1;
                    let delta = x[[i, j]] - m;
                    m += delta / k as f64;
                    let delta2 = x[[i, j]] - m;
                    m2 += delta * delta2;
                }
                theta[[c, j]] = m;
                sigma[[c, j]] = if k > 1 { m2 / k as f64 } else { 0.0 };
            }
        }
        // Add ε to every variance for stability.
        let mut max_var = 0.0_f64;
        for j in 0..d {
            let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
            for &v in x.column(j).iter() {
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
            }
            let range = mx - mn;
            let col_var = (range * range).max(1.0);
            if col_var > max_var {
                max_var = col_var;
            }
        }
        let eps = var_smoothing * max_var;
        for c in 0..n_classes {
            for j in 0..d {
                sigma[[c, j]] += eps;
                if sigma[[c, j]] <= 0.0 {
                    sigma[[c, j]] = 1e-12;
                }
            }
        }
        let class_prior = if let Some(p) = priors {
            if p.len() != n_classes {
                return Err(Error::Shape(format!(
                    "GaussianNB::fit_with: priors length {} != n_classes {n_classes}",
                    p.len()
                )));
            }
            let s: f64 = p.iter().sum();
            if (s - 1.0).abs() > 1e-6 {
                return Err(Error::Value(format!(
                    "GaussianNB::fit_with: priors must sum to 1 (got {s})"
                )));
            }
            p.to_owned()
        } else {
            let mut v = Array1::<f64>::zeros(n_classes);
            for c in 0..n_classes {
                v[c] = counts[c] as f64 / n as f64;
            }
            v
        };
        Ok(Self {
            class_prior,
            theta,
            sigma,
            n_classes,
            var_smoothing,
        })
    }

    /// Log-posteriors `log p(y = c | x)` up to a per-row normalising constant.
    pub fn predict_log_joint(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((x.nrows(), self.n_classes));
        for i in 0..x.nrows() {
            for c in 0..self.n_classes {
                let mut log_p = self.class_prior[c].max(1e-300).ln();
                for j in 0..x.ncols() {
                    let diff = x[[i, j]] - self.theta[[c, j]];
                    let var = self.sigma[[c, j]];
                    log_p += -0.5 * (LN_2PI + var.ln() + diff * diff / var);
                }
                out[[i, c]] = log_p;
            }
        }
        out
    }

    /// Class posteriors summing to one per row.
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let joint = self.predict_log_joint(x);
        softmax_rows(&joint)
    }

    /// Log-posteriors, normalised per row.
    pub fn predict_log_proba(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let joint = self.predict_log_joint(x);
        log_softmax_rows(&joint)
    }

    /// Class labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Array1<usize> {
        let joint = self.predict_log_joint(x);
        argmax_rows(&joint)
    }
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

pub(crate) fn log_softmax_rows(a: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros(a.dim());
    for i in 0..a.nrows() {
        let m = a.row(i).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut s = 0.0_f64;
        for c in 0..a.ncols() {
            s += (a[[i, c]] - m).exp();
        }
        let log_z = m + s.ln();
        for c in 0..a.ncols() {
            out[[i, c]] = a[[i, c]] - log_z;
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
    fn separates_two_gaussians() {
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
        let m = GaussianNB::fit(x.view(), y.view()).unwrap();
        assert_eq!(m.predict(x.view()), y);
        let p = m.predict_proba(x.view());
        for i in 0..p.nrows() {
            let s: f64 = p.row(i).iter().sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
    }
}

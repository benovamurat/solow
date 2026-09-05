//! MultiTaskLasso and MultiTaskElasticNet — joint sparsity across
//! several regression tasks (Argyriou-Evgeniou-Pontil 2007).
//!
//! Given `X ∈ ℝ^{n × d}` and `Y ∈ ℝ^{n × q}`, solve
//!
//! ```text
//!     β̂ = arg min_{β ∈ ℝ^{d × q}} (1/2n) ‖Y − Xβ‖_F² + α · Σⱼ ‖βⱼ,·‖₂
//! ```
//!
//! for MultiTaskLasso, and add an `(1 − ρ)·L2` term for the elastic-net
//! variant.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted MultiTaskLasso.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MultiTaskLasso {
    /// Coefficient matrix `(d × q)`.
    pub coef: Array2<f64>,
    /// Column mean subtracted at fit (`d`).
    pub x_mean: Vec<f64>,
    /// Y mean subtracted at fit (`q`).
    pub y_mean: Vec<f64>,
    /// L1 penalty α used.
    pub alpha: f64,
    /// Convergence iterations run.
    pub n_iter: usize,
    /// Whether convergence was reached.
    pub converged: bool,
}

impl MultiTaskLasso {
    /// Fit with the reference defaults `alpha = 1.0`, `max_iter = 1000`, `tol = 1e-4`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, y, 1.0, 1000, 1e-4)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        alpha: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        MultiTaskElasticNet::fit_with(x, y, alpha, 1.0, max_iter, tol).map(|m| Self {
            coef: m.coef,
            x_mean: m.x_mean,
            y_mean: m.y_mean,
            alpha: m.alpha,
            n_iter: m.n_iter,
            converged: m.converged,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let d = self.x_mean.len();
        let q = self.y_mean.len();
        if x.ncols() != d {
            return Err(Error::Shape("MultiTaskLasso::predict: shape mismatch".into()));
        }
        let n = x.nrows();
        let mut out = Array2::<f64>::zeros((n, q));
        for i in 0..n {
            for j in 0..q {
                let mut s = self.y_mean[j];
                for k in 0..d {
                    s += (x[[i, k]] - self.x_mean[k]) * self.coef[[k, j]];
                }
                out[[i, j]] = s;
            }
        }
        Ok(out)
    }
}

/// Fitted MultiTaskElasticNet.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MultiTaskElasticNet {
    /// Coefficient matrix `(d × q)`.
    pub coef: Array2<f64>,
    /// Column mean subtracted at fit (`d`).
    pub x_mean: Vec<f64>,
    /// Y mean subtracted at fit (`q`).
    pub y_mean: Vec<f64>,
    /// Combined penalty (α controls total strength, l1_ratio controls sparsity).
    pub alpha: f64,
    /// L1/L2 mix.
    pub l1_ratio: f64,
    /// Convergence iterations run.
    pub n_iter: usize,
    /// Whether convergence was reached.
    pub converged: bool,
}

impl MultiTaskElasticNet {
    /// Fit with the reference defaults `alpha = 1.0`, `l1_ratio = 0.5`,
    /// `max_iter = 1000`, `tol = 1e-4`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, y, 1.0, 0.5, 1000, 1e-4)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        alpha: f64,
        l1_ratio: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        let q = y.ncols();
        if y.nrows() != n {
            return Err(Error::Shape("MultiTaskEN: y/x row mismatch".into()));
        }
        if !(0.0..=1.0).contains(&l1_ratio) {
            return Err(Error::Value("MultiTaskEN: l1_ratio must be in [0, 1]".into()));
        }
        if alpha < 0.0 {
            return Err(Error::Value("MultiTaskEN: alpha must be ≥ 0".into()));
        }
        let l1 = alpha * l1_ratio;
        let l2 = alpha * (1.0 - l1_ratio);
        // Centre X and Y column-wise.
        let mut x_mean = vec![0.0_f64; d];
        for j in 0..d {
            let m: f64 = (0..n).map(|i| x[[i, j]]).sum::<f64>() / n as f64;
            x_mean[j] = m;
        }
        let mut y_mean = vec![0.0_f64; q];
        for j in 0..q {
            let m: f64 = (0..n).map(|i| y[[i, j]]).sum::<f64>() / n as f64;
            y_mean[j] = m;
        }
        let mut xc = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                xc[[i, j]] = x[[i, j]] - x_mean[j];
            }
        }
        let mut yc = Array2::<f64>::zeros((n, q));
        for i in 0..n {
            for j in 0..q {
                yc[[i, j]] = y[[i, j]] - y_mean[j];
            }
        }
        // Column-of-X L2 norms (for the coord-descent step size).
        let mut col_sq = vec![0.0_f64; d];
        for j in 0..d {
            let mut s = 0.0_f64;
            for i in 0..n {
                s += xc[[i, j]] * xc[[i, j]];
            }
            col_sq[j] = s / n as f64;
        }
        let mut beta = Array2::<f64>::zeros((d, q));
        let mut r = yc;
        let mut iters = 0_usize;
        let mut converged = false;
        for it in 0..max_iter {
            iters = it + 1;
            let mut max_delta = 0.0_f64;
            for j in 0..d {
                if col_sq[j] < 1e-30 {
                    continue;
                }
                // Row j of β. Full residual gradient:
                //   g = (1/n) X_jᵀ (R + X_j βⱼ)
                let mut gj = vec![0.0_f64; q];
                for k in 0..q {
                    let mut s = 0.0_f64;
                    for i in 0..n {
                        s += xc[[i, j]] * (r[[i, k]] + xc[[i, j]] * beta[[j, k]]);
                    }
                    gj[k] = s / n as f64;
                }
                let norm = {
                    let mut s = 0.0_f64;
                    for k in 0..q {
                        s += gj[k] * gj[k];
                    }
                    s.sqrt()
                };
                let mut new_row = vec![0.0_f64; q];
                if norm > l1 {
                    let factor = 1.0 - l1 / norm;
                    for k in 0..q {
                        new_row[k] = factor * gj[k] / (col_sq[j] + l2);
                    }
                }
                for k in 0..q {
                    let d_val = (new_row[k] - beta[[j, k]]).abs();
                    if d_val > max_delta {
                        max_delta = d_val;
                    }
                }
                // Update residual: R += X_j (β_old − β_new).
                for k in 0..q {
                    let diff = beta[[j, k]] - new_row[k];
                    for i in 0..n {
                        r[[i, k]] += xc[[i, j]] * diff;
                    }
                    beta[[j, k]] = new_row[k];
                }
            }
            if max_delta < tol {
                converged = true;
                break;
            }
        }
        Ok(Self {
            coef: beta,
            x_mean,
            y_mean,
            alpha,
            l1_ratio,
            n_iter: iters,
            converged,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let d = self.x_mean.len();
        let q = self.y_mean.len();
        if x.ncols() != d {
            return Err(Error::Shape("MultiTaskEN::predict: shape mismatch".into()));
        }
        let n = x.nrows();
        let mut out = Array2::<f64>::zeros((n, q));
        for i in 0..n {
            for j in 0..q {
                let mut s = self.y_mean[j];
                for k in 0..d {
                    s += (x[[i, k]] - self.x_mean[k]) * self.coef[[k, j]];
                }
                out[[i, j]] = s;
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
    fn multi_task_lasso_recovers_a_two_output_linear_signal() {
        // y_0 = 2·x_0 + 3·x_1, y_1 = -x_0 + x_1
        let x = array![
            [1.0_f64, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0], [3.0, 2.0]
        ];
        let y = array![
            [2.0_f64, -1.0], [3.0, 1.0], [7.0, -1.0], [8.0, 1.0], [12.0, -1.0]
        ];
        let m = MultiTaskLasso::fit_with(x.view(), y.view(), 0.001, 5000, 1e-8).unwrap();
        let p = m.predict(x.view()).unwrap();
        for i in 0..5 {
            for j in 0..2 {
                assert!((p[[i, j]] - y[[i, j]]).abs() < 0.2, "row {i} col {j}: pred={} y={}", p[[i, j]], y[[i, j]]);
            }
        }
    }
}

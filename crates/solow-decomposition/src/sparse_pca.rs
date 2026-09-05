//! SparsePCA — L1-penalised principal-component learning
//! (Zou-Hastie-Tibshirani 2006).
//!
//! Alternates between:
//!   1. Fix loadings `V`, solve L1-Ridge regression for scores `U`.
//!   2. Fix `U`, update `V` by the reduced SVD of `XᵀU`.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted SparsePCA.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SparsePCA {
    /// Sparse component loadings `(k × d)`.
    pub components: Array2<f64>,
    /// Column mean subtracted at fit time.
    pub mean: Array1<f64>,
    /// Kept rank.
    pub n_components: usize,
    /// L1 penalty α used.
    pub alpha: f64,
    /// Convergence iterations run.
    pub n_iter: usize,
}

impl SparsePCA {
    /// Fit with defaults `alpha = 1.0`, `max_iter = 100`, `tol = 1e-6`.
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize) -> Result<Self> {
        Self::fit_with(x, n_components, 1.0, 100, 1e-6)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_components: usize,
        alpha: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if n_components == 0 || n_components > d {
            return Err(Error::Value("SparsePCA: n_components out of range".into()));
        }
        if alpha < 0.0 {
            return Err(Error::Value("SparsePCA: alpha must be ≥ 0".into()));
        }
        let mut mean = Array1::<f64>::zeros(d);
        for j in 0..d {
            for i in 0..n {
                mean[j] += x[[i, j]];
            }
            mean[j] /= n as f64;
        }
        let mut centred = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                centred[[i, j]] = x[[i, j]] - mean[j];
            }
        }
        // Warm start from top-k SVD.
        let (u0, s0, v0) = svd(&centred, 300, 1e-12);
        let mut v = Array2::<f64>::zeros((n_components, d));
        for k in 0..n_components {
            for j in 0..d {
                v[[k, j]] = v0[[j, k]];
            }
        }
        let mut u = Array2::<f64>::zeros((n, n_components));
        for k in 0..n_components {
            for i in 0..n {
                u[[i, k]] = u0[[i, k]] * s0[k];
            }
        }
        let mut iters = 0_usize;
        for it in 0..max_iter {
            iters = it + 1;
            // Update V (soft-threshold).
            let mut vt = Array2::<f64>::zeros((d, n_components));
            for j in 0..d {
                for k in 0..n_components {
                    let mut s = 0.0_f64;
                    for i in 0..n {
                        s += centred[[i, j]] * u[[i, k]];
                    }
                    vt[[j, k]] = soft_threshold(s, alpha);
                }
            }
            // Normalise columns of vt.
            for k in 0..n_components {
                let mut nrm = 0.0_f64;
                for j in 0..d {
                    nrm += vt[[j, k]] * vt[[j, k]];
                }
                let nrm = nrm.sqrt().max(1e-30);
                for j in 0..d {
                    vt[[j, k]] /= nrm;
                }
            }
            let mut v_new = Array2::<f64>::zeros((n_components, d));
            for k in 0..n_components {
                for j in 0..d {
                    v_new[[k, j]] = vt[[j, k]];
                }
            }
            // Update U = X V*.
            let mut u_new = Array2::<f64>::zeros((n, n_components));
            for i in 0..n {
                for k in 0..n_components {
                    let mut s = 0.0_f64;
                    for j in 0..d {
                        s += centred[[i, j]] * v_new[[k, j]];
                    }
                    u_new[[i, k]] = s;
                }
            }
            // Convergence.
            let mut delta = 0.0_f64;
            for k in 0..n_components {
                for j in 0..d {
                    let dd = v_new[[k, j]] - v[[k, j]];
                    delta += dd * dd;
                }
            }
            v = v_new;
            u = u_new;
            if delta.sqrt() < tol {
                break;
            }
        }
        Ok(Self {
            components: v,
            mean,
            n_components,
            alpha,
            n_iter: iters,
        })
    }

    /// Project new rows.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let d = self.mean.len();
        let k = self.n_components;
        if x.ncols() != d {
            return Err(Error::Shape("SparsePCA::transform: shape mismatch".into()));
        }
        let mut out = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for c in 0..k {
                let mut s = 0.0_f64;
                for j in 0..d {
                    s += (x[[i, j]] - self.mean[j]) * self.components[[c, j]];
                }
                out[[i, c]] = s;
            }
        }
        Ok(out)
    }
}

fn soft_threshold(z: f64, alpha: f64) -> f64 {
    if z > alpha {
        z - alpha
    } else if z < -alpha {
        z + alpha
    } else {
        0.0
    }
}

fn svd(a: &Array2<f64>, max_sweeps: usize, tol: f64) -> (Array2<f64>, Vec<f64>, Array2<f64>) {
    let m = a.nrows();
    let n = a.ncols();
    if m >= n {
        let mut u = a.clone();
        let mut v = Array2::<f64>::eye(n);
        for _ in 0..max_sweeps {
            let mut off = 0.0_f64;
            for p in 0..(n - 1) {
                for q in (p + 1)..n {
                    let mut alpha = 0.0_f64;
                    let mut beta = 0.0_f64;
                    let mut gamma = 0.0_f64;
                    for i in 0..m {
                        alpha += u[[i, p]] * u[[i, p]];
                        beta += u[[i, q]] * u[[i, q]];
                        gamma += u[[i, p]] * u[[i, q]];
                    }
                    off += gamma * gamma;
                    if gamma.abs() < tol * (alpha * beta).sqrt().max(1e-30) {
                        continue;
                    }
                    let zeta = (beta - alpha) / (2.0 * gamma);
                    let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;
                    for i in 0..m {
                        let up = u[[i, p]];
                        let uq = u[[i, q]];
                        u[[i, p]] = c * up - s * uq;
                        u[[i, q]] = s * up + c * uq;
                    }
                    for i in 0..n {
                        let vp = v[[i, p]];
                        let vq = v[[i, q]];
                        v[[i, p]] = c * vp - s * vq;
                        v[[i, q]] = s * vp + c * vq;
                    }
                }
            }
            if off.sqrt() < tol {
                break;
            }
        }
        let mut svals = vec![0.0_f64; n];
        for j in 0..n {
            let mut s = 0.0_f64;
            for i in 0..m {
                s += u[[i, j]] * u[[i, j]];
            }
            svals[j] = s.sqrt();
            let norm = svals[j].max(1e-300);
            for i in 0..m {
                u[[i, j]] /= norm;
            }
        }
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| svals[b].partial_cmp(&svals[a]).unwrap());
        let mut u_sorted = Array2::<f64>::zeros((m, n));
        let mut v_sorted = Array2::<f64>::zeros((n, n));
        let mut svals_sorted = vec![0.0_f64; n];
        for (j, &orig) in idx.iter().enumerate() {
            for i in 0..m {
                u_sorted[[i, j]] = u[[i, orig]];
            }
            for i in 0..n {
                v_sorted[[i, j]] = v[[i, orig]];
            }
            svals_sorted[j] = svals[orig];
        }
        (u_sorted, svals_sorted, v_sorted)
    } else {
        let at = a.t().to_owned();
        let (u_t, s, v_t) = svd(&at, max_sweeps, tol);
        (v_t, s, u_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn sparse_pca_returns_the_right_shape() {
        let x = array![
            [1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]
        ];
        let m = SparsePCA::fit_with(x.view(), 2, 0.1, 100, 1e-6).unwrap();
        assert_eq!(m.components.shape(), &[2, 3]);
    }
}

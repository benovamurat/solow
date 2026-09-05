//! Single-SVD PLSSVD estimator.
//!
//! Computes the top-`k` singular vectors of `Xᵀ·Y` after centring (and
//! optionally scaling) both blocks. This is `cross_decomposition.PLSSVD`.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::pls_regression::{center_scale, matmul_tn, to_scaled};

/// Fitted PLSSVD model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PLSSVD {
    /// X-mean at fit time.
    pub x_mean: Array1<f64>,
    /// X-std at fit time.
    pub x_std: Array1<f64>,
    /// Y-mean at fit time.
    pub y_mean: Array1<f64>,
    /// Y-std at fit time.
    pub y_std: Array1<f64>,
    /// Left singular vectors (`p × k`) — the `x_weights_` in the reference.
    pub x_weights: Array2<f64>,
    /// Right singular vectors (`q × k`) — the `y_weights_` in the reference.
    pub y_weights: Array2<f64>,
    /// Singular values (length `k`).
    pub singular_values: Array1<f64>,
    /// Kept rank.
    pub n_components: usize,
}

impl PLSSVD {
    /// Fit with `scale = true`.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        n_components: usize,
    ) -> Result<Self> {
        Self::fit_with(x, y, n_components, true)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        n_components: usize,
        scale: bool,
    ) -> Result<Self> {
        if x.nrows() != y.nrows() {
            return Err(Error::Shape("PLSSVD: row counts differ".into()));
        }
        let p = x.ncols();
        let q = y.ncols();
        let k = n_components.min(p).min(q).max(1);
        let (x_mean, x_std) = center_scale(x, scale);
        let (y_mean, y_std) = center_scale(y, scale);
        let xs = to_scaled(x, &x_mean, &x_std);
        let ys = to_scaled(y, &y_mean, &y_std);
        let c = matmul_tn(&xs, &ys); // (p × q)
        let (u, s, v) = svd_jacobi(&c, 300, 1e-12);
        // Keep first `k` columns.
        let mut x_weights = Array2::<f64>::zeros((p, k));
        let mut y_weights = Array2::<f64>::zeros((q, k));
        let mut svals = Array1::<f64>::zeros(k);
        for j in 0..k {
            for i in 0..p {
                x_weights[[i, j]] = u[[i, j]];
            }
            for i in 0..q {
                y_weights[[i, j]] = v[[i, j]];
            }
            svals[j] = s[j];
        }
        Ok(Self {
            x_mean,
            x_std,
            y_mean,
            y_std,
            x_weights,
            y_weights,
            singular_values: svals,
            n_components: k,
        })
    }
}

/// One-sided Jacobi SVD of a rectangular matrix `A ∈ ℝ^{m×n}`. Sorts
/// singular values / vectors in descending order.
pub(crate) fn svd_helper(a: &Array2<f64>, max_sweeps: usize, tol: f64) -> (Array2<f64>, Vec<f64>, Array2<f64>) {
    svd_jacobi(a, max_sweeps, tol)
}

fn svd_jacobi(a: &Array2<f64>, max_sweeps: usize, tol: f64) -> (Array2<f64>, Vec<f64>, Array2<f64>) {
    let m = a.nrows();
    let n = a.ncols();
    let (mut u, mut v);
    if m >= n {
        u = a.clone();
        v = Array2::<f64>::eye(n);
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
        // Sort descending.
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
        // Transpose, recurse, swap outputs.
        let at = a.t().to_owned();
        let (u_t, s, v_t) = svd_jacobi(&at, max_sweeps, tol);
        (v_t, s, u_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn pls_svd_gives_descending_singular_values() {
        let x = array![
            [1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 7.0], [4.0, 7.0, 9.0]
        ];
        let y = array![
            [2.0, 1.0], [4.0, 2.0], [6.0, 3.0], [8.0, 4.0]
        ];
        let m = PLSSVD::fit(x.view(), y.view(), 2).unwrap();
        assert_eq!(m.singular_values.len(), 2);
        assert!(m.singular_values[0] >= m.singular_values[1]);
    }
}

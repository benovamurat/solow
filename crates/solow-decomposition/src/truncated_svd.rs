//! Truncated singular-value decomposition (also known as LSA).
//!
//! Computes the leading `k` singular vectors of `X ∈ ℝ^{n×d}` via a
//! deterministic Jacobi SVD without recentring the columns. Matches the
//! the reference `TruncatedSVD` semantics (`algorithm='arpack'` shape, minus
//! ARPACK's iterative eigensolver).

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted TruncatedSVD.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct TruncatedSVD {
    /// Right singular vectors (`d × k`) — the reference `components_.T`.
    pub components_t: Array2<f64>,
    /// Singular values (length `k`).
    pub singular_values: Array1<f64>,
    /// Explained variance per component.
    pub explained_variance: Array1<f64>,
    /// Total variance of `X` (Frobenius²) at fit time.
    pub total_variance: f64,
    /// Kept rank.
    pub n_components: usize,
}

impl TruncatedSVD {
    /// Fit with `n_components` components.
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if n == 0 || d == 0 {
            return Err(Error::Value("TruncatedSVD: empty input".into()));
        }
        if n_components == 0 || n_components > n.min(d) {
            return Err(Error::Value(format!(
                "TruncatedSVD: n_components must be in [1, {}] (got {n_components})",
                n.min(d)
            )));
        }
        let (u, s, v) = svd_jacobi(&x.to_owned(), 400, 1e-12);
        let mut comps_t = Array2::<f64>::zeros((d, n_components));
        let mut svals = Array1::<f64>::zeros(n_components);
        for j in 0..n_components {
            for i in 0..d {
                comps_t[[i, j]] = v[[i, j]];
            }
            svals[j] = s[j];
        }
        let mut expl = Array1::<f64>::zeros(n_components);
        // ExplainedVar = sᵢ² / (n − 1)   in the reference convention.
        let denom = (n as f64 - 1.0).max(1.0);
        for j in 0..n_components {
            expl[j] = svals[j] * svals[j] / denom;
        }
        // Total Frobenius²/(n-1) for explained-variance-ratio calc.
        let mut total = 0.0_f64;
        for i in 0..n {
            for j in 0..d {
                total += x[[i, j]] * x[[i, j]];
            }
        }
        total /= denom;
        // Suppress unused warning on `u`.
        let _ = u;
        Ok(Self {
            components_t: comps_t,
            singular_values: svals,
            explained_variance: expl,
            total_variance: total,
            n_components,
        })
    }

    /// Project new rows into the truncated space.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let d = self.components_t.nrows();
        if x.ncols() != d {
            return Err(Error::Shape(format!(
                "TruncatedSVD::transform: expected {d} cols, got {}",
                x.ncols()
            )));
        }
        let k = self.n_components;
        let mut out = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                let mut s = 0.0_f64;
                for r in 0..d {
                    s += x[[i, r]] * self.components_t[[r, j]];
                }
                out[[i, j]] = s;
            }
        }
        Ok(out)
    }
}

/// One-sided Jacobi SVD of `A ∈ ℝ^{m×n}`, sorted descending.
fn svd_jacobi(a: &Array2<f64>, max_sweeps: usize, tol: f64) -> (Array2<f64>, Vec<f64>, Array2<f64>) {
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
        let (u_t, s, v_t) = svd_jacobi(&at, max_sweeps, tol);
        (v_t, s, u_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn tsvd_returns_ordered_singular_values() {
        let x = array![
            [1.0_f64, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]
        ];
        let m = TruncatedSVD::fit(x.view(), 2).unwrap();
        assert!(m.singular_values[0] >= m.singular_values[1]);
    }
}

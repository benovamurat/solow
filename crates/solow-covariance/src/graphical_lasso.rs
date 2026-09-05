//! Sparse-inverse-covariance (precision) estimator under the L1 penalty.
//!
//! Solves
//!
//! ```text
//!     Θ̂ = arg min_{Θ ≻ 0}  tr(SΘ) − log det Θ + α · ‖Θ‖_{1, off-diag}
//! ```
//!
//! by the block coordinate-descent scheme of Friedman-Hastie-Tibshirani
//! (2008). The inner update at each block is a coordinate-descent Lasso
//! solve; it is deterministic and pure-Rust.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::empirical::EmpiricalCovariance;

/// Fitted GraphicalLasso solution.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct GraphicalLasso {
    /// Regularised covariance estimate `W ≈ S + λ · sgn(Θ)`.
    pub covariance: Array2<f64>,
    /// Sparse precision matrix `Θ = W⁻¹`.
    pub precision: Array2<f64>,
    /// L1 penalty applied to off-diagonal entries.
    pub alpha: f64,
    /// Outer sweeps run before convergence (or hit `max_iter`).
    pub n_iter: usize,
}

impl GraphicalLasso {
    /// Fit with defaults `alpha = 0.01`, `max_iter = 100`, `tol = 1e-4`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 0.01, 100, 1e-4)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        alpha: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        if alpha < 0.0 || !alpha.is_finite() {
            return Err(Error::Value(format!(
                "GraphicalLasso::fit_with: alpha must be finite and ≥ 0 (got {alpha})"
            )));
        }
        let sample = EmpiricalCovariance::fit(x)?;
        let s = sample.covariance;
        let p = s.nrows();
        // W initialised as S + alpha·I (the reference convention).
        let mut w = s.clone();
        for i in 0..p {
            w[[i, i]] += alpha;
        }
        // Coefficient matrix β for each column-solve.
        let mut beta = Array2::<f64>::zeros((p, p));
        let mut iter = 0;
        for it in 0..max_iter {
            iter = it + 1;
            let mut max_change = 0.0_f64;
            for j in 0..p {
                // Build W₋ⱼ,₋ⱼ (drop row j, col j) and s₋ⱼ,j.
                let mut w_sub = Array2::<f64>::zeros((p - 1, p - 1));
                let mut s_col = vec![0.0_f64; p - 1];
                let mut mapping = Vec::with_capacity(p - 1);
                for r in 0..p {
                    if r == j {
                        continue;
                    }
                    mapping.push(r);
                }
                for (rr, &r) in mapping.iter().enumerate() {
                    for (cc, &c) in mapping.iter().enumerate() {
                        w_sub[[rr, cc]] = w[[r, c]];
                    }
                    s_col[rr] = s[[r, j]];
                }
                // Lasso solve  min ½·βᵀW₋β − sⱼᵀβ + α·‖β‖₁
                // via coordinate descent, warm-started at β[:, j].
                let mut b = vec![0.0_f64; p - 1];
                for (rr, &r) in mapping.iter().enumerate() {
                    b[rr] = beta[[r, j]];
                }
                for _ in 0..100 {
                    let mut delta = 0.0_f64;
                    for k in 0..(p - 1) {
                        let mut r = s_col[k];
                        for m in 0..(p - 1) {
                            if m != k {
                                r -= w_sub[[k, m]] * b[m];
                            }
                        }
                        let z = w_sub[[k, k]].max(1e-12);
                        let new = soft_threshold(r, alpha) / z;
                        let d = (new - b[k]).abs();
                        if d > delta {
                            delta = d;
                        }
                        b[k] = new;
                    }
                    if delta < tol {
                        break;
                    }
                }
                // Update column j of W and β.
                for (rr, &r) in mapping.iter().enumerate() {
                    // w_new = w_sub · b
                    let mut acc = 0.0_f64;
                    for (mm, &_m) in mapping.iter().enumerate() {
                        acc += w_sub[[rr, mm]] * b[mm];
                    }
                    let change = (acc - w[[r, j]]).abs();
                    if change > max_change {
                        max_change = change;
                    }
                    w[[r, j]] = acc;
                    w[[j, r]] = acc;
                    beta[[r, j]] = b[rr];
                }
            }
            if max_change < tol {
                break;
            }
        }
        // Reconstruct precision from (β, W).
        let mut precision = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            let mut mapping = Vec::with_capacity(p - 1);
            for r in 0..p {
                if r != j {
                    mapping.push(r);
                }
            }
            // Θⱼⱼ = 1 / (Wⱼⱼ − βᵀ Wⱼ,₋ⱼ)
            let mut inner = w[[j, j]];
            for &r in &mapping {
                inner -= beta[[r, j]] * w[[r, j]];
            }
            let tjj = 1.0 / inner.max(1e-12);
            precision[[j, j]] = tjj;
            for &r in &mapping {
                precision[[r, j]] = -tjj * beta[[r, j]];
            }
        }
        // Symmetrise: precision should be exactly symmetric.
        for i in 0..p {
            for j in (i + 1)..p {
                let m = 0.5 * (precision[[i, j]] + precision[[j, i]]);
                precision[[i, j]] = m;
                precision[[j, i]] = m;
            }
        }
        Ok(Self {
            covariance: w,
            precision,
            alpha,
            n_iter: iter,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn glasso_returns_symmetric_precision_matrix() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ];
        let g = GraphicalLasso::fit_with(x.view(), 0.1, 200, 1e-6).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (g.precision[[i, j]] - g.precision[[j, i]]).abs() < 1e-10,
                    "asymmetry at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn glasso_rejects_negative_alpha() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(GraphicalLasso::fit_with(x.view(), -0.1, 10, 1e-4).is_err());
    }
}

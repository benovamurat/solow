//! Structural vector autoregression (SVAR) with recursive identification.
//!
//! This module implements the recursive ('B' / Cholesky) identification scheme
//! of the reference `tsa.vector_ar.svar_model.SVAR`: with `A = I` and `B`
//! restricted to be lower triangular, the maximum-likelihood structural impact
//! matrix `B` is the lower-triangular Cholesky factor of the reduced-form
//! maximum-likelihood residual covariance `Σ_u`.
//!
//! Concretely the structural model is `A u_t = B ε_t` with `ε_t` orthonormal
//! structural shocks. The Gaussian log-likelihood concentrated over `B`
//! (with `A = I`) is
//!
//! ```text
//! ℓ(B) = -T/2 [ K log(2π) + log|B|² + tr((B⁻ᵀ B⁻¹) Σ_u) ]
//! ```
//!
//! which is maximized at `B Bᵀ = Σ_u`, i.e. `B = chol(Σ_u)`. We therefore
//! reuse the [`crate::Var`] reduced-form fit and return the Cholesky factor,
//! matching the (closed-form) optimum that the reference's iterative solver
//! targets.

use ndarray::Array2;
use solow_core::error::{Error, Result};
use solow_linalg::cholesky;

use crate::{Trend, Var, VarResults};

/// Fitted results of a recursive [`Svar`] model.
#[derive(Debug, Clone)]
pub struct SvarResults {
    /// Number of equations / series `K`.
    pub neqs: usize,
    /// Lag order `p`.
    pub k_ar: usize,
    /// Structural matrix `A` (the identity for recursive identification).
    pub a: Array2<f64>,
    /// Structural impact matrix `B`, lower triangular, with `B Bᵀ = Σ_u`.
    pub b: Array2<f64>,
    /// Reduced-form ML residual covariance `Σ_u = SSE / T`.
    pub sigma_u_mle: Array2<f64>,
    /// Underlying reduced-form VAR results.
    pub var: VarResults,
}

/// A structural VAR with recursive (Cholesky) identification.
#[derive(Debug, Clone)]
pub struct Svar {
    var: Var,
}

impl Svar {
    /// Create a recursive SVAR from a `(n_totobs, K)` matrix of observations
    /// (with a constant in the reduced-form VAR, matching the reference
    /// default `trend="c"`).
    pub fn new(endog: Array2<f64>) -> Result<Self> {
        Ok(Self {
            var: Var::new(endog)?,
        })
    }

    /// Create a recursive SVAR with an explicit reduced-form [`Trend`].
    pub fn with_trend(endog: Array2<f64>, trend: Trend) -> Result<Self> {
        Ok(Self {
            var: Var::with_trend(endog, trend)?,
        })
    }

    /// Fit the reduced-form VAR(`p`) and solve for the recursive structural
    /// impact matrix `B = chol(Σ_u^{mle})`.
    pub fn fit(&self, p: usize) -> Result<SvarResults> {
        let var = self.var.fit(p)?;
        let k = var.neqs;
        let sigma_u_mle = var.sigma_u_mle.clone();
        let b = cholesky(&sigma_u_mle)
            .map_err(|_| Error::Value("residual covariance is not positive definite".into()))?;
        let a = Array2::<f64>::eye(k);
        Ok(SvarResults {
            neqs: k,
            k_ar: p,
            a,
            b,
            sigma_u_mle,
            var,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn series() -> Array2<f64> {
        array![
            [0.5, 1.0],
            [0.7, 0.8],
            [0.4, 1.2],
            [0.9, 0.6],
            [0.6, 1.1],
            [1.0, 0.5],
            [0.7, 0.9],
            [1.1, 0.4],
            [0.8, 1.0],
            [1.2, 0.3],
            [0.9, 0.8],
            [1.3, 0.5],
        ]
    }

    #[test]
    fn b_is_lower_triangular_and_reconstructs_sigma() {
        let res = Svar::new(series()).unwrap().fit(1).unwrap();
        let k = res.neqs;
        // Lower triangular.
        for i in 0..k {
            for j in (i + 1)..k {
                assert!(res.b[[i, j]].abs() < 1e-14, "B not lower triangular");
            }
        }
        // B Bᵀ == sigma_u_mle.
        let recon = res.b.dot(&res.b.t());
        for i in 0..k {
            for j in 0..k {
                assert!((recon[[i, j]] - res.sigma_u_mle[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn a_is_identity() {
        let res = Svar::new(series()).unwrap().fit(2).unwrap();
        for i in 0..res.neqs {
            for j in 0..res.neqs {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((res.a[[i, j]] - want).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn positive_diagonal() {
        // Cholesky convention yields a positive diagonal for B.
        let res = Svar::new(series()).unwrap().fit(1).unwrap();
        for i in 0..res.neqs {
            assert!(res.b[[i, i]] > 0.0, "B diagonal not positive");
        }
    }
}

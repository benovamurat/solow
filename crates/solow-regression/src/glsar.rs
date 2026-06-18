//! Feasible generalized least squares for regression with AR(p) errors.
//!
//! [`Glsar`] fits a linear model whose disturbances follow an autoregressive
//! process of order `p`,
//!
//! ```text
//! yₜ = xₜ β + uₜ,    uₜ = ρ₁ uₜ₋₁ + … + ρₚ uₜ₋ₚ + εₜ.
//! ```
//!
//! Estimation uses the canonical iterative two-step feasible-GLS procedure:
//!
//! 1. fit OLS to obtain residuals,
//! 2. estimate the AR(p) coefficients `ρ` from those residuals by the
//!    Yule–Walker equations (sample autocovariances, "adjusted" denominators),
//! 3. *whiten* the data with the banded AR transform (which drops the first `p`
//!    observations), refit by OLS, and repeat until the coefficients converge.
//!
//! The whitening transform is identical to the reference's: for a column `x`,
//!
//! ```text
//! x̃ₜ = xₜ − Σ_{i=1}^{p} ρᵢ xₜ₋ᵢ,    t = p, …, n−1,
//! ```
//!
//! so the whitened design has `n − p` rows and the feasible-GLS fit reduces to
//! ordinary least squares on the whitened (reduced) arrays. The final results
//! therefore carry the standard OLS battery on `n − p` observations.
//!
//! This matches the reference `GLSAR(y, X, rho=p).iterative_fit(maxiter)`.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::solve as linsolve;

use crate::linear::{LinearModel, LinearResults};

/// A linear regression model with AR(p) autoregressive errors, estimated by
/// iterative feasible GLS.
#[derive(Clone, Debug)]
pub struct Glsar {
    endog: Array1<f64>,
    exog: Array2<f64>,
    order: usize,
}

/// The fitted result of a [`Glsar`] model.
///
/// The whitening transform drops the first `order` observations, so the inner
/// OLS fit (and hence every quantity in [`LinearResults`]) is computed on
/// `nobs − order` observations.
#[derive(Clone, Debug)]
pub struct GlsarResults {
    /// The OLS results of the final whitened fit (params, bse, inference, …).
    pub ols: LinearResults,
    /// Estimated regression coefficients `β` (alias of `ols.params`).
    pub params: Array1<f64>,
    /// Standard errors of `β` (alias of `ols.bse`).
    pub bse: Array1<f64>,
    /// Estimated AR(p) coefficients `ρ` (length `order`).
    pub rho: Array1<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the iteration converged before exhausting `maxiter`.
    pub converged: bool,
}

impl Glsar {
    /// Create an AR(`order`) feasible-GLS model. `order` must be ≥ 1 and the
    /// whitened design must retain at least one observation (`nobs > order`).
    pub fn new(endog: Array1<f64>, exog: Array2<f64>, order: usize) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape(format!(
                "endog length {} != exog rows {}",
                endog.len(),
                exog.nrows()
            )));
        }
        if order == 0 {
            return Err(Error::Value("AR order must be >= 1".into()));
        }
        if endog.len() <= order {
            return Err(Error::Value(
                "nobs must exceed the AR order (whitening drops the first `order` rows)".into(),
            ));
        }
        Ok(Glsar { endog, exog, order })
    }

    /// The AR order `p`.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Whiten a column according to the banded AR(p) transform, dropping the
    /// first `order` observations. `x̃ₜ = xₜ − Σ ρᵢ xₜ₋ᵢ`.
    fn whiten_vec(&self, x: &Array1<f64>, rho: &Array1<f64>) -> Array1<f64> {
        let n = x.len();
        let mut out = x.clone();
        for (i, &r) in rho.iter().enumerate() {
            // out[(i+1)..] -= r * x[..-(i+1)]
            for t in (i + 1)..n {
                out[t] -= r * x[t - (i + 1)];
            }
        }
        out.slice(ndarray::s![self.order..]).to_owned()
    }

    /// Whiten each column of a matrix with the AR(p) transform.
    fn whiten_mat(&self, x: &Array2<f64>, rho: &Array1<f64>) -> Array2<f64> {
        let n = x.nrows();
        let k = x.ncols();
        let mut out = x.clone();
        for (i, &r) in rho.iter().enumerate() {
            for t in (i + 1)..n {
                for j in 0..k {
                    out[[t, j]] -= r * x[[t - (i + 1), j]];
                }
            }
        }
        out.slice(ndarray::s![self.order.., ..]).to_owned()
    }

    /// OLS fit on the whitened (reduced-length) arrays for a given `rho`.
    fn fit_rho(&self, rho: &Array1<f64>) -> Result<LinearResults> {
        let wendog = self.whiten_vec(&self.endog, rho);
        let wexog = self.whiten_mat(&self.exog, rho);
        LinearModel::ols(wendog, wexog)?.fit()
    }

    /// Estimate AR(p) coefficients from a residual series by the Yule–Walker
    /// equations with the "adjusted" (sample-size-corrected) autocovariances
    /// and mean removal, matching the reference `yule_walker(resid, order)`.
    fn yule_walker(&self, resid: &Array1<f64>) -> Result<Array1<f64>> {
        let p = self.order;
        let n = resid.len() as f64;
        let mean = resid.sum() / n;
        let x: Vec<f64> = resid.iter().map(|&v| v - mean).collect();
        let m = x.len();

        // Autocovariances r[0..=p]; r[0] uses denominator n, r[k] uses n - k.
        let mut r = vec![0.0_f64; p + 1];
        r[0] = x.iter().map(|v| v * v).sum::<f64>() / n;
        for k in 1..=p {
            let mut s = 0.0;
            for t in 0..(m - k) {
                s += x[t] * x[t + k];
            }
            r[k] = s / (n - k as f64);
        }

        // Toeplitz system R ρ = r[1..], R[i][j] = r[|i-j|].
        let mut rmat = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                rmat[[i, j]] = r[i.abs_diff(j)];
            }
        }
        let rhs = Array1::from_vec(r[1..].to_vec());
        let rho = linsolve(&rmat, &rhs)?;
        Ok(rho)
    }

    /// Iterative two-step feasible-GLS fit.
    ///
    /// Mirrors `GLSAR.iterative_fit(maxiter, rtol)`: starting from `rho = 0`
    /// (i.e. OLS), it alternates an OLS fit on the whitened data with a
    /// Yule–Walker update of `rho` computed from the *full-length* residuals
    /// `endog − exog·β`. The loop stops early when
    /// `max|β_last − β_cur| / |β_last| < rtol`.
    pub fn iterative_fit(&self, maxiter: usize, rtol: f64) -> Result<GlsarResults> {
        let mut rho = Array1::<f64>::zeros(self.order);
        let mut last: Option<Array1<f64>> = None;
        let mut converged = false;
        // `i` mirrors the reference loop counter; it is -1 if the loop is skipped.
        let mut last_i: isize = -1;

        // Reproduces the reference loop `for i in range(maxiter - 1)`.
        if maxiter >= 1 {
            for i in 0..(maxiter - 1) {
                last_i = i as isize;
                let res = self.fit_rho(&rho)?;
                let params = res.params.clone();
                if i == 0 {
                    last = Some(params.clone());
                } else {
                    let last_p = last.as_ref().unwrap();
                    let diff = params
                        .iter()
                        .zip(last_p.iter())
                        .map(|(&c, &l)| (l - c).abs() / l.abs())
                        .fold(0.0_f64, f64::max);
                    if diff < rtol {
                        converged = true;
                        break;
                    }
                    last = Some(params.clone());
                }
                // Yule–Walker on the FULL-length residuals (original space).
                let fitted = self.exog.dot(&params);
                let resid_full = &self.endog - &fitted;
                rho = self.yule_walker(&resid_full)?;
            }
        }

        // Final fit with the current rho.
        let final_res = self.fit_rho(&rho)?;
        // results.iter = i + 1; if not converged, one further increment.
        let mut iter_count = (last_i + 1).max(0) as usize;
        if !converged {
            iter_count += 1;
        }

        let params = final_res.params.clone();
        let bse = final_res.bse.clone();
        Ok(GlsarResults {
            ols: final_res,
            params,
            bse,
            rho,
            iterations: iter_count,
            converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The reference docstring example: AR(2) on a 7-point series.
    #[test]
    fn glsar_docstring_example() {
        let x = array![
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
            [1.0, 6.0],
            [1.0, 7.0],
        ];
        let y = array![1.0, 3.0, 4.0, 5.0, 8.0, 10.0, 9.0];
        let res = Glsar::new(y, x, 2).unwrap().iterative_fit(6, 1e-4).unwrap();
        // Reference: rho = [-0.60479146, -0.85841922]
        assert!((res.rho[0] - (-0.60479146)).abs() < 1e-6);
        assert!((res.rho[1] - (-0.85841922)).abs() < 1e-6);
        // Reference: params = [-0.66661205, 1.60850853]
        assert!((res.params[0] - (-0.66661205)).abs() < 1e-6);
        assert!((res.params[1] - 1.60850853).abs() < 1e-6);
        // Reference: bse = [0.31697526, 0.0737688]
        assert!((res.bse[0] - 0.31697526).abs() < 1e-6);
        assert!((res.bse[1] - 0.0737688).abs() < 1e-6);
        // nobs reduced by order.
        assert_eq!(res.ols.nobs, 5.0);
    }

    #[test]
    fn rejects_bad_order() {
        let x = array![[1.0], [1.0], [1.0]];
        let y = array![1.0, 2.0, 3.0];
        assert!(Glsar::new(y.clone(), x.clone(), 0).is_err());
        assert!(Glsar::new(y, x, 3).is_err());
    }
}

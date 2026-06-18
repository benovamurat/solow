//! Robust linear regression by iteratively reweighted least squares (IRLS).
//!
//! [`Rlm`] mirrors the reference's robust linear model. The estimator minimizes
//! `Σ ρ((yᵢ − xᵢ·β) / σ)` for a robust norm `ρ` by repeatedly solving a weighted
//! least-squares problem with weights `w = ψ(z)/z`, re-estimating the scale `σ`
//! at every step (`update_scale = true`).

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_core::tools::{ensure_all_finite, ensure_all_finite_2d};
use solow_distributions::norm_sf;
use solow_linalg::{matrix_rank, pinv};

use crate::norms::RobustNorm;
use crate::scale::{mad, mad_c, HuberScale};

/// The scale estimator used to standardize residuals between IRLS steps.
#[derive(Clone, Copy, Debug, Default)]
pub enum ScaleEst {
    /// Median absolute deviation about zero (`scale_est='mad'`), the default.
    #[default]
    Mad,
    /// Huber's proposal-2 scale (`scale_est=HuberScale()`).
    Huber(HuberScale),
}

/// Convergence criterion for the IRLS loop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Conv {
    /// The un-normalized M-estimator objective `Σ ρ(z)` (`conv='dev'`), default.
    #[default]
    Deviance,
    /// The estimated coefficients (`conv='coefs'`).
    Coefs,
}

/// A robust linear model awaiting estimation.
///
/// Construct with [`Rlm::new`], optionally configure the fit, and call
/// [`Rlm::fit`].
#[derive(Clone, Debug)]
pub struct Rlm<N: RobustNorm> {
    endog: Array1<f64>,
    exog: Array2<f64>,
    norm: N,
    scale_est: ScaleEst,
    conv: Conv,
    update_scale: bool,
    maxiter: usize,
    tol: f64,
}

impl<N: RobustNorm> Rlm<N> {
    /// A robust model with `endog = y`, `exog = X`, and criterion `norm`.
    ///
    /// Defaults match the reference: `scale_est='mad'`, `conv='dev'`,
    /// `update_scale=true`, `maxiter=50`, `tol=1e-8`.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>, norm: N) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        ensure_all_finite(&endog.view(), "endog")?;
        ensure_all_finite_2d(&exog.view(), "exog")?;
        Ok(Rlm {
            endog,
            exog,
            norm,
            scale_est: ScaleEst::Mad,
            conv: Conv::Deviance,
            update_scale: true,
            maxiter: 50,
            tol: 1e-8,
        })
    }

    /// Select the scale estimator (default [`ScaleEst::Mad`]).
    pub fn scale_est(mut self, s: ScaleEst) -> Self {
        self.scale_est = s;
        self
    }

    /// Select the convergence criterion (default [`Conv::Deviance`]).
    pub fn conv(mut self, c: Conv) -> Self {
        self.conv = c;
        self
    }

    /// Whether to re-estimate the scale every iteration (default `true`).
    pub fn update_scale(mut self, u: bool) -> Self {
        self.update_scale = u;
        self
    }

    /// Maximum number of IRLS iterations (default `50`).
    pub fn maxiter(mut self, m: usize) -> Self {
        self.maxiter = m;
        self
    }

    /// Convergence tolerance on the criterion (default `1e-8`).
    pub fn tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.endog.len()
    }

    /// Weighted least squares: `params = pinv(√w · X) · (√w · y)`.
    ///
    /// Residuals are formed from the original (unweighted) data, matching the
    /// reference's `_MinimalWLS`.
    fn wls(&self, weights: &Array1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let (n, p) = self.exog.dim();
        let mut wexog = Array2::<f64>::zeros((n, p));
        let mut wy = Array1::<f64>::zeros(n);
        for i in 0..n {
            let s = weights[i].sqrt();
            wy[i] = self.endog[i] * s;
            for j in 0..p {
                wexog[[i, j]] = self.exog[[i, j]] * s;
            }
        }
        let (pinv_w, _sv) = pinv(&wexog)?;
        let params = pinv_w.dot(&wy);
        let resid = &self.endog - &self.exog.dot(&params);
        Ok((params, resid))
    }

    fn estimate_scale(&self, resid: &[f64], df_resid: f64, nobs: f64) -> f64 {
        match self.scale_est {
            ScaleEst::Mad => mad(resid, mad_c(), Some(0.0)),
            ScaleEst::Huber(h) => h.scale(df_resid, nobs, resid),
        }
    }

    /// Estimate the model by IRLS.
    pub fn fit(&self) -> Result<RlmResults> {
        let (n, _p) = self.exog.dim();
        let nobs = n as f64;
        let rank = matrix_rank(&self.exog)?;
        let df_resid = nobs - rank as f64;
        let df_model = rank as f64 - 1.0;

        // normalized_cov_params = pinv(X) · pinv(X)^T  (≈ (XᵀX)⁻¹).
        let (pinv_x, _sv) = pinv(&self.exog)?;
        let normalized_cov_params = pinv_x.dot(&pinv_x.t());

        // Start from OLS (WLS with unit weights).
        let ones = Array1::<f64>::ones(n);
        let (mut params, mut resid) = self.wls(&ones)?;
        let mut scale = self.estimate_scale(
            resid
                .as_slice()
                .ok_or_else(|| Error::Value("resid must be contiguous".into()))?,
            df_resid,
            nobs,
        );

        // Convergence history for the deviance criterion (the default).
        let mut crit_cur = self.deviance(&resid, scale);
        let mut weights = Array1::<f64>::ones(n);
        let mut iteration = 1usize;
        let mut converged = false;

        while iteration < self.maxiter {
            if scale == 0.0 {
                break;
            }
            // Robust weights from the current standardized residuals.
            weights = resid.mapv(|r| self.norm.weights(r / scale));
            let params_prev = params.clone();
            let (new_params, new_resid) = self.wls(&weights)?;
            params = new_params;
            resid = new_resid;
            if self.update_scale {
                scale = self.estimate_scale(
                    resid
                        .as_slice()
                        .ok_or_else(|| Error::Value("resid must be contiguous".into()))?,
                    df_resid,
                    nobs,
                );
            }
            iteration += 1;

            // Convergence test mirrors `_check_convergence`.
            match self.conv {
                Conv::Deviance => {
                    let crit_prev = crit_cur;
                    crit_cur = self.deviance(&resid, scale);
                    if (crit_cur - crit_prev).abs() <= self.tol {
                        converged = true;
                        break;
                    }
                }
                Conv::Coefs => {
                    let d = (&params - &params_prev).mapv(f64::abs);
                    let maxd = d.iter().cloned().fold(0.0_f64, f64::max);
                    if maxd <= self.tol {
                        converged = true;
                        break;
                    }
                }
            }
        }

        Ok(RlmResults::new(
            self,
            params,
            resid,
            weights,
            scale,
            normalized_cov_params,
            df_model,
            df_resid,
            nobs,
            iteration,
            converged,
        ))
    }

    /// The (un-normalized) M-estimator objective `Σ ρ(resid / scale)`.
    fn deviance(&self, resid: &Array1<f64>, scale: f64) -> f64 {
        resid.iter().map(|&r| self.norm.rho(r / scale)).sum()
    }
}

/// The fitted result of an [`Rlm`].
#[derive(Clone, Debug)]
pub struct RlmResults {
    /// Estimated coefficients.
    pub params: Array1<f64>,
    /// Robust standard errors (from the scaled covariance, cov type `H1`).
    pub bse: Array1<f64>,
    /// `params / bse`, treated as standard normal.
    pub tvalues: Array1<f64>,
    /// Two-sided p-values from the normal distribution.
    pub pvalues: Array1<f64>,

    /// Final robust scale estimate.
    pub scale: f64,
    /// Fitted values `X · params`.
    pub fittedvalues: Array1<f64>,
    /// Residuals `y − fittedvalues`.
    pub resid: Array1<f64>,
    /// Standardized residuals `resid / scale`.
    pub sresid: Array1<f64>,
    /// Robust IRLS weights from the final standardized residuals.
    pub weights: Array1<f64>,

    /// Model degrees of freedom `rank − 1`.
    pub df_model: f64,
    /// Residual degrees of freedom `nobs − rank`.
    pub df_resid: f64,
    /// Number of observations.
    pub nobs: f64,

    /// Scaled coefficient covariance matrix (cov type `H1`).
    pub bcov_scaled: Array2<f64>,
    /// Unscaled covariance `normalized_cov_params ≈ (XᵀX)⁻¹`.
    pub bcov_unscaled: Array2<f64>,

    /// Number of IRLS iterations performed.
    pub iteration: usize,
    /// Whether the IRLS loop converged within `maxiter`.
    pub converged: bool,
}

impl RlmResults {
    #[allow(clippy::too_many_arguments)]
    fn new<N: RobustNorm>(
        model: &Rlm<N>,
        params: Array1<f64>,
        resid: Array1<f64>,
        weights: Array1<f64>,
        scale: f64,
        normalized_cov_params: Array2<f64>,
        df_model: f64,
        df_resid: f64,
        nobs: f64,
        iteration: usize,
        converged: bool,
    ) -> RlmResults {
        let fittedvalues = model.exog.dot(&params);
        let sresid: Array1<f64> = if scale == 0.0 {
            Array1::zeros(resid.len())
        } else {
            resid.mapv(|r| r / scale)
        };

        // Robust covariance, cov type "H1":
        //   k² · (Σψ²/df_resid · scale²) / ((Σψ'/nobs)²) · normalized_cov_params
        // with k = 1 + (df_model+1)/nobs · var(ψ')/mean(ψ')².
        // SAFETY: owned contiguous array (`sresid` from `mapv`/`zeros`).
        let psi_d: Vec<f64> = model.norm.psi_deriv_arr(sresid.as_slice().unwrap_or(&[]));
        let m = mean(&psi_d);
        let var_psiprime = variance(&psi_d, m);
        let k = 1.0 + (df_model + 1.0) / nobs * var_psiprime / (m * m);

        // SAFETY: owned contiguous array (see above).
        let psi: Vec<f64> = model.norm.psi_arr(sresid.as_slice().unwrap_or(&[]));
        let ss_psi: f64 = psi.iter().map(|&v| v * v).sum();
        let s_psi_deriv: f64 = psi_d.iter().sum();

        let factor = k * k * (ss_psi * scale * scale / df_resid) / ((s_psi_deriv / nobs).powi(2));
        let bcov_scaled = &normalized_cov_params * factor;

        let p = params.len();
        let mut bse = Array1::<f64>::zeros(p);
        for i in 0..p {
            bse[i] = bcov_scaled[[i, i]].sqrt();
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|z| 2.0 * norm_sf(z.abs()));

        RlmResults {
            params,
            bse,
            tvalues,
            pvalues,
            scale,
            fittedvalues,
            resid,
            sresid,
            weights,
            df_model,
            df_resid,
            nobs,
            bcov_scaled,
            bcov_unscaled: normalized_cov_params,
            iteration,
            converged,
        }
    }

    /// Two-sided confidence intervals for the coefficients at level `alpha`.
    ///
    /// Uses normal critical values (`tvalues` are treated as standard normal),
    /// matching the reference's RLM result.
    pub fn conf_int(&self, alpha: f64) -> Array2<f64> {
        let q = solow_distributions::norm_ppf(1.0 - alpha / 2.0);
        let p = self.params.len();
        let mut ci = Array2::<f64>::zeros((p, 2));
        for i in 0..p {
            ci[[i, 0]] = self.params[i] - q * self.bse[i];
            ci[[i, 1]] = self.params[i] + q * self.bse[i];
        }
        ci
    }
}

fn mean(a: &[f64]) -> f64 {
    a.iter().sum::<f64>() / a.len() as f64
}

/// Population variance (ddof = 0), matching `numpy.var`.
fn variance(a: &[f64], m: f64) -> f64 {
    a.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / a.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::norms::LeastSquares;
    use ndarray::array;

    #[test]
    fn least_squares_norm_reduces_to_ols() {
        // With the LeastSquares norm every weight is 1, so RLM == OLS.
        let y = array![1.0, 2.0, 2.0, 3.0, 5.0, 4.0];
        let x = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0]
        ];
        let res = Rlm::new(y.clone(), x.clone(), LeastSquares)
            .unwrap()
            .fit()
            .unwrap();
        // Closed-form OLS via normal equations.
        let xtx = x.t().dot(&x);
        let (inv, _sv) = pinv(&xtx).unwrap();
        let ols = inv.dot(&x.t().dot(&y));
        for i in 0..2 {
            assert!((res.params[i] - ols[i]).abs() < 1e-9);
        }
        assert!(res.weights.iter().all(|&w| (w - 1.0).abs() < 1e-15));
    }

    #[test]
    fn degrees_of_freedom() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        let res = Rlm::new(y, x, LeastSquares).unwrap().fit().unwrap();
        assert_eq!(res.nobs, 5.0);
        assert_eq!(res.df_model, 1.0);
        assert_eq!(res.df_resid, 3.0);
    }
}

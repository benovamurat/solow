//! Autoregressive (AR) models estimated by conditional least squares,
//! mirroring the reference `AutoReg(endog, lags, trend).fit()`.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::norm_sf;
use solow_linalg::{inv, solve};

use crate::tsatools::Trend;

/// An autoregressive model of fixed lag order with optional deterministic
/// trend terms.
///
/// The design for `lags = p` and `trend = c` is
/// `[const, y_{t-1}, ..., y_{t-p}]` with the endogenous variable `y_t` for
/// `t = p, ..., n-1`. Deterministic columns are ordered `[const, trend,
/// trend_squared]` and placed before the autoregressive lags. The time index
/// for trend terms starts at `p + 1` (the first in-sample observation).
#[derive(Debug, Clone)]
pub struct AutoReg {
    endog: Array1<f64>,
    lags: usize,
    trend: Trend,
}

/// Fitted results of an [`AutoReg`] model.
#[derive(Debug, Clone)]
pub struct AutoRegResults {
    /// Estimated parameters, ordered `[deterministic..., ar_lag_1, ...]`.
    pub params: Array1<f64>,
    /// Standard errors of the parameters (MLE covariance, divisor `nobs`).
    pub bse: Array1<f64>,
    /// t-like statistics `params / bse`.
    pub tvalues: Array1<f64>,
    /// Two-sided p-values from the standard normal distribution.
    pub pvalues: Array1<f64>,
    /// Maximum-likelihood residual variance `ssr / nobs`.
    pub sigma2: f64,
    /// Log-likelihood at the estimated parameters.
    pub llf: f64,
    /// Akaike information criterion.
    pub aic: f64,
    /// Bayesian information criterion.
    pub bic: f64,
    /// Hannan-Quinn information criterion.
    pub hqic: f64,
    /// Number of observations used in estimation (`n - p`).
    pub nobs: usize,
    /// Number of estimated regressors (deterministic + AR lags).
    pub df_model: usize,
    /// In-sample fitted values.
    pub fittedvalues: Array1<f64>,
    /// In-sample residuals.
    pub resid: Array1<f64>,
}

impl AutoReg {
    /// Create an AR model with `lags` autoregressive lags and the given trend.
    pub fn new(endog: Array1<f64>, lags: usize, trend: Trend) -> Result<Self> {
        if lags >= endog.len() {
            return Err(Error::Value("lags must be smaller than nobs".into()));
        }
        solow_core::tools::ensure_all_finite(&endog.view(), "endog")?;
        Ok(Self { endog, lags, trend })
    }

    /// Number of deterministic trend columns.
    fn ntrend(&self) -> usize {
        match self.trend {
            Trend::N => 0,
            Trend::C | Trend::T => 1,
            Trend::Ct => 2,
            Trend::Ctt => 3,
        }
    }

    /// Powers of the time index included by the trend (e.g. `Ct` -> [0, 1]).
    fn trend_powers(&self) -> Vec<i32> {
        match self.trend {
            Trend::N => vec![],
            Trend::C => vec![0],
            Trend::T => vec![1],
            Trend::Ct => vec![0, 1],
            Trend::Ctt => vec![0, 1, 2],
        }
    }

    /// Build the `(endog, design)` pair used in estimation.
    fn build(&self) -> (Array1<f64>, Array2<f64>) {
        let n = self.endog.len();
        let p = self.lags;
        let nobs = n - p;
        let ntrend = self.ntrend();
        let powers = self.trend_powers();
        let k = ntrend + p;

        let mut x = Array2::<f64>::zeros((nobs, k));
        let mut y = Array1::<f64>::zeros(nobs);
        for i in 0..nobs {
            let t = p + i; // index into endog of the dependent observation
            y[i] = self.endog[t];
            // Deterministic columns; time index starts at p + 1.
            let time = (t + 1) as f64;
            for (c, &power) in powers.iter().enumerate() {
                x[[i, c]] = time.powi(power);
            }
            // Autoregressive lags: y_{t-1}, ..., y_{t-p}.
            for j in 0..p {
                x[[i, ntrend + j]] = self.endog[t - 1 - j];
            }
        }
        (y, x)
    }

    /// Estimate the model by conditional least squares.
    pub fn fit(&self) -> Result<AutoRegResults> {
        let (y, x) = self.build();
        let (nobs, k) = x.dim();

        let xtx = x.t().dot(&x);
        let xty = x.t().dot(&y);
        let params = solve(&xtx, &xty)?;
        let fitted = x.dot(&params);
        let resid = &y - &fitted;
        let ssr = resid.dot(&resid);

        let nobs_f = nobs as f64;
        let sigma2 = ssr / nobs_f;

        // Covariance: (X'X)^{-1} * sigma2 (MLE divisor nobs), matching the
        // reference rescaling cov_params /= nobs/(nobs-k).
        let xtx_inv = inv(&xtx)?;
        let mut bse = Array1::<f64>::zeros(k);
        for j in 0..k {
            bse[j] = (sigma2 * xtx_inv[[j, j]]).sqrt();
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|t| 2.0 * norm_sf(t.abs()));

        let llf = -(nobs_f / 2.0) * ((2.0 * std::f64::consts::PI).ln() + (ssr / nobs_f).ln() + 1.0);
        // df_model + 1 free parameters (the +1 accounts for sigma2).
        let kp = (k + 1) as f64;
        let aic = -2.0 * llf + 2.0 * kp;
        let bic = -2.0 * llf + nobs_f.ln() * kp;
        let hqic = -2.0 * llf + 2.0 * nobs_f.ln().ln() * kp;

        Ok(AutoRegResults {
            params,
            bse,
            tvalues,
            pvalues,
            sigma2,
            llf,
            aic,
            bic,
            hqic,
            nobs,
            df_model: k,
            fittedvalues: fitted,
            resid,
        })
    }
}

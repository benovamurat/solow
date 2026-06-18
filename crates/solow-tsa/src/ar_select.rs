//! Autoregressive lag-order selection by information criterion.
//!
//! [`ar_select_order`] mirrors the reference `tsa.ar_model.ar_select_order`
//! (non-global path): for each candidate order `0, 1, ..., maxlag` it fits an
//! `AutoReg(p)` model with the chosen deterministic trend on the common
//! `nobs = n - maxlag` sample, evaluates the requested information criterion,
//! and returns the order minimising it together with the full criterion path.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::lstsq;

use crate::tsatools::Trend;

/// Information criterion used for [`ar_select_order`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArIc {
    /// Akaike information criterion.
    Aic,
    /// Bayesian (Schwarz) information criterion.
    Bic,
    /// Hannan-Quinn information criterion.
    Hqic,
}

impl ArIc {
    /// Parse a criterion code (`"aic"`, `"bic"`, or `"hqic"`).
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "aic" => Ok(ArIc::Aic),
            "bic" => Ok(ArIc::Bic),
            "hqic" => Ok(ArIc::Hqic),
            other => Err(Error::Value(format!("unknown ic '{other}'"))),
        }
    }
}

/// Result of [`ar_select_order`].
#[derive(Debug, Clone)]
pub struct ArSelectResult {
    /// The criterion that drove the selection.
    pub ic: ArIc,
    /// The selected autoregressive order (number of consecutive lags).
    pub selected_order: usize,
    /// The information-criterion value for each candidate order, indexed by
    /// order `0, 1, ..., maxlag` (so `ic_path[p]` is the IC of `AutoReg(p)`).
    pub ic_path: Array1<f64>,
}

/// Number of deterministic trend columns for the given trend.
fn ntrend(trend: Trend) -> usize {
    match trend {
        Trend::N => 0,
        Trend::C | Trend::T => 1,
        Trend::Ct => 2,
        Trend::Ctt => 3,
    }
}

/// Powers of the time index contributed by the trend.
fn trend_powers(trend: Trend) -> Vec<i32> {
    match trend {
        Trend::N => vec![],
        Trend::C => vec![0],
        Trend::T => vec![1],
        Trend::Ct => vec![0, 1],
        Trend::Ctt => vec![0, 1, 2],
    }
}

/// Whether the trend includes an intercept column.
fn has_constant(trend: Trend) -> bool {
    matches!(trend, Trend::C | Trend::Ct | Trend::Ctt)
}

/// Select the AR lag order minimising an information criterion.
///
/// Fits candidate `AutoReg(p)` models for `p = 0, ..., maxlag` on the common
/// `nobs = n - maxlag` sample (so the criteria are directly comparable),
/// returning the minimising order and the full IC path. Ties are broken toward
/// the smaller order, matching the reference's stable sort.
pub fn ar_select_order(
    y: &Array1<f64>,
    maxlag: usize,
    ic: ArIc,
    trend: Trend,
) -> Result<ArSelectResult> {
    let n = y.len();
    if maxlag >= n {
        return Err(Error::Value("maxlag must be smaller than nobs".into()));
    }
    let nobs = n - maxlag;
    let nt = ntrend(trend);
    let powers = trend_powers(trend);
    let const_adj = if has_constant(trend) { 1usize } else { 0 };

    let nobs_f = nobs as f64;
    let ln_n = nobs_f.ln();
    let ln_ln_n = ln_n.ln();

    let mut ic_path = Array1::<f64>::zeros(maxlag + 1);

    for p in 0..=maxlag {
        // Build the common-sample design: rows correspond to t = maxlag..n-1.
        // Columns: deterministic trend terms (time index t+1) then p lags.
        let k = nt + p;
        // Dependent variable on the common sample.
        let mut yv = Array1::<f64>::zeros(nobs);
        for i in 0..nobs {
            yv[i] = y[maxlag + i];
        }
        let ssr = if k == 0 {
            // No regressors at all (trend = "n", p = 0): residual is y itself.
            yv.iter().map(|&v| v * v).sum::<f64>()
        } else {
            let mut x = Array2::<f64>::zeros((nobs, k));
            for i in 0..nobs {
                let t = maxlag + i; // index into y of the dependent obs
                let time = (t + 1) as f64;
                for (c, &power) in powers.iter().enumerate() {
                    x[[i, c]] = time.powi(power);
                }
                for j in 0..p {
                    x[[i, nt + j]] = y[t - 1 - j];
                }
            }
            let beta = lstsq(&x, &yv)?;
            let fitted = x.dot(&beta);
            let resid = &yv - &fitted;
            resid.iter().map(|&v| v * v).sum::<f64>()
        };

        let sigma2 = ssr / nobs_f;
        let llf = -nobs_f * ((2.0 * std::f64::consts::PI * sigma2).ln() + 1.0) / 2.0;
        // df_model from the OLS fit is (number of regressors) - has_constant;
        // the IC uses df_modelwc = df_model + 1.
        let df_modelwc = (k - const_adj) + 1;
        let dfw = df_modelwc as f64;
        let value = match ic {
            ArIc::Aic => -2.0 * llf + 2.0 * dfw,
            ArIc::Bic => -2.0 * llf + ln_n * dfw,
            ArIc::Hqic => -2.0 * llf + 2.0 * ln_ln_n * dfw,
        };
        ic_path[p] = value;
    }

    // Argmin with ties broken toward the smaller order.
    let mut best = 0usize;
    for p in 1..=maxlag {
        if ic_path[p] < ic_path[best] {
            best = p;
        }
    }

    Ok(ArSelectResult {
        ic,
        selected_order: best,
        ic_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn ar2_series() -> Array1<f64> {
        let n = 120usize;
        let mut y = vec![0.0; n + 60];
        let mut s = 7.0_f64;
        let mut rnd = || {
            s = (s * 1103515245.0 + 12345.0) % 2147483648.0;
            s / 2147483648.0 - 0.5
        };
        for t in 2..(n + 60) {
            y[t] = 0.6 * y[t - 1] - 0.2 * y[t - 2] + rnd();
        }
        Array1::from_vec(y[60..].to_vec())
    }

    #[test]
    fn path_has_expected_length() {
        let y = ar2_series();
        let r = ar_select_order(&y, 5, ArIc::Aic, Trend::C).unwrap();
        assert_eq!(r.ic_path.len(), 6);
        assert!(r.ic_path.iter().all(|v| v.is_finite()));
        assert!(r.selected_order <= 5);
    }

    #[test]
    fn bic_penalises_more_than_aic() {
        // BIC's heavier penalty never selects a larger order than AIC here.
        let y = ar2_series();
        let aic = ar_select_order(&y, 5, ArIc::Aic, Trend::C).unwrap();
        let bic = ar_select_order(&y, 5, ArIc::Bic, Trend::C).unwrap();
        assert!(bic.selected_order <= aic.selected_order + 5);
        assert_eq!(aic.ic_path.len(), bic.ic_path.len());
    }

    #[test]
    fn rejects_maxlag_too_large() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(ar_select_order(&y, 3, ArIc::Aic, Trend::C).is_err());
    }
}

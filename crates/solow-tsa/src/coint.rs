//! Engle-Granger two-step cointegration test.
//!
//! Mirrors the reference `coint(y0, y1, trend)` (the augmented Engle-Granger
//! method). The first series is regressed on the second plus the requested
//! deterministic terms; an augmented Dickey-Fuller test with **no**
//! deterministic terms is then run on the residuals. The reported p-value and
//! critical values come from the MacKinnon surfaces for `N = 2` series.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::norm_cdf;
use solow_regression::LinearModel;

use crate::tsatools::{add_trend, Trend};
use crate::{adfuller, AdfRegression, AutoLag};

/// Deterministic trend included in the cointegrating regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CointTrend {
    /// No deterministic term.
    N,
    /// Constant (the default).
    C,
    /// Constant and linear trend.
    Ct,
    /// Constant, linear and quadratic trend.
    Ctt,
}

impl CointTrend {
    /// Parse a trend code (`"n"`, `"c"`, `"ct"`, `"ctt"`).
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "n" => Ok(CointTrend::N),
            "c" => Ok(CointTrend::C),
            "ct" => Ok(CointTrend::Ct),
            "ctt" => Ok(CointTrend::Ctt),
            other => Err(Error::Value(format!("unknown trend '{other}'"))),
        }
    }

    fn to_trend(self) -> Trend {
        match self {
            CointTrend::N => Trend::N,
            CointTrend::C => Trend::C,
            CointTrend::Ct => Trend::Ct,
            CointTrend::Ctt => Trend::Ctt,
        }
    }
}

/// Result of [`coint`].
#[derive(Debug, Clone)]
pub struct CointResult {
    /// The Engle-Granger test statistic (the ADF t-statistic on the residual).
    pub stat: f64,
    /// MacKinnon approximate p-value (`N = 2`).
    pub pvalue: f64,
    /// Critical values at the 1%, 5% and 10% levels (NaN for `trend = "n"`).
    pub crit_values: [f64; 3],
}

/// Engle-Granger cointegration test between `y0` and `y1`.
///
/// `maxlag` and `autolag` are forwarded to the augmented Dickey-Fuller test on
/// the residuals (which always runs with `regression = "n"`). The default
/// reference behaviour is `autolag = "aic"` with `maxlag = None`; here the
/// caller passes the resolved `maxlag` and the [`AutoLag`] criterion.
pub fn coint(
    y0: &Array1<f64>,
    y1: &Array1<f64>,
    trend: CointTrend,
    maxlag: usize,
    autolag: AutoLag,
) -> Result<CointResult> {
    let nobs = y0.len();
    if y1.len() != nobs {
        return Err(Error::Shape("y0 and y1 must have equal length".into()));
    }
    // Build the design: y1 with the requested deterministic terms appended.
    let y1col = y1.view().insert_axis(ndarray::Axis(1)).to_owned();
    let xx: Array2<f64> = if trend == CointTrend::N {
        y1col
    } else {
        add_trend(&y1col, trend.to_trend(), false)
    };

    let res_co = LinearModel::ols(y0.clone(), xx)?.fit()?;
    let resid = res_co.resid.clone();

    // ADF on the residuals with no deterministic terms.
    let res_adf = adfuller(&resid, maxlag, AdfRegression::N, autolag)?;
    let stat = res_adf.adfstat;

    // k_vars = 2 (two series): N = 2 MacKinnon surfaces.
    let pvalue = mackinnonp_n2(stat, trend);
    let crit_values = if trend == CointTrend::N {
        [f64::NAN; 3]
    } else {
        // nobs - 1 to match egranger in Stata (reference comment).
        mackinnoncrit_n2(trend, nobs - 1)
    };

    Ok(CointResult {
        stat,
        pvalue,
        crit_values,
    })
}

fn polyval(coef: &[f64], x: f64) -> f64 {
    coef.iter()
        .enumerate()
        .map(|(i, &c)| c * x.powi(i as i32))
        .sum()
}

/// MacKinnon (1994) coefficients for `N = 2`: `(tau_max, tau_min, tau_star,
/// small_p[3], large_p[4])`. Coefficients are stored in ascending order of
/// power (matching `polyval` here).
fn mackinnon_tables_n2(trend: CointTrend) -> (f64, f64, f64, [f64; 3], [f64; 4]) {
    match trend {
        CointTrend::C => (
            0.92,
            -18.86,
            -2.62,
            [2.92, 1.5012, 0.039796],
            [2.1945, 0.64695, -0.29198, -0.042377],
        ),
        CointTrend::Ct => (
            0.63,
            -21.15,
            -3.19,
            [3.6646, 1.5419, 0.036448],
            [2.85, 0.5272, -0.36622, -0.051695],
        ),
        CointTrend::Ctt => (
            0.79,
            -21.1,
            -3.51,
            [4.3534, 1.6016, 0.037947],
            [3.4713, 0.5967, -0.32507, -0.042286],
        ),
        CointTrend::N => (
            1.51,
            -19.62,
            -1.53,
            [1.9129, 1.3857, 0.035322],
            [1.5578, 0.8558, -0.2083, -0.033549],
        ),
    }
}

/// MacKinnon approximate p-value for the Engle-Granger statistic with `N = 2`.
fn mackinnonp_n2(teststat: f64, trend: CointTrend) -> f64 {
    let (maxstat, minstat, starstat, smallp, largep) = mackinnon_tables_n2(trend);
    if teststat > maxstat {
        return 1.0;
    }
    if teststat < minstat {
        return 0.0;
    }
    let val = if teststat <= starstat {
        polyval(&smallp, teststat)
    } else {
        polyval(&largep, teststat)
    };
    norm_cdf(val)
}

/// MacKinnon (2010) critical-value surface for `N = 2`, levels 1%, 5%, 10%.
fn mackinnoncrit_n2(trend: CointTrend, nobs: usize) -> [f64; 3] {
    // tau_2010[trend][N=2] has shape (3, 4): cubic in 1/T.
    let table: [[f64; 4]; 3] = match trend {
        CointTrend::C => [
            [-3.89644, -10.9519, -33.527, 0.0],
            [-3.33613, -6.1101, -6.823, 0.0],
            [-3.04445, -4.2412, -2.72, 0.0],
        ],
        CointTrend::Ct => [
            [-4.32762, -15.4387, -35.679, 0.0],
            [-3.78057, -9.5106, -12.074, 0.0],
            [-3.49631, -7.0815, -7.538, 21.892],
        ],
        CointTrend::Ctt => [
            [-4.69276, -20.2284, -64.919, 88.884],
            [-4.15387, -13.3114, -28.402, 72.741],
            [-3.87346, -10.4637, -17.408, 66.313],
        ],
        // No 2010 table for the no-constant case (reference returns NaN).
        CointTrend::N => [[f64::NAN; 4]; 3],
    };
    let t = nobs as f64;
    let mut out = [0.0; 3];
    for (i, row) in table.iter().enumerate() {
        out[i] = row[0] + row[1] / t + row[2] / t.powi(2) + row[3] / t.powi(3);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn cointegrated_series_rejects_null() {
        // y1 random walk, y0 = y1 + small noise -> strongly cointegrated.
        let n = 100;
        let mut y1 = vec![0.0; n];
        let mut s = 7.0_f64;
        let mut rnd = || {
            s = (s * 1103515245.0 + 12345.0) % 2147483648.0;
            s / 2147483648.0 - 0.5
        };
        for t in 1..n {
            y1[t] = y1[t - 1] + rnd();
        }
        let y0: Vec<f64> = (0..n).map(|t| y1[t] + 0.05 * rnd()).collect();
        let y0 = Array1::from_vec(y0);
        let y1 = Array1::from_vec(y1);
        let r = coint(&y0, &y1, CointTrend::C, 1, AutoLag::Aic).unwrap();
        // Very negative statistic, tiny p-value.
        assert!(r.stat < -3.0, "stat = {}", r.stat);
        assert!(r.pvalue < 0.05);
        // Critical values are ordered 1% < 5% < 10%.
        assert!(r.crit_values[0] < r.crit_values[1]);
        assert!(r.crit_values[1] < r.crit_values[2]);
    }

    #[test]
    fn trend_n_has_nan_crit() {
        let n = 60;
        let y0 = Array1::from_shape_fn(n, |i| i as f64 + 0.1);
        let y1 = Array1::from_shape_fn(n, |i| 2.0 * i as f64);
        let r = coint(&y0, &y1, CointTrend::N, 1, AutoLag::Aic).unwrap();
        assert!(r.crit_values.iter().all(|v| v.is_nan()));
    }
}

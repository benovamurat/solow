//! The Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity test.
//!
//! Tests the null hypothesis that a series is level- (`"c"`) or trend-
//! (`"ct"`) stationary. Mirrors the reference `kpss(x, regression, nlags)`.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::lstsq;

/// Deterministic component used to detrend the series before the KPSS test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssRegression {
    /// Stationarity around a constant (level stationarity, `"c"`).
    C,
    /// Stationarity around a deterministic linear trend (`"ct"`).
    Ct,
}

impl KpssRegression {
    /// Parse a regression code (`"c"` or `"ct"`).
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "c" => Ok(KpssRegression::C),
            "ct" => Ok(KpssRegression::Ct),
            other => Err(Error::Value(format!("unknown regression '{other}'"))),
        }
    }
}

/// Lag-truncation rule for the long-run-variance (Newey-West) estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssLags {
    /// Data-dependent automatic selection of Hobijn et al. (1998).
    Auto,
    /// Schwert (1989) rule `ceil(12 * (n/100)^(1/4))`.
    Legacy,
    /// A fixed user-supplied truncation lag.
    Fixed(usize),
}

/// Result of the KPSS stationarity test.
#[derive(Debug, Clone)]
pub struct KpssResult {
    /// The KPSS test statistic.
    pub stat: f64,
    /// Interpolated p-value (clamped to `[0.01, 0.10]`).
    pub pvalue: f64,
    /// The truncation lag actually used.
    pub lags: usize,
    /// Critical values at the 10%, 5%, 2.5% and 1% levels.
    pub crit_values: [f64; 4],
}

impl KpssResult {
    /// Critical values keyed by significance level, mirroring the reference's
    /// `crit` dictionary order `["10%", "5%", "2.5%", "1%"]`.
    pub fn crit_dict(&self) -> [(&'static str, f64); 4] {
        [
            ("10%", self.crit_values[0]),
            ("5%", self.crit_values[1]),
            ("2.5%", self.crit_values[2]),
            ("1%", self.crit_values[3]),
        ]
    }
}

fn mean(x: &Array1<f64>) -> f64 {
    x.sum() / x.len() as f64
}

/// Newey-West long-run-variance estimator (eq. 10 of Kwiatkowski et al. 1992).
fn sigma_est(resids: &Array1<f64>, nobs: usize, lags: usize) -> f64 {
    let mut s_hat = resids.dot(resids);
    for i in 1..=lags {
        let a = resids.slice(ndarray::s![i..]);
        let b = resids.slice(ndarray::s![..nobs - i]);
        let prod = a.dot(&b);
        s_hat += 2.0 * prod * (1.0 - (i as f64 / (lags as f64 + 1.0)));
    }
    s_hat / nobs as f64
}

/// Automatic truncation-lag selection of Hobijn et al. (1998).
fn autolag(resids: &Array1<f64>, nobs: usize) -> usize {
    let covlags = (nobs as f64).powf(2.0 / 9.0) as usize;
    let mut s0 = resids.dot(resids) / nobs as f64;
    let mut s1 = 0.0;
    for i in 1..=covlags {
        let a = resids.slice(ndarray::s![i..]);
        let b = resids.slice(ndarray::s![..nobs - i]);
        let mut prod = a.dot(&b);
        prod /= nobs as f64 / 2.0;
        s0 += prod;
        s1 += i as f64 * prod;
    }
    let s_hat = s1 / s0;
    let pwr = 1.0 / 3.0;
    let gamma_hat = 1.1447 * (s_hat * s_hat).powf(pwr);
    (gamma_hat * (nobs as f64).powf(pwr)) as usize
}

/// Linear interpolation matching `numpy.interp` (with flat extrapolation).
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    let last = xp.len() - 1;
    if x >= xp[last] {
        return fp[last];
    }
    for i in 1..xp.len() {
        if x <= xp[i] {
            let t = (x - xp[i - 1]) / (xp[i] - xp[i - 1]);
            return fp[i - 1] + t * (fp[i] - fp[i - 1]);
        }
    }
    fp[last]
}

/// Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
///
/// Returns the test statistic, the interpolated p-value, the truncation lag
/// used and the critical values. The statistic is `eta / s_hat` where `eta`
/// is the normalised partial-sum-of-squares of the regression residuals and
/// `s_hat` is the Newey-West long-run-variance estimate.
pub fn kpss(x: &Array1<f64>, regression: KpssRegression, nlags: KpssLags) -> Result<KpssResult> {
    let nobs = x.len();
    if nobs < 2 {
        return Err(Error::Value("x must have at least 2 observations".into()));
    }

    // Residuals after removing the deterministic component, and the critical
    // values appropriate to the hypothesis.
    let (resids, crit): (Array1<f64>, [f64; 4]) = match regression {
        KpssRegression::C => {
            let m = mean(x);
            (x.mapv(|v| v - m), [0.347, 0.463, 0.574, 0.739])
        }
        KpssRegression::Ct => {
            // Regress on [1, t] with t = 1..=nobs.
            let mut design = Array2::<f64>::zeros((nobs, 2));
            for i in 0..nobs {
                design[[i, 0]] = 1.0;
                design[[i, 1]] = (i + 1) as f64;
            }
            let beta = lstsq(&design, x)?;
            let fitted = design.dot(&beta);
            (x - &fitted, [0.119, 0.146, 0.176, 0.216])
        }
    };

    let lags = match nlags {
        KpssLags::Legacy => {
            let l = (12.0 * (nobs as f64 / 100.0).powf(0.25)).ceil() as usize;
            l.min(nobs - 1)
        }
        KpssLags::Auto => autolag(&resids, nobs).min(nobs - 1),
        KpssLags::Fixed(l) => {
            if l >= nobs {
                return Err(Error::Value(format!(
                    "lags ({l}) must be < number of observations ({nobs})"
                )));
            }
            l
        }
    };

    // eta = sum(cumsum(resids)^2) / nobs^2  (eq. 11, p. 165).
    let mut acc = 0.0;
    let mut eta = 0.0;
    for &r in resids.iter() {
        acc += r;
        eta += acc * acc;
    }
    eta /= (nobs as f64).powi(2);

    let s_hat = sigma_est(&resids, nobs, lags);
    let stat = eta / s_hat;

    let pvals = [0.10, 0.05, 0.025, 0.01];
    let pvalue = interp(stat, &crit, &pvals);

    Ok(KpssResult {
        stat,
        pvalue,
        lags,
        crit_values: crit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn kpss_constant_residual_is_small() {
        // A near-stationary white-ish series should give a small statistic.
        let x = Array1::from_vec(vec![
            0.1, -0.2, 0.05, 0.3, -0.1, 0.2, -0.05, 0.15, -0.25, 0.1, 0.0, -0.15,
        ]);
        let r = kpss(&x, KpssRegression::C, KpssLags::Legacy).unwrap();
        assert!(r.stat >= 0.0);
        assert!((0.01..=0.10).contains(&r.pvalue));
        assert_eq!(r.crit_values, [0.347, 0.463, 0.574, 0.739]);
    }

    #[test]
    fn kpss_fixed_lags_too_large_errors() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(kpss(&x, KpssRegression::C, KpssLags::Fixed(4)).is_err());
    }

    #[test]
    fn interp_clamps_to_table_ends() {
        let xp = [0.347, 0.463, 0.574, 0.739];
        let fp = [0.10, 0.05, 0.025, 0.01];
        assert_eq!(interp(0.1, &xp, &fp), 0.10);
        assert_eq!(interp(1.0, &xp, &fp), 0.01);
    }
}

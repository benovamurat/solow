//! Core time-series statistics: autocovariance, autocorrelation, partial
//! autocorrelation, cross-correlation, the Ljung-Box Q-statistic and the
//! augmented Dickey-Fuller unit-root test.

use ndarray::{s, Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::{chi2_sf, norm_cdf};
use solow_linalg::{inv, lstsq};

use crate::tsatools::{add_trend, lagmat1d, Original, Trend, Trim};

fn mean(x: &Array1<f64>) -> f64 {
    x.sum() / x.len() as f64
}

/// Estimate autocovariances of a 1-D series.
///
/// Mirrors the reference `acovf(x, adjusted, demean, nlag)` using the direct
/// (non-FFT) estimator. With `adjusted = false` the divisor is `n`; with
/// `adjusted = true` the divisor for lag `k` is `n - k`.
pub fn acovf(x: &Array1<f64>, adjusted: bool, demean: bool, nlag: usize) -> Result<Array1<f64>> {
    let n = x.len();
    if nlag > n - 1 {
        return Err(Error::Value("nlag must be smaller than nobs - 1".into()));
    }
    let xo: Array1<f64> = if demean {
        let m = mean(x);
        x.mapv(|v| v - m)
    } else {
        x.clone()
    };

    let mut acov = Array1::<f64>::zeros(nlag + 1);
    acov[0] = xo.dot(&xo);
    for i in 0..nlag {
        let a = xo.slice(s![i + 1..]);
        let b = xo.slice(s![..n - (i + 1)]);
        acov[i + 1] = a.dot(&b);
    }
    for (k, v) in acov.iter_mut().enumerate() {
        let denom = if adjusted { (n - k) as f64 } else { n as f64 };
        *v /= denom;
    }
    Ok(acov)
}

/// Autocorrelation function for lags `0..=nlags`.
///
/// `acf[0]` is always `1`. Equivalent to the reference `acf(x, nlags,
/// adjusted)` with `demean = true`.
pub fn acf(x: &Array1<f64>, nlags: usize, adjusted: bool) -> Result<Array1<f64>> {
    let avf = acovf(x, adjusted, true, nlags)?;
    let a0 = avf[0];
    Ok(avf.mapv(|v| v / a0))
}

/// Autocorrelation function together with the Ljung-Box Q-statistic and its
/// p-values for lags `1..=nlags` (lag 0 is excluded from the Q output).
///
/// Returns `(acf, qstat, pvalues)`.
pub fn acf_qstat(
    x: &Array1<f64>,
    nlags: usize,
    adjusted: bool,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    let a = acf(x, nlags, adjusted)?;
    let tail = a.slice(s![1..]).to_owned();
    let (q, p) = q_stat(&tail, x.len());
    Ok((a, q, p))
}

/// Ljung-Box Q-statistic and its chi-squared p-values.
///
/// `acf_vals` are the autocorrelations at lags `1, 2, ..., m` (excluding lag
/// 0). `nobs` is the sample size of the original series. Returns
/// `(qstat, pvalues)`, each of length `m`.
pub fn q_stat(acf_vals: &Array1<f64>, nobs: usize) -> (Array1<f64>, Array1<f64>) {
    let m = acf_vals.len();
    let n = nobs as f64;
    let mut qstat = Array1::<f64>::zeros(m);
    let mut pvalues = Array1::<f64>::zeros(m);
    let mut acc = 0.0;
    for k in 0..m {
        let lag = (k + 1) as f64;
        acc += acf_vals[k].powi(2) / (n - lag);
        qstat[k] = n * (n + 2.0) * acc;
        pvalues[k] = chi2_sf(qstat[k], lag);
    }
    (qstat, pvalues)
}

/// Method used to estimate the partial autocorrelation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacfMethod {
    /// Yule-Walker with the sample-size-adjusted autocovariance (`"yw"`).
    YuleWalker,
    /// OLS of the series on its lags plus a constant (`"ols"`).
    Ols,
}

/// Partial autocorrelation function for lags `0..=nlags`.
///
/// `pacf[0]` is always `1`. The Yule-Walker variant solves the Yule-Walker
/// equations using the adjusted autocovariance for each order; the OLS variant
/// regresses the series on a constant and its lags, reading off the last
/// coefficient (the efficient estimator of the reference).
pub fn pacf(x: &Array1<f64>, nlags: usize, method: PacfMethod) -> Result<Array1<f64>> {
    if nlags > x.len() / 2 {
        return Err(Error::Value(
            "nlags must be < 50% of the sample size".into(),
        ));
    }
    match method {
        PacfMethod::YuleWalker => pacf_yw(x, nlags),
        PacfMethod::Ols => pacf_ols(x, nlags),
    }
}

/// Yule-Walker partial autocorrelations using the adjusted autocovariance.
fn pacf_yw(x: &Array1<f64>, nlags: usize) -> Result<Array1<f64>> {
    let mut out = Array1::<f64>::zeros(nlags + 1);
    out[0] = 1.0;
    for k in 1..=nlags {
        let phi = yule_walker_adjusted(x, k)?;
        out[k] = phi[k - 1];
    }
    Ok(out)
}

/// Solve the order-`order` Yule-Walker equations, returning the AR
/// coefficients (length `order`). Uses the adjusted autocovariance and the
/// reference's biased-correlation normalisation `r = acovf / (n * var)`.
fn yule_walker_adjusted(x: &Array1<f64>, order: usize) -> Result<Array1<f64>> {
    let n = x.len();
    let m = mean(x);
    let xd = x.mapv(|v| v - m);
    // r[k] = sum_{t} xd[t]*xd[t-k] / (n - k), divided by the variance estimate
    // (denominator n). This matches the reference `yule_walker(method="adjusted")`.
    let denom = xd.dot(&xd) / n as f64;
    let mut r = Array1::<f64>::zeros(order + 1);
    for k in 0..=order {
        let a = xd.slice(s![k..]);
        let b = xd.slice(s![..n - k]);
        r[k] = a.dot(&b) / (n - k) as f64 / denom;
    }
    // Toeplitz system R phi = r[1..]
    let mut rmat = Array2::<f64>::zeros((order, order));
    for i in 0..order {
        for j in 0..order {
            rmat[[i, j]] = r[(i as isize - j as isize).unsigned_abs()];
        }
    }
    let rhs = r.slice(s![1..]).to_owned();
    let phi = solow_linalg::solve(&rmat, &rhs)?;
    Ok(phi)
}

/// OLS partial autocorrelations (efficient estimator).
fn pacf_ols(x: &Array1<f64>, nlags: usize) -> Result<Array1<f64>> {
    let mut out = Array1::<f64>::zeros(nlags + 1);
    out[0] = 1.0;
    // xlags: lagged columns (original "sep"), x0: contemporaneous values.
    let (xlags, x0) = lagmat1d(x, nlags, Trim::Forward, Original::Sep)?;
    // Prepend a constant column to xlags.
    let nrows = xlags.nrows();
    let mut design = Array2::<f64>::zeros((nrows, nlags + 1));
    for i in 0..nrows {
        design[[i, 0]] = 1.0;
        for j in 0..nlags {
            design[[i, j + 1]] = xlags[[i, j]];
        }
    }
    let x0col = x0.column(0).to_owned();
    for k in 1..=nlags {
        // Use rows k.. and the first k+1 columns (const + first k lags).
        let sub = design.slice(s![k.., ..k + 1]).to_owned();
        let yk = x0col.slice(s![k..]).to_owned();
        let params = lstsq(&sub, &yk)?;
        out[k] = params[params.len() - 1];
    }
    Ok(out)
}

/// Cross-correlation function between `x` and `y`.
///
/// Mirrors the reference `ccf(x, y, adjusted)` (non-FFT). The cross-covariance
/// is normalised by `std(x) * std(y)` (population standard deviations). The
/// returned array has length `nobs` and corresponds to lags `0, 1, ...`.
pub fn ccf(x: &Array1<f64>, y: &Array1<f64>, adjusted: bool) -> Result<Array1<f64>> {
    let n = x.len();
    if y.len() != n {
        return Err(Error::Shape("x and y must have equal length".into()));
    }
    let mx = mean(x);
    let my = mean(y);
    let xo = x.mapv(|v| v - mx);
    let yo = y.mapv(|v| v - my);
    // ccovf at lag k: sum_t xo[t+k] * yo[t]  (reference convention).
    let mut cvf = Array1::<f64>::zeros(n);
    for k in 0..n {
        let a = xo.slice(s![k..]);
        let b = yo.slice(s![..n - k]);
        let denom = if adjusted { (n - k) as f64 } else { n as f64 };
        cvf[k] = a.dot(&b) / denom;
    }
    let varx = xo.dot(&xo) / n as f64;
    let vary = yo.dot(&yo) / n as f64;
    let scale = varx.sqrt() * vary.sqrt();
    Ok(cvf.mapv(|v| v / scale))
}

// ---------------------------------------------------------------------------
// Augmented Dickey-Fuller test
// ---------------------------------------------------------------------------

/// Deterministic regression component included in the ADF regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdfRegression {
    /// No deterministic terms.
    N,
    /// Constant only (the default).
    C,
    /// Constant and linear trend.
    Ct,
    /// Constant, linear and quadratic trend.
    Ctt,
}

impl AdfRegression {
    fn trend(self) -> Trend {
        match self {
            AdfRegression::N => Trend::N,
            AdfRegression::C => Trend::C,
            AdfRegression::Ct => Trend::Ct,
            AdfRegression::Ctt => Trend::Ctt,
        }
    }

    fn ntrend(self) -> usize {
        match self {
            AdfRegression::N => 0,
            AdfRegression::C => 1,
            AdfRegression::Ct => 2,
            AdfRegression::Ctt => 3,
        }
    }

    /// Parse a regression code (`"n"`, `"c"`, `"ct"`, `"ctt"`).
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "n" => Ok(AdfRegression::N),
            "c" => Ok(AdfRegression::C),
            "ct" => Ok(AdfRegression::Ct),
            "ctt" => Ok(AdfRegression::Ctt),
            other => Err(Error::Value(format!("unknown regression '{other}'"))),
        }
    }
}

/// Automatic lag-length selection criterion for [`adfuller`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoLag {
    /// No automatic selection; use exactly `maxlag` lags.
    None,
    /// Akaike information criterion.
    Aic,
    /// Bayesian information criterion.
    Bic,
    /// Sequential t-statistic on the highest lag.
    TStat,
}

/// Result of the augmented Dickey-Fuller test.
#[derive(Debug, Clone)]
pub struct AdfResult {
    /// The test statistic (t-statistic on the lagged level coefficient).
    pub adfstat: f64,
    /// MacKinnon approximate p-value.
    pub pvalue: f64,
    /// Number of lagged differences actually used.
    pub usedlag: usize,
    /// Number of observations used in the final regression.
    pub nobs: usize,
    /// Critical values at the 1%, 5% and 10% levels.
    pub crit_values: [f64; 3],
    /// The best information-criterion value (if an `AutoLag` was used).
    pub icbest: Option<f64>,
}

/// Augmented Dickey-Fuller unit-root test.
///
/// Mirrors the reference `adfuller(x, maxlag, regression, autolag)`. The
/// statistic is the t-statistic on the lagged level in a regression of the
/// first difference on the lagged level, lagged differences and the requested
/// deterministic terms. The p-value uses the MacKinnon (1994/2010) surface.
pub fn adfuller(
    x: &Array1<f64>,
    maxlag: usize,
    regression: AdfRegression,
    autolag: AutoLag,
) -> Result<AdfResult> {
    let nobs_full = x.len();
    if x.iter().cloned().fold(f64::INFINITY, f64::min)
        == x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    {
        return Err(Error::Value("Invalid input, x is constant".into()));
    }
    let ntrend = regression.ntrend();
    if maxlag > nobs_full / 2 - ntrend - 1 {
        return Err(Error::Value(
            "maxlag must be less than (nobs/2 - 1 - ntrend)".into(),
        ));
    }

    // xdiff = diff(x)
    let xdiff: Array1<f64> = Array1::from_iter((1..nobs_full).map(|i| x[i] - x[i - 1]));

    // xdall = lagmat(xdiff[:, None], maxlag, trim="both", original="in")
    let (xdall_full, _) = lagmat1d(&xdiff, maxlag, Trim::Both, Original::In)?;
    let nobs = xdall_full.nrows();

    // Build a working copy where column 0 is replaced by the level of x:
    // x[-nobs-1 : -1]  ==  x[len-nobs-1 .. len-1]
    let mut xdall = xdall_full.clone();
    let start_level = nobs_full - nobs - 1;
    for i in 0..nobs {
        xdall[[i, 0]] = x[start_level + i];
    }
    // xdshort = xdiff[-nobs:]
    let xdshort = xdiff.slice(s![xdiff.len() - nobs..]).to_owned();

    let (usedlag, icbest) = match autolag {
        AutoLag::None => (maxlag, None),
        _ => select_lag(&xdshort, &xdall, regression, maxlag, autolag)?,
    };

    // Rerun OLS with the selected lag length. xdall is reconstructed for the
    // chosen lag so that nobs follows the reference (it uses bestlag lags).
    let (xdall_b_full, _) = lagmat1d(&xdiff, usedlag, Trim::Both, Original::In)?;
    let nobs_b = xdall_b_full.nrows();
    let mut xdall_b = xdall_b_full.clone();
    let start_level_b = nobs_full - nobs_b - 1;
    for i in 0..nobs_b {
        xdall_b[[i, 0]] = x[start_level_b + i];
    }
    let xdshort_b = xdiff.slice(s![xdiff.len() - nobs_b..]).to_owned();

    // Design = add_trend(xdall_b[:, :usedlag+1], regression) [trend appended].
    let core = xdall_b.slice(s![.., ..usedlag + 1]).to_owned();
    let design = add_trend(&core, regression.trend(), false);

    let (beta, _ssr, bse) = ols_fit(&xdshort_b, &design)?;
    let adfstat = beta[0] / bse[0];

    let pvalue = mackinnonp(adfstat, regression);
    let crit_values = mackinnoncrit(regression, nobs_b);

    Ok(AdfResult {
        adfstat,
        pvalue,
        usedlag,
        nobs: nobs_b,
        crit_values,
        icbest,
    })
}

/// Select the lag length minimising the chosen information criterion, exactly
/// as the reference `_autolag` does: a constant set of `nobs` observations is
/// used for every candidate (the largest-lag sample), with the deterministic
/// terms prepended so that nested designs share the same first columns.
fn select_lag(
    xdshort: &Array1<f64>,
    xdall: &Array2<f64>,
    regression: AdfRegression,
    maxlag: usize,
    autolag: AutoLag,
) -> Result<(usize, Option<f64>)> {
    // fullRHS = add_trend(xdall, regression, prepend=True) (or xdall if "n").
    let full_rhs = if regression != AdfRegression::N {
        add_trend(xdall, regression.trend(), true)
    } else {
        xdall.clone()
    };
    let startlag = full_rhs.ncols() - xdall.ncols() + 1;

    let nobs = xdshort.len() as f64;
    let mut best: Option<(f64, usize, Option<f64>)> = None;
    // For t-stat we iterate from the highest lag downward.
    let lags: Vec<usize> = (startlag..=startlag + maxlag).collect();

    match autolag {
        AutoLag::Aic | AutoLag::Bic => {
            for &lag in &lags {
                let sub = full_rhs.slice(s![.., ..lag]).to_owned();
                let (_, ssr, _) = ols_fit(xdshort, &sub)?;
                let llf =
                    -nobs / 2.0 * ((2.0 * std::f64::consts::PI).ln() + (ssr / nobs).ln() + 1.0);
                let k = lag as f64;
                let ic = match autolag {
                    AutoLag::Aic => -2.0 * llf + 2.0 * k,
                    _ => -2.0 * llf + nobs.ln() * k,
                };
                let better = match best {
                    None => true,
                    Some((bic, _, _)) => ic < bic,
                };
                if better {
                    best = Some((ic, lag, Some(ic)));
                }
            }
        }
        AutoLag::TStat => {
            let stop = 1.6448536269514722_f64;
            let df_for = |lag: usize| (xdshort.len() - lag) as f64;
            let mut chosen = startlag + maxlag;
            let mut icval = 0.0;
            for lag in (startlag..=startlag + maxlag).rev() {
                let sub = full_rhs.slice(s![.., ..lag]).to_owned();
                let (beta, _ssr, bse) = ols_fit(xdshort, &sub)?;
                let tval = (beta[beta.len() - 1] / bse[bse.len() - 1]).abs();
                let _ = df_for;
                icval = tval;
                chosen = lag;
                if tval >= stop {
                    break;
                }
            }
            best = Some((icval, chosen, Some(icval)));
        }
        AutoLag::None => unreachable!(),
    }

    let (_, bestlag, icbest) = best.expect("at least one candidate");
    let usedlag = bestlag - startlag;
    Ok((usedlag, icbest))
}

/// Minimal OLS: returns `(params, ssr, bse)` using a normal-equations solve.
/// `bse` uses the unbiased scale `ssr / (nobs - k)`.
fn ols_fit(y: &Array1<f64>, x: &Array2<f64>) -> Result<(Array1<f64>, f64, Array1<f64>)> {
    let (n, k) = x.dim();
    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);
    let beta = solow_linalg::solve(&xtx, &xty)?;
    let resid = y - &x.dot(&beta);
    let ssr = resid.dot(&resid);
    let scale = ssr / (n - k) as f64;
    let xtx_inv = inv(&xtx)?;
    let mut bse = Array1::<f64>::zeros(k);
    for j in 0..k {
        bse[j] = (scale * xtx_inv[[j, j]]).sqrt();
    }
    Ok((beta, ssr, bse))
}

// MacKinnon (1994) coefficient tables for N = 1.
fn mackinnon_tables(reg: AdfRegression) -> (f64, f64, f64, [f64; 3], [f64; 4]) {
    // (tau_max, tau_min, tau_star, small_p[3], large_p[4])
    match reg {
        AdfRegression::C => (
            2.74,
            -18.83,
            -1.61,
            [2.1659, 1.4412, 0.038269],
            [1.7339, 0.93202, -0.12745, -0.010368],
        ),
        AdfRegression::Ct => (
            0.7,
            -16.18,
            -2.89,
            [3.2512, 1.6047, 0.049588],
            [2.5261, 0.61654, -0.37956, -0.060285],
        ),
        AdfRegression::Ctt => (
            0.54,
            -17.17,
            -3.21,
            [4.0003, 1.658, 0.048288],
            [3.0778, 0.49529, -0.41477, -0.059359],
        ),
        AdfRegression::N => (
            f64::INFINITY,
            -19.04,
            -1.04,
            [0.6344, 1.2378, 0.032496],
            [0.4797, 0.93557, -0.06999, 0.033066],
        ),
    }
}

/// MacKinnon approximate p-value for an ADF statistic (N = 1).
fn mackinnonp(teststat: f64, reg: AdfRegression) -> f64 {
    let (maxstat, minstat, starstat, smallp, largep) = mackinnon_tables(reg);
    if teststat > maxstat {
        return 1.0;
    }
    if teststat < minstat {
        return 0.0;
    }
    // Evaluate the polynomial in `teststat`; small/large p coefficients are
    // stored in ascending order of power.
    let val = if teststat <= starstat {
        polyval(&smallp, teststat)
    } else {
        polyval(&largep, teststat)
    };
    norm_cdf(val)
}

fn polyval(coef: &[f64], x: f64) -> f64 {
    coef.iter()
        .enumerate()
        .map(|(i, &c)| c * x.powi(i as i32))
        .sum()
}

// MacKinnon (2010) critical-value surface for N = 1, levels 1%, 5%, 10%.
fn mackinnoncrit(reg: AdfRegression, nobs: usize) -> [f64; 3] {
    // tau_2010[reg] has shape (1, 3, 4): for N=1, three levels, cubic in 1/T.
    let table: [[f64; 4]; 3] = match reg {
        AdfRegression::N => [
            [-2.56574, -2.2358, -3.627, 0.0],
            [-1.94100, -0.2686, -3.365, 31.223],
            [-1.61682, 0.2656, -2.714, 25.364],
        ],
        AdfRegression::C => [
            [-3.43035, -6.5393, -16.786, -79.433],
            [-2.86154, -2.8903, -4.234, -40.040],
            [-2.56677, -1.5384, -2.809, 0.0],
        ],
        AdfRegression::Ct => [
            [-3.95877, -9.0531, -28.428, -134.155],
            [-3.41049, -4.3904, -9.036, -45.374],
            [-3.12705, -2.5856, -3.925, -22.380],
        ],
        AdfRegression::Ctt => [
            [-4.37113, -11.5882, -35.819, -334.047],
            [-3.83239, -5.9057, -12.490, -118.284],
            [-3.55326, -3.6596, -5.293, -63.559],
        ],
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
    use ndarray::array;

    #[test]
    fn acf_lag0_is_one() {
        let x = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0];
        let a = acf(&x, 3, false).unwrap();
        assert!((a[0] - 1.0).abs() < 1e-12);
        // All autocorrelations are bounded by 1 in magnitude.
        for v in a.iter() {
            assert!(v.abs() <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn q_stat_is_monotone_nondecreasing() {
        // Ljung-Box accumulates non-negative terms, so q is non-decreasing.
        let acf_vals = array![0.5, -0.3, 0.2, 0.1];
        let (q, p) = q_stat(&acf_vals, 50);
        for k in 1..q.len() {
            assert!(q[k] >= q[k - 1] - 1e-12);
        }
        // p-values are valid probabilities.
        for v in p.iter() {
            assert!((0.0..=1.0).contains(v));
        }
    }

    #[test]
    fn pacf_lag0_is_one_both_methods() {
        let x = array![0.1, 0.5, 0.2, 0.6, 0.3, 0.7, 0.35, 0.75, 0.4, 0.8, 0.45, 0.85, 0.5, 0.9];
        for m in [PacfMethod::YuleWalker, PacfMethod::Ols] {
            let p = pacf(&x, 3, m).unwrap();
            assert!((p[0] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn adfuller_rejects_constant_series() {
        let x = Array1::from_elem(20, 3.0);
        assert!(adfuller(&x, 2, AdfRegression::C, AutoLag::None).is_err());
    }

    #[test]
    fn acovf_adjusted_uses_smaller_divisor() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        let biased = acovf(&x, false, true, 3).unwrap();
        let adjusted = acovf(&x, true, true, 3).unwrap();
        // Lag 0 divisor is identical (n), so the values match there.
        assert!((biased[0] - adjusted[0]).abs() < 1e-12);
        // For positive lags the adjusted estimator divides by a smaller number,
        // so |adjusted| >= |biased|.
        assert!(adjusted[1].abs() >= biased[1].abs() - 1e-12);
    }
}

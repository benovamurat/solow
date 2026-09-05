//! Heteroscedasticity and serial-correlation diagnostic tests.

use ndarray::{Array1, Array2};
use solow_core::error::Result;
use solow_distributions::chi2_sf;
use solow_regression::LinearModel;

/// Breusch–Pagan Lagrange-multiplier test for heteroscedasticity.
///
/// Regresses the squared residuals on `exog_het` (which must contain a
/// constant) and reports the Koenker (studentized, robust) LM statistic
/// `nobs · R²` with its chi-squared p-value (`k − 1` d.o.f.), together with the
/// auxiliary regression's F statistic and its p-value. Returns
/// `(lm, lm_pvalue, fvalue, f_pvalue)`. Mirrors the reference
/// `het_breuschpagan` with `robust=True`.
pub fn het_breuschpagan(
    resid: &Array1<f64>,
    exog_het: &Array2<f64>,
) -> Result<(f64, f64, f64, f64)> {
    let (nobs, nvars) = exog_het.dim();
    let y: Array1<f64> = resid.mapv(|r| r * r);
    let res = LinearModel::ols(y, exog_het.clone())?.fit()?;
    let lm = nobs as f64 * res.rsquared;
    let lm_pvalue = chi2_sf(lm, (nvars - 1) as f64);
    Ok((lm, lm_pvalue, res.fvalue, res.f_pvalue))
}

/// White's Lagrange-multiplier test for heteroscedasticity.
///
/// Builds the auxiliary design from all squares and pairwise cross-products of
/// the columns of `exog` (which must contain a constant), regresses the squared
/// residuals on it, and reports the LM statistic `nobs · R²` with a chi-squared
/// p-value using the auxiliary model's degrees of freedom (`rank − 1`),
/// together with that model's F statistic and p-value. Returns
/// `(lm, lm_pvalue, fvalue, f_pvalue)`. Mirrors the reference `het_white`.
pub fn het_white(resid: &Array1<f64>, exog: &Array2<f64>) -> Result<(f64, f64, f64, f64)> {
    let (nobs, k) = exog.dim();
    // Upper-triangular index pairs (i <= j), matching numpy's triu_indices.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..k {
        for j in i..k {
            pairs.push((i, j));
        }
    }
    let mut aux = Array2::<f64>::zeros((nobs, pairs.len()));
    for (col, &(i, j)) in pairs.iter().enumerate() {
        for row in 0..nobs {
            aux[[row, col]] = exog[[row, i]] * exog[[row, j]];
        }
    }
    let y: Array1<f64> = resid.mapv(|r| r * r);
    let res = LinearModel::ols(y, aux)?.fit()?;
    let lm = nobs as f64 * res.rsquared;
    // Degrees of freedom take a possibly reduced rank into account: rank - 1.
    let df = res.df_model;
    let lm_pvalue = chi2_sf(lm, df);
    Ok((lm, lm_pvalue, res.fvalue, res.f_pvalue))
}

/// Per-lag output row of [`acorr_ljungbox`].
#[derive(Debug, Clone, Copy)]
pub struct LjungBox {
    /// The lag (1-based) this row reports.
    pub lag: usize,
    /// Ljung–Box cumulative test statistic up to this lag.
    pub lb_stat: f64,
    /// Chi-squared p-value of `lb_stat` with `lag` degrees of freedom.
    pub lb_pvalue: f64,
}

/// Sample autocorrelation function of `x` for lags `0..=maxlag`.
///
/// Uses the biased estimator: the data are demeaned and each autocovariance is
/// divided by `n`, matching the reference `acf` (via `acovf`).
fn acf(x: &Array1<f64>, maxlag: usize) -> Vec<f64> {
    let n = x.len();
    let mean = x.sum() / n as f64;
    let xo: Vec<f64> = x.iter().map(|&v| v - mean).collect();
    let mut acov = vec![0.0; maxlag + 1];
    for (lag, a) in acov.iter_mut().enumerate() {
        let mut s = 0.0;
        for t in lag..n {
            s += xo[t] * xo[t - lag];
        }
        *a = s / n as f64;
    }
    let a0 = acov[0];
    acov.iter().map(|&c| c / a0).collect()
}

/// Ljung–Box test of autocorrelation in the series `x`.
///
/// Computes, for every lag `1..=lags`, the cumulative Ljung–Box statistic
/// `n(n+2) Σ_{k=1}^{lag} ρ_k² / (n−k)` and its chi-squared p-value with `lag`
/// degrees of freedom (`model_df = 0`). Returns one [`LjungBox`] row per lag.
/// Mirrors the reference `acorr_ljungbox` with default `model_df` and
/// `boxpierce=False`.
pub fn acorr_ljungbox(x: &Array1<f64>, lags: usize) -> Vec<LjungBox> {
    let nobs = x.len();
    let n = nobs as f64;
    let sacf = acf(x, lags);
    let mut cum = 0.0;
    let mut out = Vec::with_capacity(lags);
    for (lag, &rho) in sacf.iter().enumerate().take(lags + 1).skip(1) {
        cum += rho * rho / (n - lag as f64);
        let lb = n * (n + 2.0) * cum;
        out.push(LjungBox {
            lag,
            lb_stat: lb,
            lb_pvalue: chi2_sf(lb, lag as f64),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ljungbox_lag0_count() {
        let x = array![0.5, -0.2, 0.1, 0.3, -0.4, 0.2, 0.0, -0.1, 0.25, -0.15];
        let res = acorr_ljungbox(&x, 3);
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].lag, 1);
        for r in &res {
            assert!(r.lb_stat >= 0.0);
            assert!((0.0..=1.0).contains(&r.lb_pvalue));
        }
    }

    #[test]
    fn acf_lag0_is_one() {
        let x = array![1.0, 2.0, 3.0, 2.0, 1.0, 0.5];
        let a = acf(&x, 2);
        assert!((a[0] - 1.0).abs() < 1e-12);
    }
}

//! Collinearity and goodness-of-fit-to-normal diagnostics.
//!
//! Provides the variance-inflation factor ([`variance_inflation_factor`]) for a
//! single regressor and the Lilliefors / Kolmogorov–Smirnov test of normality
//! ([`lilliefors`], [`kstest_normal`]).
//!
//! Mirrors the reference `…stats.outliers_influence.variance_inflation_factor`
//! and `…stats.diagnostic.lilliefors` / `kstest_normal`.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::norm_cdf;
use solow_regression::LinearModel;

/// Variance-inflation factor of column `exog_idx` of the design matrix `exog`.
///
/// Regresses column `exog_idx` on all *other* columns (no separate constant is
/// added — `exog` should already contain whatever constant the model uses) and
/// returns `VIF = 1 / (1 − R²)` of that auxiliary regression. Mirrors the
/// reference `variance_inflation_factor`.
pub fn variance_inflation_factor(exog: &Array2<f64>, exog_idx: usize) -> Result<f64> {
    let (n, k) = exog.dim();
    if exog_idx >= k {
        return Err(Error::Value("exog_idx out of range".into()));
    }
    if k < 2 {
        return Err(Error::Value("need at least two columns for a VIF".into()));
    }
    // x_i is the target column; x_noti is everything else (column order kept).
    let mut x_noti = Array2::<f64>::zeros((n, k - 1));
    let mut cc = 0usize;
    for j in 0..k {
        if j == exog_idx {
            continue;
        }
        for i in 0..n {
            x_noti[[i, cc]] = exog[[i, j]];
        }
        cc += 1;
    }
    let x_i = exog.column(exog_idx).to_owned();
    let res = LinearModel::ols(x_i, x_noti)?.fit()?;
    Ok(1.0 / (1.0 - res.rsquared))
}

/// Assumed reference distribution for the Lilliefors test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LillieforsDist {
    /// Normal distribution with mean and variance estimated from the sample.
    Norm,
}

/// Two-sided Kolmogorov–Smirnov statistic of the sorted standardized sample
/// `z` against the standard-normal CDF.
///
/// `D = max(D⁺, D⁻)` with `D⁺ = maxᵢ (i/n − Φ(z₍ᵢ₎))` and
/// `D⁻ = maxᵢ (Φ(z₍ᵢ₎) − (i−1)/n)` for `i = 1 … n` (sorted ascending). This is
/// the reference `ksstat(z, norm.cdf, 'two_sided')`.
fn ks_stat_normal(z: &mut [f64]) -> f64 {
    z.sort_by(|a, b| a.total_cmp(b));
    let n = z.len() as f64;
    let mut d_plus = f64::NEG_INFINITY;
    let mut d_min = f64::NEG_INFINITY;
    for (idx, &zi) in z.iter().enumerate() {
        let cdf = norm_cdf(zi);
        let i = idx as f64; // 0-based
        let dp = (i + 1.0) / n - cdf; // (i+1)/n - F
        let dm = cdf - i / n; // F - i/n
        if dp > d_plus {
            d_plus = dp;
        }
        if dm > d_min {
            d_min = dm;
        }
    }
    d_plus.max(d_min)
}

/// Dalal–Wilkinson approximation of the Lilliefors p-value for normality.
///
/// Valid (per the reference) for p-values below ~0.1; this is the closed-form
/// `pval_lf` used by the reference's `pvalmethod="approx"`. For `n > 100` the
/// statistic is rescaled and `n` capped at 100.
fn pval_lf(d_max: f64, n: usize) -> f64 {
    let mut d = d_max;
    let mut nn = n as f64;
    if n > 100 {
        d *= (nn / 100.0).powf(0.49);
        nn = 100.0;
    }
    (-7.01256 * d * d * (nn + 2.78019) + 2.99587 * d * (nn + 2.78019).sqrt() - 0.122119
        + 0.974598 / nn.sqrt()
        + 1.67997 / nn)
        .exp()
}

/// Lilliefors test of normality with estimated parameters.
///
/// Standardizes `x` by its sample mean and (ddof = 1) standard deviation, then
/// returns the two-sided KS statistic against the standard normal together with
/// the Dalal–Wilkinson approximate p-value (the reference's
/// `pvalmethod="approx"`). The statistic is closed form; the p-value is the
/// closed-form approximation. Requires `n ≥ 4`.
///
/// Mirrors the reference `lilliefors(x, dist="norm", pvalmethod="approx")`.
pub fn lilliefors(x: &Array1<f64>, dist: LillieforsDist) -> Result<(f64, f64)> {
    let LillieforsDist::Norm = dist;
    let n = x.len();
    if n < 4 {
        return Err(Error::Value(
            "Lilliefors test requires at least 4 observations".into(),
        ));
    }
    let nf = n as f64;
    let mean = x.sum() / nf;
    // Sample standard deviation (ddof = 1).
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (nf - 1.0);
    let sd = var.sqrt();
    let mut z: Vec<f64> = x.iter().map(|&v| (v - mean) / sd).collect();
    let d_ks = ks_stat_normal(&mut z);
    let pval = pval_lf(d_ks, n);
    Ok((d_ks, pval))
}

/// Alias for the Lilliefors normality test, matching the reference name
/// `kstest_normal`. Returns `(statistic, pvalue)`.
pub fn kstest_normal(x: &Array1<f64>) -> Result<(f64, f64)> {
    lilliefors(x, LillieforsDist::Norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn vif_orthogonal_columns_is_one() {
        // Two orthogonal slopes plus a constant: VIF of a slope == 1.
        let x = array![
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
        ];
        let v = variance_inflation_factor(&x, 1).unwrap();
        assert!((v - 1.0).abs() < 1e-9);
    }

    #[test]
    fn lilliefors_statistic_in_unit_range() {
        let x = array![0.1, -0.5, 0.3, 1.2, -0.7, 0.4, -0.2, 0.9, -1.1, 0.05];
        let (stat, p) = lilliefors(&x, LillieforsDist::Norm).unwrap();
        assert!((0.0..=1.0).contains(&stat));
        assert!(p > 0.0);
        let (stat2, _) = kstest_normal(&x).unwrap();
        assert!((stat - stat2).abs() < 1e-15);
    }
}

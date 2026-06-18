//! Normality and serial-correlation diagnostics on a residual series.

use ndarray::Array1;
use solow_distributions::chi2_sf;

/// Durbin–Watson statistic for first-order serial correlation in `resid`.
///
/// Defined as `sum((e_t - e_{t-1})^2) / sum(e_t^2)`; lies in `[0, 4]`, equals
/// `2` under the null of no serial correlation.
pub fn durbin_watson(resid: &Array1<f64>) -> f64 {
    let n = resid.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut num = 0.0;
    let mut den = resid[0] * resid[0];
    for i in 1..n {
        let d = resid[i] - resid[i - 1];
        num += d * d;
        den += resid[i] * resid[i];
    }
    num / den
}

/// The `k`-th central moment of `x` about its mean (population / biased form,
/// i.e. divided by `n`).
fn central_moment(x: &Array1<f64>, mean: f64, k: i32) -> f64 {
    let n = x.len() as f64;
    x.iter().map(|&v| (v - mean).powi(k)).sum::<f64>() / n
}

/// Output of [`jarque_bera`].
#[derive(Debug, Clone, Copy)]
pub struct JarqueBera {
    /// Jarque–Bera test statistic.
    pub statistic: f64,
    /// Two-sided p-value from the chi-squared distribution with 2 d.o.f.
    pub pvalue: f64,
    /// Sample skewness (biased estimator).
    pub skew: f64,
    /// Sample kurtosis (biased, non-excess: normal distribution gives `3`).
    pub kurtosis: f64,
}

/// Jarque–Bera test of normality.
///
/// Returns the statistic `n·(S²/6 + (K−3)²/24)` where `S` is the (biased)
/// sample skewness and `K` the (biased, non-excess) sample kurtosis, its
/// chi-squared(2) p-value, and the underlying skewness and kurtosis.
pub fn jarque_bera(resid: &Array1<f64>) -> JarqueBera {
    let n = resid.len() as f64;
    let mean = resid.sum() / n;
    let m2 = central_moment(resid, mean, 2);
    let m3 = central_moment(resid, mean, 3);
    let m4 = central_moment(resid, mean, 4);
    let skew = m3 / m2.powf(1.5);
    let kurtosis = m4 / (m2 * m2); // non-excess kurtosis (== 3 + excess)
    let jb = (n / 6.0) * (skew * skew + 0.25 * (kurtosis - 3.0).powi(2));
    JarqueBera {
        statistic: jb,
        pvalue: chi2_sf(jb, 2.0),
        skew,
        kurtosis,
    }
}

/// Z-score of D'Agostino's skewness test (transformed sample skewness).
///
/// Mirrors the reference `scipy.stats.skewtest` transform.
fn skew_z(skew: f64, n: f64) -> f64 {
    let y = skew * (((n + 1.0) * (n + 3.0)) / (6.0 * (n - 2.0))).sqrt();
    let beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
        / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    let w2 = -1.0 + (2.0 * (beta2 - 1.0)).sqrt();
    let delta = 1.0 / (0.5 * w2.ln()).sqrt();
    let alpha = (2.0 / (w2 - 1.0)).sqrt();
    let y = if y == 0.0 { 1.0 } else { y };
    let ya = y / alpha;
    delta * (ya + (ya * ya + 1.0).sqrt()).ln()
}

/// Z-score of Anscombe–Glynn kurtosis test (transformed sample kurtosis).
///
/// Mirrors the reference `scipy.stats.kurtosistest` transform. `kurt` is the
/// non-excess (Pearson) kurtosis.
fn kurtosis_z(kurt: f64, n: f64) -> f64 {
    let e = 3.0 * (n - 1.0) / (n + 1.0);
    let varb2 = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) * (n + 1.0) * (n + 3.0) * (n + 5.0));
    let x = (kurt - e) / varb2.sqrt();
    let sqrtbeta1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0))
        * (6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))).sqrt();
    let a =
        6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + (1.0 + 4.0 / (sqrtbeta1 * sqrtbeta1)).sqrt());
    let term1 = 1.0 - 2.0 / (9.0 * a);
    let denom = 1.0 + x * (2.0 / (a - 4.0)).sqrt();
    let term2 = denom.signum() * ((1.0 - 2.0 / a) / denom.abs()).powf(1.0 / 3.0);
    (term1 - term2) / (2.0 / (9.0 * a)).sqrt()
}

/// D'Agostino–Pearson omnibus test of normality (the reference's
/// `omni_normtest`).
///
/// Returns `(statistic, pvalue)` where the statistic is `Z_skew² + Z_kurt²`,
/// asymptotically chi-squared with 2 d.o.f. Requires at least 8 observations;
/// fewer yields `(NaN, NaN)`.
pub fn omni_normtest(resid: &Array1<f64>) -> (f64, f64) {
    let n = resid.len();
    if n < 8 {
        return (f64::NAN, f64::NAN);
    }
    let nf = n as f64;
    let mean = resid.sum() / nf;
    let m2 = central_moment(resid, mean, 2);
    let m3 = central_moment(resid, mean, 3);
    let m4 = central_moment(resid, mean, 4);
    let skew = m3 / m2.powf(1.5);
    let kurt = m4 / (m2 * m2);
    let zs = skew_z(skew, nf);
    let zk = kurtosis_z(kurt, nf);
    let stat = zs * zs + zk * zk;
    (stat, chi2_sf(stat, 2.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn dw_no_correlation_near_two() {
        // Alternating residuals -> strong negative correlation -> dw near 4.
        let r = array![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let dw = durbin_watson(&r);
        // Strong negative serial correlation pushes the statistic toward 4.
        assert!(dw > 3.0, "dw = {dw}");
    }

    #[test]
    fn jb_symmetric_low_skew() {
        let r = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let jb = jarque_bera(&r);
        assert!(jb.skew.abs() < 1e-12);
    }
}

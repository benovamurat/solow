//! Extended descriptive statistics for a numeric sample.
//!
//! [`describe`] computes the standard battery of summary statistics for a 1-D
//! numeric sample (location, dispersion, shape, mode, percentiles and the
//! Jarque–Bera normality test), mirroring the numeric columns of the reference
//! `descriptivestats.describe` / `Description`.

use crate::normality::jarque_bera;
use ndarray::Array1;
use solow_distributions::{norm_ppf, t_ppf};

/// Percentile levels reported by [`describe`] (the reference default).
pub const PERCENTILES: [f64; 9] = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0];

/// Summary statistics for a numeric sample (the numeric block of the reference
/// `Description`). All values are computed with the same conventions as the
/// reference: `std` uses `ddof = 1`, percentiles use linear interpolation.
#[derive(Debug, Clone)]
pub struct Description {
    /// Number of observations.
    pub nobs: f64,
    /// Number of missing (NaN) observations dropped before computation.
    pub missing: f64,
    /// Sample mean.
    pub mean: f64,
    /// Standard error of the mean, `std / sqrt(nobs)`.
    pub std_err: f64,
    /// Upper confidence limit `mean + q * std_err`.
    pub upper_ci: f64,
    /// Lower confidence limit `mean - q * std_err`.
    pub lower_ci: f64,
    /// Sample standard deviation (`ddof = 1`).
    pub std: f64,
    /// Interquartile range, `q75 - q25`.
    pub iqr: f64,
    /// IQR rescaled to a normal-distribution standard deviation.
    pub iqr_normal: f64,
    /// Mean absolute deviation about the mean.
    pub mad: f64,
    /// MAD rescaled to a normal-distribution standard deviation.
    pub mad_normal: f64,
    /// Coefficient of variation, `std / mean`.
    pub coef_var: f64,
    /// Range, `max - min`.
    pub range: f64,
    /// Maximum value.
    pub max: f64,
    /// Minimum value.
    pub min: f64,
    /// Sample skewness (biased estimator).
    pub skew: f64,
    /// Sample kurtosis (biased, non-excess: a normal gives `3`).
    pub kurtosis: f64,
    /// Jarque–Bera statistic.
    pub jarque_bera: f64,
    /// Jarque–Bera chi-squared(2) p-value.
    pub jarque_bera_pval: f64,
    /// Mode (smallest most-frequent value).
    pub mode: f64,
    /// Relative frequency of the mode, `count / nobs`.
    pub mode_freq: f64,
    /// Median (the 50th percentile).
    pub median: f64,
    /// Percentile values at [`PERCENTILES`].
    pub percentiles: Vec<f64>,
}

/// Linear-interpolation percentile, matching `numpy.percentile` / pandas
/// `quantile` (method "linear"). `sorted` must be in ascending order and `q` is
/// a probability in `[0, 1]`.
fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Compute the descriptive statistics of a numeric sample.
///
/// NaN entries are dropped (and counted in `missing`). `alpha` sets the
/// confidence-interval coverage to `1 - alpha`; with `use_t = true` the
/// critical value is from the Student-t distribution with `nobs - 1` degrees of
/// freedom, otherwise from the standard normal. Mirrors the numeric block of
/// the reference `describe`.
pub fn describe(data: &Array1<f64>, alpha: f64, use_t: bool) -> Description {
    let total = data.len() as f64;
    let mut clean: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = clean.len() as f64;
    let missing = total - n;

    let mean = clean.iter().sum::<f64>() / n;
    // ddof = 1 sample variance / std.
    let var = clean.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();
    let std_err = std / n.sqrt();

    let q = if use_t {
        t_ppf(1.0 - alpha / 2.0, n - 1.0)
    } else {
        norm_ppf(1.0 - alpha / 2.0)
    };

    clean.sort_by(|a, b| a.total_cmp(b));
    let q25 = percentile_sorted(&clean, 0.25);
    let q50 = percentile_sorted(&clean, 0.5);
    let q75 = percentile_sorted(&clean, 0.75);
    let iqr = q75 - q25;
    // iqr_normal divisor: norm.ppf(0.75) - norm.ppf(0.25).
    let iqr_normal = iqr / (norm_ppf(0.75) - norm_ppf(0.25));

    let mad = clean.iter().map(|&v| (v - mean).abs()).sum::<f64>() / n;
    let mad_normal = mad / (2.0 / std::f64::consts::PI).sqrt();

    let coef_var = std / mean;
    let max = *clean.last().unwrap();
    let min = clean[0];
    let range = max - min;

    let jb = jarque_bera(&Array1::from(clean.clone()));

    // Mode: smallest value with the largest count (scipy.stats.mode convention).
    let (mode, mode_count) = compute_mode(&clean);
    let mode_freq = mode_count / n;

    let percentiles: Vec<f64> = PERCENTILES
        .iter()
        .map(|&p| percentile_sorted(&clean, p / 100.0))
        .collect();

    Description {
        nobs: n,
        missing,
        mean,
        std_err,
        upper_ci: mean + q * std_err,
        lower_ci: mean - q * std_err,
        std,
        iqr,
        iqr_normal,
        mad,
        mad_normal,
        coef_var,
        range,
        max,
        min,
        skew: jb.skew,
        kurtosis: jb.kurtosis,
        jarque_bera: jb.statistic,
        jarque_bera_pval: jb.pvalue,
        mode,
        mode_freq,
        median: q50,
        percentiles,
    }
}

/// Smallest most-frequent value and its count, on a pre-sorted slice.
fn compute_mode(sorted: &[f64]) -> (f64, f64) {
    let mut best_val = sorted[0];
    let mut best_count = 0usize;
    let mut i = 0;
    while i < sorted.len() {
        let v = sorted[i];
        let mut j = i;
        while j < sorted.len() && sorted[j] == v {
            j += 1;
        }
        let count = j - i;
        if count > best_count {
            best_count = count;
            best_val = v;
        }
        i = j;
    }
    (best_val, best_count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn basic_mean_and_median() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = describe(&x, 0.05, false);
        assert!((d.mean - 3.0).abs() < 1e-12);
        assert!((d.median - 3.0).abs() < 1e-12);
        assert!((d.nobs - 5.0).abs() < 1e-12);
    }

    #[test]
    fn nan_counts_as_missing() {
        let x = array![1.0, f64::NAN, 3.0];
        let d = describe(&x, 0.05, false);
        assert!((d.missing - 1.0).abs() < 1e-12);
        assert!((d.nobs - 2.0).abs() < 1e-12);
    }
}

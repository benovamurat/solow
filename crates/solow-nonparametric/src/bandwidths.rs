//! Rule-of-thumb bandwidth selectors for univariate kernel density estimation.
//!
//! Each selector returns a bandwidth of the form `C * A * n^(-1/5)`, where
//! `A = min(std(x, ddof=1), IQR / 1.349)` is a robust scale estimate and `C`
//! is a rule-specific constant.

use ndarray::Array1;
use solow_core::error::{Error, Result};

/// Constant `1.349 ≈ Φ⁻¹(0.75) − Φ⁻¹(0.25)` used to normalize the inter-quartile
/// range to a standard-deviation scale, matching the reference's hard-coded value.
const IQR_NORMALIZE: f64 = 1.349;

/// Normal-reference plug-in constant for the Gaussian (second-order) kernel,
/// `C = 2·(√π · (ν!)³ · R(k) / (2ν · (2ν)! · κ_ν²))^(1/(2ν+1))` with `ν = 2`.
///
/// This is the value `kernels.Gaussian().normal_reference_constant` returns,
/// `≈ 1.0592238410488122`.
const GAUSSIAN_NORMAL_REFERENCE_CONSTANT: f64 = 1.059_223_841_048_812_2;

/// Linear-interpolation percentile, equivalent to NumPy's default
/// (`method="linear"`) and SciPy's `scoreatpercentile`.
///
/// `q` is given in percent (e.g. `25.0`). The data is copied and sorted.
fn percentile(x: &Array1<f64>, q: f64) -> f64 {
    let n = x.len();
    debug_assert!(n > 0);
    let mut sorted: Vec<f64> = x.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if n == 1 {
        return sorted[0];
    }
    // Position on the [0, n-1] index scale.
    let pos = (q / 100.0) * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Sample standard deviation with one delta degree of freedom (`ddof = 1`).
fn std_ddof1(x: &Array1<f64>) -> f64 {
    let n = x.len();
    debug_assert!(n > 1);
    let mean = x.sum() / n as f64;
    let ss: f64 = x.iter().map(|&v| (v - mean) * (v - mean)).sum();
    (ss / (n as f64 - 1.0)).sqrt()
}

/// Robust scale estimate `A = min(std(x, ddof=1), IQR / 1.349)`.
///
/// When the inter-quartile range is zero the standard deviation is used
/// directly. Mirrors the reference's `_select_sigma`.
///
/// # Errors
/// Returns an error if `x` has fewer than two elements.
pub fn select_sigma(x: &Array1<f64>) -> Result<f64> {
    if x.len() < 2 {
        return Err(Error::Value(
            "bandwidth selection requires at least two observations".into(),
        ));
    }
    let iqr = (percentile(x, 75.0) - percentile(x, 25.0)) / IQR_NORMALIZE;
    let std_dev = std_ddof1(x);
    if iqr > 0.0 {
        Ok(std_dev.min(iqr))
    } else {
        Ok(std_dev)
    }
}

/// Silverman's rule-of-thumb bandwidth: `0.9 · A · n^(-1/5)`.
///
/// # Errors
/// Returns an error if `x` has fewer than two elements.
pub fn bw_silverman(x: &Array1<f64>) -> Result<f64> {
    let a = select_sigma(x)?;
    let n = x.len() as f64;
    Ok(0.9 * a * n.powf(-0.2))
}

/// Scott's rule-of-thumb bandwidth: `1.059 · A · n^(-1/5)`.
///
/// # Errors
/// Returns an error if `x` has fewer than two elements.
pub fn bw_scott(x: &Array1<f64>) -> Result<f64> {
    let a = select_sigma(x)?;
    let n = x.len() as f64;
    Ok(1.059 * a * n.powf(-0.2))
}

/// Normal-reference plug-in bandwidth for the Gaussian kernel:
/// `C · A · n^(-1/5)` with `C ≈ 1.0592238410488122`.
///
/// This minimizes the mean integrated squared error when the underlying
/// distribution is normal. For the Gaussian kernel it agrees with Scott's
/// rule to two decimal places.
///
/// # Errors
/// Returns an error if `x` has fewer than two elements.
pub fn bw_normal_reference(x: &Array1<f64>) -> Result<f64> {
    let a = select_sigma(x)?;
    let n = x.len() as f64;
    Ok(GAUSSIAN_NORMAL_REFERENCE_CONSTANT * a * n.powf(-0.2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn percentile_matches_numpy_linear() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((percentile(&x, 75.0) - 7.75).abs() < 1e-12);
        assert!((percentile(&x, 25.0) - 3.25).abs() < 1e-12);
        assert!((percentile(&x, 50.0) - 5.5).abs() < 1e-12);
    }

    #[test]
    fn std_uses_ddof1() {
        let x = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        // population std is 2.0 with ddof=0; ddof=1 is sqrt(32/7).
        assert!((std_ddof1(&x) - (32.0_f64 / 7.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn bandwidth_ordering() {
        let x = array![0.1, 0.5, 0.9, 1.3, 2.1, 2.8, 3.0, 3.6, 4.4, 5.0];
        let s = bw_silverman(&x).unwrap();
        let sc = bw_scott(&x).unwrap();
        let nr = bw_normal_reference(&x).unwrap();
        assert!(s > 0.0 && sc > 0.0 && nr > 0.0);
        // 0.9 < 1.059 ≈ 1.0592 so silverman < scott < normal_reference.
        assert!(s < sc);
        assert!(sc < nr);
    }

    #[test]
    fn too_few_points_errors() {
        let x = array![1.0];
        assert!(bw_silverman(&x).is_err());
    }
}

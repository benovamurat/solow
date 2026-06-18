//! Nearest positive-semidefinite correlation / covariance matrices and the
//! correlation ↔ covariance helpers.
//!
//! Provides [`cov2corr`] / [`corr2cov`] (the reference `moment_helpers`),
//! [`corr_clipped`] (single eigenvalue clip), [`corr_nearest`] (Higham-style
//! iterative alternating projection), and [`cov_nearest`] which routes a
//! covariance matrix through the chosen correlation correction.
//!
//! Mirrors the reference `…stats.correlation_tools` (`corr_nearest`,
//! `corr_clipped`, `cov_nearest`) and `…stats.moment_helpers`
//! (`cov2corr`, `corr2cov`).

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::eigh;

/// Convert a covariance matrix to a correlation matrix, `Cᵢⱼ / (σᵢ σⱼ)`.
///
/// Mirrors the reference `cov2corr`.
pub fn cov2corr(cov: &Array2<f64>) -> Array2<f64> {
    let (corr, _) = cov2corr_std(cov);
    corr
}

/// Like [`cov2corr`] but also returns the per-variable standard deviations
/// `σᵢ = sqrt(Cᵢᵢ)` (the reference `cov2corr(..., return_std=True)`).
pub fn cov2corr_std(cov: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let k = cov.nrows();
    let std: Array1<f64> = Array1::from_iter((0..k).map(|i| cov[[i, i]].sqrt()));
    let mut corr = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            corr[[i, j]] = cov[[i, j]] / (std[i] * std[j]);
        }
    }
    (corr, std)
}

/// Convert a correlation matrix to a covariance matrix given standard
/// deviations `std`: `Rᵢⱼ · σᵢ σⱼ`. Mirrors the reference `corr2cov`.
pub fn corr2cov(corr: &Array2<f64>, std: &Array1<f64>) -> Array2<f64> {
    let k = corr.nrows();
    let mut cov = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            cov[[i, j]] = corr[[i, j]] * std[i] * std[j];
        }
    }
    cov
}

/// Clip the eigenvalues of `x` from below at `value`, returning the rebuilt
/// matrix `V diag(max(w, value)) Vᵀ` and whether any eigenvalue was clipped.
fn clip_evals(x: &Array2<f64>, value: f64) -> Result<(Array2<f64>, bool)> {
    let (w, v) = eigh(x)?;
    let clipped = w.iter().any(|&e| e < value);
    let k = w.len();
    // x_new = V · diag(max(w, value)) · Vᵀ
    let mut scaled = v.clone(); // columns scaled by clamped eigenvalues
    for j in 0..k {
        let ev = w[j].max(value);
        for i in 0..k {
            scaled[[i, j]] *= ev;
        }
    }
    let x_new = scaled.dot(&v.t());
    Ok((x_new, clipped))
}

/// Nearest PSD correlation matrix by a single eigenvalue clip plus rescaling so
/// the diagonal is one.
///
/// If `corr` is already PSD at the given `threshold` the input is returned
/// unchanged. Mirrors the reference `corr_clipped`.
pub fn corr_clipped(corr: &Array2<f64>, threshold: f64) -> Result<Array2<f64>> {
    let (x_new, clipped) = clip_evals(corr, threshold)?;
    if !clipped {
        return Ok(corr.clone());
    }
    // Rescale to unit diagonal: x / outer(d, d) with d = sqrt(diag).
    let k = x_new.nrows();
    let d: Vec<f64> = (0..k).map(|i| x_new[[i, i]].sqrt()).collect();
    let mut out = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            out[[i, j]] = x_new[[i, j]] / (d[i] * d[j]);
        }
    }
    Ok(out)
}

/// Nearest PSD correlation matrix by Higham-style alternating projection.
///
/// Iteratively clips the eigenvalues of `x − Δ` from below at `threshold`,
/// resets the diagonal to one, and accumulates the correction `Δ`. Stops early
/// once the clip leaves the matrix unchanged (already PSD at the threshold),
/// otherwise runs up to `k · n_fact` iterations. Mirrors the reference
/// `corr_nearest`.
pub fn corr_nearest(corr: &Array2<f64>, threshold: f64, n_fact: usize) -> Result<Array2<f64>> {
    let k = corr.nrows();
    if corr.ncols() != k {
        return Err(Error::Shape("matrix is not square".into()));
    }
    let mut diff = Array2::<f64>::zeros((k, k));
    let mut x_new = corr.clone();
    let max_iter = k * n_fact;
    for _ in 0..max_iter {
        let x_adj = &x_new - &diff;
        let (x_psd, clipped) = clip_evals(&x_adj, threshold)?;
        if !clipped {
            x_new = x_psd;
            break;
        }
        diff = &x_psd - &x_adj;
        x_new = x_psd;
        for i in 0..k {
            x_new[[i, i]] = 1.0;
        }
    }
    Ok(x_new)
}

/// Method used by [`cov_nearest`] for the correlation-matrix correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NearestMethod {
    /// Single eigenvalue clip ([`corr_clipped`]); fast, larger distance.
    Clipped,
    /// Iterative alternating projection ([`corr_nearest`]).
    Nearest,
}

/// Nearest PSD covariance matrix, leaving the variances (diagonal) unchanged.
///
/// Converts `cov` to a correlation matrix, applies the chosen correction
/// (`Clipped` → [`corr_clipped`], `Nearest` → [`corr_nearest`]), then converts
/// back with the original standard deviations. Mirrors the reference
/// `cov_nearest`.
pub fn cov_nearest(
    cov: &Array2<f64>,
    method: NearestMethod,
    threshold: f64,
    n_fact: usize,
) -> Result<Array2<f64>> {
    let (corr, std) = cov2corr_std(cov);
    let corr_fixed = match method {
        NearestMethod::Clipped => corr_clipped(&corr, threshold)?,
        NearestMethod::Nearest => corr_nearest(&corr, threshold, n_fact)?,
    };
    Ok(corr2cov(&corr_fixed, &std))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cov2corr_roundtrip() {
        let cov = array![[4.0, 2.0, 0.0], [2.0, 9.0, -3.0], [0.0, -3.0, 16.0]];
        let (corr, std) = cov2corr_std(&cov);
        assert!((corr[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((corr[[0, 1]] - 2.0 / (2.0 * 3.0)).abs() < 1e-12);
        let back = corr2cov(&corr, &std);
        for i in 0..3 {
            for j in 0..3 {
                assert!((back[[i, j]] - cov[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn corr_nearest_makes_psd() {
        // An indefinite "correlation" matrix.
        let corr = array![[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]];
        let fixed = corr_nearest(&corr, 1e-7, 100).unwrap();
        let (w, _) = eigh(&fixed).unwrap();
        assert!(w[0] >= -1e-8, "smallest eigenvalue {} negative", w[0]);
        for i in 0..3 {
            assert!((fixed[[i, i]] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn corr_clipped_psd_passthrough() {
        let corr = array![[1.0, 0.2], [0.2, 1.0]];
        let fixed = corr_clipped(&corr, 1e-7).unwrap();
        // Already PSD: returned unchanged.
        for i in 0..2 {
            for j in 0..2 {
                assert!((fixed[[i, j]] - corr[[i, j]]).abs() < 1e-15);
            }
        }
    }
}

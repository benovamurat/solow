//! Univariate Gaussian kernel density estimation.
//!
//! [`KdeUnivariate`] implements the exact Rosenblatt–Parzen estimator with a
//! Gaussian kernel,
//!
//! ```text
//! f̂(t) = 1/(n·h) · Σ_j K((t − x_j)/h),   K(z) = exp(−z²/2)/√(2π),
//! ```
//!
//! evaluated either on the reference's default grid or at user-supplied points.

use std::f64::consts::PI;

use ndarray::Array1;
use solow_core::error::{Error, Result};

use crate::bandwidths::{bw_normal_reference, bw_scott, bw_silverman};

/// Bandwidth selection rule for [`KdeUnivariate::fit`].
#[derive(Debug, Clone, Copy)]
pub enum Bandwidth {
    /// Scott's rule of thumb (`1.059 · A · n^(-1/5)`).
    Scott,
    /// Silverman's rule of thumb (`0.9 · A · n^(-1/5)`).
    Silverman,
    /// Normal-reference plug-in rule (the reference's `"normal_reference"`,
    /// the default).
    NormalReference,
    /// A user-supplied bandwidth.
    Value(f64),
}

/// Univariate kernel density estimator.
///
/// Construct from data, then call [`KdeUnivariate::fit`] to evaluate the
/// density on the default grid, or [`KdeUnivariate::evaluate`] to evaluate it
/// at arbitrary points.
#[derive(Debug, Clone)]
pub struct KdeUnivariate {
    endog: Array1<f64>,
}

/// Output of [`KdeUnivariate::fit`]: the evaluation grid (`support`), the
/// estimated `density` at each grid point, and the bandwidth `bw` used.
#[derive(Debug, Clone)]
pub struct KdeFit {
    /// Grid points at which the density was evaluated.
    pub support: Array1<f64>,
    /// Estimated density aligned with [`KdeFit::support`].
    pub density: Array1<f64>,
    /// Bandwidth used for the estimate.
    pub bw: f64,
}

/// Standard normal density `exp(−z²/2)/√(2π)`.
#[inline]
fn gaussian_kernel(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

impl KdeUnivariate {
    /// Create an estimator over the observations `endog`.
    pub fn new(endog: Array1<f64>) -> Self {
        KdeUnivariate { endog }
    }

    /// Resolve a [`Bandwidth`] choice to a concrete value for the data.
    fn resolve_bw(&self, bw: Bandwidth) -> Result<f64> {
        match bw {
            Bandwidth::Scott => bw_scott(&self.endog),
            Bandwidth::Silverman => bw_silverman(&self.endog),
            Bandwidth::NormalReference => bw_normal_reference(&self.endog),
            Bandwidth::Value(v) => {
                if v > 0.0 && v.is_finite() {
                    Ok(v)
                } else {
                    Err(Error::Value("bandwidth must be positive and finite".into()))
                }
            }
        }
    }

    /// Fit the Gaussian KDE on the reference's default grid.
    ///
    /// The grid is `linspace(min(x) − cut·bw, max(x) + cut·bw, gridsize)` with
    /// `cut = 3` and `gridsize = max(n, 50)`, matching the reference's
    /// non-FFT `kdensity` exactly.
    ///
    /// # Errors
    /// Returns an error if the data is empty, contains non-finite values, or if
    /// the bandwidth rule cannot be evaluated.
    pub fn fit(&self, bw: Bandwidth) -> Result<KdeFit> {
        let n = self.endog.len();
        if n == 0 {
            return Err(Error::Value("KDE requires at least one observation".into()));
        }
        for &v in self.endog.iter() {
            if !v.is_finite() {
                return Err(Error::Value("KDE data must be finite".into()));
            }
        }
        let h = self.resolve_bw(bw)?;

        let cut = 3.0;
        let gridsize = n.max(50);
        let xmin = self.endog.iter().cloned().fold(f64::INFINITY, f64::min);
        let xmax = self.endog.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let a = xmin - cut * h;
        let b = xmax + cut * h;

        let support = linspace(a, b, gridsize);
        let density = self.evaluate_with_bw(&support, h);

        Ok(KdeFit {
            support,
            density,
            bw: h,
        })
    }

    /// Evaluate the density at the given `points` using a resolved bandwidth.
    ///
    /// # Errors
    /// Returns an error like [`KdeUnivariate::fit`].
    pub fn evaluate(&self, points: &Array1<f64>, bw: Bandwidth) -> Result<Array1<f64>> {
        if self.endog.is_empty() {
            return Err(Error::Value("KDE requires at least one observation".into()));
        }
        for &v in self.endog.iter() {
            if !v.is_finite() {
                return Err(Error::Value("KDE data must be finite".into()));
            }
        }
        let h = self.resolve_bw(bw)?;
        Ok(self.evaluate_with_bw(points, h))
    }

    /// Core evaluator: `f̂(t) = 1/(n·h) Σ_j K((t − x_j)/h)` at each point.
    fn evaluate_with_bw(&self, points: &Array1<f64>, h: f64) -> Array1<f64> {
        let n = self.endog.len() as f64;
        points
            .iter()
            .map(|&t| {
                let s: f64 = self
                    .endog
                    .iter()
                    .map(|&xj| gaussian_kernel((t - xj) / h))
                    .sum();
                s / (n * h)
            })
            .collect()
    }
}

/// `gridsize` evenly-spaced points from `a` to `b` inclusive (like NumPy's
/// `linspace` default `endpoint=True`).
fn linspace(a: f64, b: f64, gridsize: usize) -> Array1<f64> {
    if gridsize == 1 {
        return Array1::from_vec(vec![a]);
    }
    let step = (b - a) / (gridsize as f64 - 1.0);
    Array1::from_iter((0..gridsize).map(|i| a + step * i as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn density_integrates_to_one() {
        // Trapezoidal integral over the support should be ~1.
        let x = array![-1.0, -0.3, 0.1, 0.4, 0.9, 1.2, 1.8, 2.0, 2.5, 3.0];
        let kde = KdeUnivariate::new(x);
        let fit = kde.fit(Bandwidth::Scott).unwrap();
        let mut area = 0.0;
        for i in 0..fit.support.len() - 1 {
            let dx = fit.support[i + 1] - fit.support[i];
            area += 0.5 * dx * (fit.density[i] + fit.density[i + 1]);
        }
        assert!((area - 1.0).abs() < 1e-3, "area = {area}");
    }

    #[test]
    fn grid_endpoints_use_cut_three() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let kde = KdeUnivariate::new(x);
        let fit = kde.fit(Bandwidth::Silverman).unwrap();
        let a = fit.support[0];
        let b = fit.support[fit.support.len() - 1];
        assert!((a - (0.0 - 3.0 * fit.bw)).abs() < 1e-12);
        assert!((b - (4.0 + 3.0 * fit.bw)).abs() < 1e-12);
        // gridsize = max(n, 50) = 50.
        assert_eq!(fit.support.len(), 50);
    }

    #[test]
    fn evaluate_matches_manual_sum() {
        let x = array![0.0, 1.0, 2.0];
        let kde = KdeUnivariate::new(x.clone());
        let h = 0.5;
        let pts = array![0.5, 1.5];
        let got = kde.evaluate(&pts, Bandwidth::Value(h)).unwrap();
        for (k, &t) in pts.iter().enumerate() {
            let manual: f64 = x
                .iter()
                .map(|&xj| gaussian_kernel((t - xj) / h))
                .sum::<f64>()
                / (3.0 * h);
            assert!((got[k] - manual).abs() < 1e-12);
        }
    }
}

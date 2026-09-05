//! Robust criterion functions (norms) used by M-estimation.
//!
//! Each norm exposes four functions evaluated element-wise on standardized
//! residuals `z = r / scale`:
//!
//! * [`RobustNorm::rho`] — the criterion `ρ(z)` (an even, non-decreasing-in-`|z|`
//!   function whose sum is minimized by the estimator);
//! * [`RobustNorm::psi`] — its derivative `ψ(z) = ρ'(z)`, the influence function;
//! * [`RobustNorm::weights`] — `ψ(z) / z`, the IRLS weighting function;
//! * [`RobustNorm::psi_deriv`] — `ψ'(z)`, used for the robust covariance matrix.

use std::f64::consts::PI;

/// A robust criterion function for M-estimation.
///
/// Implementors provide `rho`, `psi`, `weights`, and `psi_deriv` evaluated at a
/// single standardized residual; the slice-valued helpers map them element-wise.
pub trait RobustNorm {
    /// The criterion `ρ(z)`.
    fn rho(&self, z: f64) -> f64;
    /// The influence function `ψ(z) = ρ'(z)`.
    fn psi(&self, z: f64) -> f64;
    /// The IRLS weighting function `ψ(z) / z`.
    fn weights(&self, z: f64) -> f64;
    /// The derivative `ψ'(z)` of the influence function.
    fn psi_deriv(&self, z: f64) -> f64;

    /// `ρ` applied element-wise.
    fn rho_arr(&self, z: &[f64]) -> Vec<f64> {
        z.iter().map(|&v| self.rho(v)).collect()
    }
    /// `ψ` applied element-wise.
    fn psi_arr(&self, z: &[f64]) -> Vec<f64> {
        z.iter().map(|&v| self.psi(v)).collect()
    }
    /// `weights` applied element-wise.
    fn weights_arr(&self, z: &[f64]) -> Vec<f64> {
        z.iter().map(|&v| self.weights(v)).collect()
    }
    /// `ψ'` applied element-wise.
    fn psi_deriv_arr(&self, z: &[f64]) -> Vec<f64> {
        z.iter().map(|&v| self.psi_deriv(v)).collect()
    }
}

/// Huber's `t` function — quadratic near zero, linear in the tails.
///
/// The default tuning constant `t = 1.345` yields roughly 95% efficiency at the
/// Gaussian model.
#[derive(Clone, Copy, Debug)]
pub struct HuberT {
    /// Tuning constant; residuals with `|z| > t` are downweighted.
    pub t: f64,
}

impl Default for HuberT {
    fn default() -> Self {
        HuberT { t: 1.345 }
    }
}

impl HuberT {
    /// A Huber norm with an explicit tuning constant.
    pub fn new(t: f64) -> Self {
        HuberT { t }
    }
    #[inline]
    fn subset(&self, z: f64) -> bool {
        z.abs() <= self.t
    }
}

impl RobustNorm for HuberT {
    fn rho(&self, z: f64) -> f64 {
        if self.subset(z) {
            0.5 * z * z
        } else {
            z.abs() * self.t - 0.5 * self.t * self.t
        }
    }
    fn psi(&self, z: f64) -> f64 {
        if self.subset(z) {
            z
        } else {
            self.t * z.signum()
        }
    }
    fn weights(&self, z: f64) -> f64 {
        if self.subset(z) {
            1.0
        } else {
            self.t / z.abs()
        }
    }
    fn psi_deriv(&self, z: f64) -> f64 {
        if z.abs() <= self.t {
            1.0
        } else {
            0.0
        }
    }
}

/// Tukey's biweight (bisquare) — redescending, fully rejecting `|z| > c`.
///
/// The default tuning constant `c = 4.685` yields roughly 95% efficiency at the
/// Gaussian model.
#[derive(Clone, Copy, Debug)]
pub struct TukeyBiweight {
    /// Tuning constant; residuals with `|z| > c` receive zero weight.
    pub c: f64,
}

impl Default for TukeyBiweight {
    fn default() -> Self {
        TukeyBiweight { c: 4.685 }
    }
}

impl TukeyBiweight {
    /// A Tukey biweight norm with an explicit tuning constant.
    pub fn new(c: f64) -> Self {
        TukeyBiweight { c }
    }
    #[inline]
    fn subset(&self, z: f64) -> bool {
        z.abs() <= self.c
    }
}

impl RobustNorm for TukeyBiweight {
    fn rho(&self, z: f64) -> f64 {
        let factor = self.c * self.c / 6.0;
        if self.subset(z) {
            let u = 1.0 - (z / self.c).powi(2);
            -u.powi(3) * factor + factor
        } else {
            factor
        }
    }
    fn psi(&self, z: f64) -> f64 {
        if self.subset(z) {
            let u = 1.0 - (z / self.c).powi(2);
            z * u * u
        } else {
            0.0
        }
    }
    fn weights(&self, z: f64) -> f64 {
        if self.subset(z) {
            let u = 1.0 - (z / self.c).powi(2);
            u * u
        } else {
            0.0
        }
    }
    fn psi_deriv(&self, z: f64) -> f64 {
        if self.subset(z) {
            let r2 = (z / self.c).powi(2);
            let u = 1.0 - r2;
            u * u - (4.0 * z * z / (self.c * self.c)) * u
        } else {
            0.0
        }
    }
}

/// Andrew's wave — sinusoidal, redescending and fully rejecting `|z| > aπ`.
///
/// The default tuning constant is `a = 1.339`.
#[derive(Clone, Copy, Debug)]
pub struct AndrewWave {
    /// Tuning constant; residuals with `|z| > a·π` receive zero weight.
    pub a: f64,
}

impl Default for AndrewWave {
    fn default() -> Self {
        AndrewWave { a: 1.339 }
    }
}

impl AndrewWave {
    /// An Andrew wave norm with an explicit tuning constant.
    pub fn new(a: f64) -> Self {
        AndrewWave { a }
    }
    #[inline]
    fn subset(&self, z: f64) -> bool {
        z.abs() <= self.a * PI
    }
}

impl RobustNorm for AndrewWave {
    fn rho(&self, z: f64) -> f64 {
        let a = self.a;
        if self.subset(z) {
            a * a * (1.0 - (z / a).cos())
        } else {
            a * a * 2.0
        }
    }
    fn psi(&self, z: f64) -> f64 {
        if self.subset(z) {
            self.a * (z / self.a).sin()
        } else {
            0.0
        }
    }
    fn weights(&self, z: f64) -> f64 {
        let ratio = z / self.a;
        if ratio.abs() < f64::EPSILON {
            // sin(x)/x -> 1 as x -> 0; the reference returns 1 here regardless
            // of the cutoff test, matching its small-value branch.
            1.0
        } else if self.subset(z) {
            ratio.sin() / ratio
        } else {
            0.0
        }
    }
    fn psi_deriv(&self, z: f64) -> f64 {
        if self.subset(z) {
            (z / self.a).cos()
        } else {
            0.0
        }
    }
}

/// Plain least squares (`ρ(z) = z²/2`), provided for completeness.
///
/// With this norm RLM reduces to ordinary least squares.
#[derive(Clone, Copy, Debug, Default)]
pub struct LeastSquares;

impl RobustNorm for LeastSquares {
    fn rho(&self, z: f64) -> f64 {
        0.5 * z * z
    }
    fn psi(&self, z: f64) -> f64 {
        z
    }
    fn weights(&self, _z: f64) -> f64 {
        1.0
    }
    fn psi_deriv(&self, _z: f64) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huber_is_quadratic_then_linear() {
        let h = HuberT::default();
        // Inside the band psi(z) = z and weights = 1.
        assert!((h.psi(0.5) - 0.5).abs() < 1e-15);
        assert!((h.weights(0.5) - 1.0).abs() < 1e-15);
        assert!((h.psi_deriv(0.5) - 1.0).abs() < 1e-15);
        // Outside, psi saturates at +/- t and weights decay as t/|z|.
        assert!((h.psi(10.0) - h.t).abs() < 1e-15);
        assert!((h.weights(10.0) - h.t / 10.0).abs() < 1e-15);
        assert!(h.psi_deriv(10.0) == 0.0);
    }

    #[test]
    fn psi_equals_z_times_weights() {
        // For every norm, psi(z) == z * weights(z).
        let z = 1.7_f64;
        for (psi, w) in [
            (HuberT::default().psi(z), HuberT::default().weights(z)),
            (
                TukeyBiweight::default().psi(z),
                TukeyBiweight::default().weights(z),
            ),
            (
                AndrewWave::default().psi(z),
                AndrewWave::default().weights(z),
            ),
            (LeastSquares.psi(z), LeastSquares.weights(z)),
        ] {
            assert!((psi - z * w).abs() < 1e-12);
        }
    }

    #[test]
    fn redescending_norms_reject_far_outliers() {
        let far = 1e3;
        assert_eq!(TukeyBiweight::default().weights(far), 0.0);
        assert_eq!(TukeyBiweight::default().psi(far), 0.0);
        assert_eq!(AndrewWave::default().weights(far), 0.0);
        assert_eq!(AndrewWave::default().psi(far), 0.0);
    }

    #[test]
    fn andrew_weight_at_zero_is_one() {
        // sin(z/a)/(z/a) -> 1 as z -> 0.
        assert!((AndrewWave::default().weights(0.0) - 1.0).abs() < 1e-12);
    }
}

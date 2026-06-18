//! Additional robust criterion functions (norms) for M-estimation.
//!
//! These complement the norms in [`crate::norms`] and follow the same
//! [`RobustNorm`] contract: each evaluates `rho`, `psi`, `weights` and
//! `psi_deriv` element-wise on standardized residuals `z = r / scale`.
//!
//! * [`Hampel`] — a three-part redescending norm (quadratic, linear, linear
//!   descent to zero) tuned by `(a, b, c)`;
//! * [`RamsayE`] — Ramsay's `Eₐ`, a smooth redescending norm with weight
//!   `exp(−a·|z|)`;
//! * [`TrimmedMean`] — least trimmed mean: quadratic inside `|z| ≤ c` and flat
//!   outside (a hard rejection identical to ordinary least squares with a
//!   cut-off).
//!
//! Every formula mirrors the reference's `robust.norms` module exactly, so the
//! frozen golden fixtures match to machine precision.

use crate::norms::RobustNorm;

/// Hampel's three-part redescending norm for M-estimation.
///
/// `psi` is the identity for `|z| ≤ a`, saturates at `±a` for `a < |z| ≤ b`,
/// descends linearly back to zero across `b < |z| ≤ c`, and is zero beyond `c`.
/// The default tuning constants are `a = 2`, `b = 4`, `c = 8`.
#[derive(Clone, Copy, Debug)]
pub struct Hampel {
    /// Inner knot: `psi` is the identity for `|z| ≤ a`.
    pub a: f64,
    /// Middle knot: `psi` saturates at `±a` for `a < |z| ≤ b`.
    pub b: f64,
    /// Outer knot: `psi` descends to zero across `b < |z| ≤ c`, and is zero past `c`.
    pub c: f64,
}

impl Default for Hampel {
    fn default() -> Self {
        Hampel {
            a: 2.0,
            b: 4.0,
            c: 8.0,
        }
    }
}

impl Hampel {
    /// A Hampel norm with explicit tuning constants `a < b < c`.
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Hampel { a, b, c }
    }

    /// Classify `z` into the three active regions by `|z|`.
    ///
    /// Returns `(t1, t2, t3)` where `t1 = |z| ≤ a`, `t2 = a < |z| ≤ b`,
    /// `t3 = b < |z| ≤ c`. Beyond `c` all three are false.
    #[inline]
    fn subset(&self, z: f64) -> (bool, bool, bool) {
        let za = z.abs();
        let t1 = za <= self.a;
        let t2 = za <= self.b && za > self.a;
        let t3 = za <= self.c && za > self.b;
        (t1, t2, t3)
    }
}

impl RobustNorm for Hampel {
    fn rho(&self, z: f64) -> f64 {
        let (a, b, c) = (self.a, self.b, self.c);
        let (t1, t2, t3) = self.subset(z);
        let za = z.abs();
        // The flat tail constant a*(b + c - a)/2 is added for everything past b.
        let mut v = if t1 {
            za * za * 0.5
        } else if t2 {
            a * za - a * a * 0.5
        } else if t3 {
            a * (c - za).powi(2) / (c - b) * (-0.5)
        } else {
            0.0
        };
        if !(t1 || t2) {
            v += a * (b + c - a) * 0.5;
        }
        v
    }

    fn psi(&self, z: f64) -> f64 {
        let (a, b, c) = (self.a, self.b, self.c);
        let (t1, t2, t3) = self.subset(z);
        let s = sign(z);
        if t1 {
            z
        } else if t2 {
            a * s
        } else if t3 {
            a * s * (c - z.abs()) / (c - b)
        } else {
            0.0
        }
    }

    fn weights(&self, z: f64) -> f64 {
        let (a, b, c) = (self.a, self.b, self.c);
        let (t1, t2, t3) = self.subset(z);
        let za = z.abs();
        if t1 {
            1.0
        } else if t2 {
            a / za
        } else if t3 {
            a * (c - za) / (za * (c - b))
        } else {
            0.0
        }
    }

    fn psi_deriv(&self, z: f64) -> f64 {
        let (a, b, c) = (self.a, self.b, self.c);
        let (t1, _t2, t3) = self.subset(z);
        if t1 {
            1.0
        } else if t3 {
            // Reference: -(a * sign(z) * z) / (|z| * (c - b)).
            -(a * sign(z) * z) / (z.abs() * (c - b))
        } else {
            0.0
        }
    }
}

/// Ramsay's `Eₐ` — a smooth, fully redescending norm with weight `exp(−a·|z|)`.
///
/// The default tuning constant is `a = 0.3`. Unlike the hard-cutoff norms the
/// weight is strictly positive everywhere, decaying exponentially in `|z|`.
#[derive(Clone, Copy, Debug)]
pub struct RamsayE {
    /// Tuning constant controlling how fast the weight decays.
    pub a: f64,
}

impl Default for RamsayE {
    fn default() -> Self {
        RamsayE { a: 0.3 }
    }
}

impl RamsayE {
    /// A Ramsay `Eₐ` norm with an explicit tuning constant.
    pub fn new(a: f64) -> Self {
        RamsayE { a }
    }
}

impl RobustNorm for RamsayE {
    fn rho(&self, z: f64) -> f64 {
        let a = self.a;
        let az = a * z.abs();
        (1.0 - (-az).exp() * (1.0 + az)) / (a * a)
    }

    fn psi(&self, z: f64) -> f64 {
        z * (-self.a * z.abs()).exp()
    }

    fn weights(&self, z: f64) -> f64 {
        (-self.a * z.abs()).exp()
    }

    fn psi_deriv(&self, z: f64) -> f64 {
        // Reference: x*dy + y*dx with x = exp(-a|z|), dx = -a*x*sign(z), dy = 1.
        let a = self.a;
        let x = (-a * z.abs()).exp();
        let dx = -a * x * sign(z);
        x + z * dx
    }
}

/// Least trimmed mean — quadratic inside `|z| ≤ c`, flat outside.
///
/// This is ordinary least squares with a hard rejection cut-off: residuals with
/// `|z| ≤ c` keep unit weight, residuals beyond `c` receive zero weight. The
/// default tuning constant is `c = 2`.
#[derive(Clone, Copy, Debug)]
pub struct TrimmedMean {
    /// Cut-off; residuals with `|z| > c` receive zero weight.
    pub c: f64,
}

impl Default for TrimmedMean {
    fn default() -> Self {
        TrimmedMean { c: 2.0 }
    }
}

impl TrimmedMean {
    /// A trimmed-mean norm with an explicit cut-off `c`.
    pub fn new(c: f64) -> Self {
        TrimmedMean { c }
    }

    /// Whether `z` is inside the retained band `|z| ≤ c` (closed boundary).
    #[inline]
    fn subset(&self, z: f64) -> bool {
        z.abs() <= self.c
    }
}

impl RobustNorm for TrimmedMean {
    fn rho(&self, z: f64) -> f64 {
        if self.subset(z) {
            0.5 * z * z
        } else {
            0.5 * self.c * self.c
        }
    }

    fn psi(&self, z: f64) -> f64 {
        if self.subset(z) {
            z
        } else {
            0.0
        }
    }

    fn weights(&self, z: f64) -> f64 {
        if self.subset(z) {
            1.0
        } else {
            0.0
        }
    }

    fn psi_deriv(&self, z: f64) -> f64 {
        if self.subset(z) {
            1.0
        } else {
            0.0
        }
    }
}

/// `sign` matching the reference's `numpy.sign`: `sign(0) == 0`.
///
/// `f64::signum` returns `±1` for zero, so we special-case it here to reproduce
/// the reference's piecewise formulas exactly at `z = 0`.
#[inline]
fn sign(z: f64) -> f64 {
    if z > 0.0 {
        1.0
    } else if z < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hampel_regions_and_boundaries() {
        let h = Hampel::default(); // a=2, b=4, c=8
                                   // Inner quadratic region.
        assert!((h.psi(1.0) - 1.0).abs() < 1e-15);
        assert!((h.weights(1.0) - 1.0).abs() < 1e-15);
        assert!((h.psi_deriv(1.0) - 1.0).abs() < 1e-15);
        // Plateau a<|z|<=b: psi == a, weight == a/|z|, psi_deriv == 0.
        assert!((h.psi(3.0) - 2.0).abs() < 1e-15);
        assert!((h.weights(3.0) - 2.0 / 3.0).abs() < 1e-15);
        assert_eq!(h.psi_deriv(3.0), 0.0);
        // Descent b<|z|<=c.
        assert!((h.psi(6.0) - 2.0 * (8.0 - 6.0) / (8.0 - 4.0)).abs() < 1e-15);
        assert!((h.psi_deriv(6.0) - (-2.0 / 4.0)).abs() < 1e-15);
        // Boundary at c: weight is exactly zero, psi zero.
        assert_eq!(h.weights(8.0), 0.0);
        assert!((h.psi(8.0)).abs() < 1e-15);
        // Beyond c: full rejection.
        assert_eq!(h.psi(100.0), 0.0);
        assert_eq!(h.weights(100.0), 0.0);
        assert_eq!(h.psi_deriv(100.0), 0.0);
    }

    #[test]
    fn hampel_is_odd_in_psi_even_in_rho() {
        let h = Hampel::default();
        for &z in &[0.5, 2.5, 5.0, 9.0] {
            assert!((h.psi(z) + h.psi(-z)).abs() < 1e-14, "psi odd at {z}");
            assert!((h.rho(z) - h.rho(-z)).abs() < 1e-14, "rho even at {z}");
        }
    }

    #[test]
    fn ramsay_weight_is_exp_decay() {
        let r = RamsayE::default(); // a=0.3
        assert!((r.weights(0.0) - 1.0).abs() < 1e-15);
        assert!((r.weights(2.0) - (-0.6_f64).exp()).abs() < 1e-15);
        // psi = z * weight.
        for &z in &[-3.0, -1.0, 0.0, 1.0, 4.0] {
            assert!((r.psi(z) - z * r.weights(z)).abs() < 1e-15);
        }
        // psi_deriv at 0 is 1.
        assert!((r.psi_deriv(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn trimmed_mean_is_ols_with_cutoff() {
        let t = TrimmedMean::default(); // c=2
        assert_eq!(t.weights(1.5), 1.0);
        assert_eq!(t.weights(2.0), 1.0); // closed boundary
        assert_eq!(t.weights(2.0001), 0.0);
        assert_eq!(t.psi(1.5), 1.5);
        assert_eq!(t.psi(3.0), 0.0);
        assert_eq!(t.rho(3.0), 0.5 * 4.0);
        assert_eq!(t.psi_deriv(1.0), 1.0);
        assert_eq!(t.psi_deriv(3.0), 0.0);
    }

    #[test]
    fn psi_equals_z_times_weights() {
        let z = 3.3_f64;
        for (psi, w) in [
            (Hampel::default().psi(z), Hampel::default().weights(z)),
            (RamsayE::default().psi(z), RamsayE::default().weights(z)),
            (
                TrimmedMean::default().psi(z),
                TrimmedMean::default().weights(z),
            ),
        ] {
            assert!((psi - z * w).abs() < 1e-12);
        }
    }

    #[test]
    fn sign_zero_is_zero() {
        assert_eq!(sign(0.0), 0.0);
        assert_eq!(sign(-0.0), 0.0);
        assert_eq!(sign(1.5), 1.0);
        assert_eq!(sign(-2.0), -1.0);
    }
}

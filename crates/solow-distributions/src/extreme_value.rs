//! Extreme-value distributions: Generalised Extreme Value (GEV) and
//! Generalised Pareto (GPD). Uses the standard parametrisation:
//!
//! GEV(μ, σ, ξ):
//!   CDF F(x) = exp{−[1 + ξ·(x − μ)/σ]^(−1/ξ)}, ξ ≠ 0
//!         F(x) = exp{−exp(−(x − μ)/σ)},        ξ = 0
//!
//! GPD(σ, ξ, μ = 0):
//!   CDF G(x) = 1 − [1 + ξ x / σ]^(−1/ξ), ξ ≠ 0
//!         G(x) = 1 − exp(−x/σ),         ξ = 0
//!
//! Fit is by maximum likelihood via a small deterministic coordinate
//! descent (see `Gev::fit`, `Gpd::fit`).

use solow_core::{Error, Result};

/// Generalised extreme-value distribution.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gev {
    /// Location μ.
    pub location: f64,
    /// Scale σ (> 0).
    pub scale: f64,
    /// Shape ξ.
    pub shape: f64,
}

impl Gev {
    /// Construct after validating that σ > 0.
    pub fn new(location: f64, scale: f64, shape: f64) -> Result<Self> {
        if !(scale > 0.0 && scale.is_finite()) {
            return Err(Error::Value("Gev: scale must be > 0".into()));
        }
        Ok(Self { location, scale, shape })
    }

    /// CDF.
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        if self.shape.abs() < 1e-12 {
            (-(-z).exp()).exp()
        } else {
            let arg = 1.0 + self.shape * z;
            if arg <= 0.0 {
                if self.shape > 0.0 { 0.0 } else { 1.0 }
            } else {
                (-arg.powf(-1.0 / self.shape)).exp()
            }
        }
    }

    /// Quantile / inverse CDF.
    pub fn quantile(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        if self.shape.abs() < 1e-12 {
            self.location - self.scale * (-p.ln()).ln()
        } else {
            self.location
                + self.scale / self.shape * ((-p.ln()).powf(-self.shape) - 1.0)
        }
    }

    /// Return-level for a return period `T` (`p = 1 − 1/T`).
    pub fn return_level(&self, t: f64) -> f64 {
        self.quantile(1.0 - 1.0 / t.max(1.0 + 1e-12))
    }

    /// Fit by maximum likelihood.
    pub fn fit(x: &[f64]) -> Result<Self> {
        if x.len() < 5 {
            return Err(Error::Value("Gev::fit: need ≥ 5 observations".into()));
        }
        let mean: f64 = x.iter().sum::<f64>() / x.len() as f64;
        let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
        let mut mu = mean;
        let mut sigma = var.sqrt().max(1e-6);
        let mut xi = 0.1_f64;
        let mut best = ll(x, mu, sigma, xi);
        for _ in 0..300 {
            let mut improved = false;
            for &d_mu in &[0.05 * sigma, -0.05 * sigma] {
                for &d_sigma in &[0.05 * sigma, -0.05 * sigma] {
                    for &d_xi in &[0.05_f64, -0.05_f64] {
                        let new_sigma = (sigma + d_sigma).max(1e-6);
                        let cand = ll(x, mu + d_mu, new_sigma, xi + d_xi);
                        if cand > best + 1e-9 {
                            mu += d_mu;
                            sigma = new_sigma;
                            xi += d_xi;
                            best = cand;
                            improved = true;
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
        Ok(Self { location: mu, scale: sigma, shape: xi })
    }
}

/// Generalised Pareto distribution (threshold μ = 0).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gpd {
    /// Scale σ (> 0).
    pub scale: f64,
    /// Shape ξ.
    pub shape: f64,
}

impl Gpd {
    /// Construct after validating that σ > 0.
    pub fn new(scale: f64, shape: f64) -> Result<Self> {
        if !(scale > 0.0 && scale.is_finite()) {
            return Err(Error::Value("Gpd: scale must be > 0".into()));
        }
        Ok(Self { scale, shape })
    }

    /// CDF.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if self.shape.abs() < 1e-12 {
            1.0 - (-x / self.scale).exp()
        } else {
            let arg = 1.0 + self.shape * x / self.scale;
            if arg <= 0.0 {
                1.0
            } else {
                1.0 - arg.powf(-1.0 / self.shape)
            }
        }
    }

    /// Quantile.
    pub fn quantile(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        if self.shape.abs() < 1e-12 {
            -self.scale * (1.0 - p).ln()
        } else {
            self.scale / self.shape * ((1.0 - p).powf(-self.shape) - 1.0)
        }
    }

    /// Fit — peaks-over-threshold assumes `x` is already the exceedances.
    pub fn fit(x: &[f64]) -> Result<Self> {
        if x.len() < 5 {
            return Err(Error::Value("Gpd::fit: need ≥ 5 exceedances".into()));
        }
        for &v in x.iter() {
            if v < 0.0 {
                return Err(Error::Value("Gpd::fit: exceedances must be ≥ 0".into()));
            }
        }
        let mean: f64 = x.iter().sum::<f64>() / x.len() as f64;
        let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
        let mut sigma = mean.max(1e-6);
        // Method-of-moments: σ = ½ mean (1 + mean² / var), ξ = ½ (1 − mean² / var).
        let cv2 = var / (mean * mean).max(1e-30);
        let mut xi = 0.5 * (1.0 - 1.0 / cv2.max(1e-30));
        let mut best = gpd_ll(x, sigma, xi);
        for _ in 0..300 {
            let mut improved = false;
            for &d_sigma in &[0.05 * sigma, -0.05 * sigma] {
                for &d_xi in &[0.05_f64, -0.05_f64] {
                    let new_sigma = (sigma + d_sigma).max(1e-6);
                    let cand = gpd_ll(x, new_sigma, xi + d_xi);
                    if cand > best + 1e-9 {
                        sigma = new_sigma;
                        xi += d_xi;
                        best = cand;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
        Ok(Self { scale: sigma, shape: xi })
    }
}

fn ll(x: &[f64], mu: f64, sigma: f64, xi: f64) -> f64 {
    if sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let n = x.len();
    let mut acc = -(n as f64) * sigma.ln();
    for &v in x.iter() {
        let z = (v - mu) / sigma;
        if xi.abs() < 1e-12 {
            acc -= z + (-z).exp();
        } else {
            let arg = 1.0 + xi * z;
            if arg <= 0.0 {
                return f64::NEG_INFINITY;
            }
            acc -= (1.0 + 1.0 / xi) * arg.ln() + arg.powf(-1.0 / xi);
        }
    }
    acc
}

fn gpd_ll(x: &[f64], sigma: f64, xi: f64) -> f64 {
    if sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let n = x.len() as f64;
    let mut acc = -n * sigma.ln();
    for &v in x.iter() {
        if xi.abs() < 1e-12 {
            acc -= v / sigma;
        } else {
            let arg = 1.0 + xi * v / sigma;
            if arg <= 0.0 {
                return f64::NEG_INFINITY;
            }
            acc -= (1.0 + 1.0 / xi) * arg.ln();
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gev_cdf_and_quantile_are_inverses() {
        let d = Gev::new(0.0, 1.0, 0.2).unwrap();
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let q = d.quantile(p);
            let back = d.cdf(q);
            assert!(
                (back - p).abs() < 1e-6,
                "p={p} -> q={q} -> back={back}"
            );
        }
    }

    #[test]
    fn gpd_cdf_and_quantile_are_inverses() {
        let d = Gpd::new(1.0, 0.2).unwrap();
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let q = d.quantile(p);
            let back = d.cdf(q);
            assert!((back - p).abs() < 1e-6);
        }
    }

    #[test]
    fn gev_fit_returns_finite_parameters() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64 * 0.1).sin() + 3.0).collect();
        let m = Gev::fit(&x).unwrap();
        assert!(m.scale > 0.0);
        assert!(m.location.is_finite());
        assert!(m.shape.is_finite());
    }

    #[test]
    fn gpd_fit_returns_finite_parameters() {
        let x: Vec<f64> = (0..40).map(|i| 0.5 + (i as f64 * 0.05).abs()).collect();
        let m = Gpd::fit(&x).unwrap();
        assert!(m.scale > 0.0);
        assert!(m.shape.is_finite());
    }
}

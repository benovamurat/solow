//! Link functions `g` connecting the mean `μ` to the linear predictor `η = g(μ)`.
//!
//! Each link provides `link` (`g`), `inverse` (`g⁻¹`), and `deriv` (`dη/dμ = g'(μ)`),
//! which is all the IRLS estimator requires.

use solow_distributions::{norm_cdf, norm_pdf, norm_ppf};

/// Clamp a probability strictly inside `(0, 1)` to keep links finite.
fn clip01(p: f64) -> f64 {
    const E: f64 = 1e-12;
    p.clamp(E, 1.0 - E)
}

/// A link function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Link {
    /// `g(μ) = μ`
    Identity,
    /// `g(μ) = ln μ`
    Log,
    /// `g(μ) = ln(μ / (1 − μ))`
    Logit,
    /// `g(μ) = Φ⁻¹(μ)`
    Probit,
    /// `g(μ) = ln(−ln(1 − μ))`
    CLogLog,
    /// `g(μ) = 1 / μ`
    InversePower,
    /// `g(μ) = 1 / μ²`
    InverseSquared,
    /// `g(μ) = √μ`
    Sqrt,
}

impl Link {
    /// The link's display name, matching the reference's link class name
    /// (used as the `Link Function:` field of a GLM summary).
    pub fn name(&self) -> &'static str {
        match self {
            Link::Identity => "Identity",
            Link::Log => "Log",
            Link::Logit => "Logit",
            Link::Probit => "Probit",
            Link::CLogLog => "CLogLog",
            Link::InversePower => "InversePower",
            Link::InverseSquared => "InverseSquared",
            Link::Sqrt => "Sqrt",
        }
    }

    /// `η = g(μ)`.
    pub fn link(&self, mu: f64) -> f64 {
        match self {
            Link::Identity => mu,
            Link::Log => mu.ln(),
            Link::Logit => {
                let m = clip01(mu);
                (m / (1.0 - m)).ln()
            }
            Link::Probit => norm_ppf(clip01(mu)),
            Link::CLogLog => {
                let m = clip01(mu);
                (-(1.0 - m).ln()).ln()
            }
            Link::InversePower => 1.0 / mu,
            Link::InverseSquared => 1.0 / (mu * mu),
            Link::Sqrt => mu.sqrt(),
        }
    }

    /// `μ = g⁻¹(η)`.
    pub fn inverse(&self, eta: f64) -> f64 {
        match self {
            Link::Identity => eta,
            Link::Log => eta.exp(),
            Link::Logit => clip01(1.0 / (1.0 + (-eta).exp())),
            Link::Probit => clip01(norm_cdf(eta)),
            Link::CLogLog => clip01(1.0 - (-(eta.exp())).exp()),
            Link::InversePower => 1.0 / eta,
            Link::InverseSquared => eta.powf(-0.5),
            Link::Sqrt => eta * eta,
        }
    }

    /// `dη/dμ = g'(μ)`.
    pub fn deriv(&self, mu: f64) -> f64 {
        match self {
            Link::Identity => 1.0,
            Link::Log => 1.0 / mu,
            Link::Logit => {
                let m = clip01(mu);
                1.0 / (m * (1.0 - m))
            }
            Link::Probit => 1.0 / norm_pdf(norm_ppf(clip01(mu))),
            Link::CLogLog => {
                let m = clip01(mu);
                -1.0 / ((1.0 - m) * (1.0 - m).ln())
            }
            Link::InversePower => -1.0 / (mu * mu),
            Link::InverseSquared => -2.0 / (mu * mu * mu),
            Link::Sqrt => 0.5 / mu.sqrt(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn link_inverse_roundtrip() {
        let links = [
            Link::Identity,
            Link::Log,
            Link::Logit,
            Link::Probit,
            Link::CLogLog,
            Link::InversePower,
            Link::InverseSquared,
            Link::Sqrt,
        ];
        for l in links {
            for &mu in &[0.1f64, 0.3, 0.5, 0.8, 1.7, 3.2] {
                // Probit/logit/cloglog need mu in (0,1).
                let mu = match l {
                    Link::Logit | Link::Probit | Link::CLogLog => mu.min(0.95),
                    _ => mu,
                };
                let eta = l.link(mu);
                assert_abs_diff_eq!(l.inverse(eta), mu, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn deriv_matches_finite_difference() {
        let links = [Link::Log, Link::Logit, Link::Probit, Link::Sqrt];
        for l in links {
            for &mu in &[0.2, 0.4, 0.7] {
                let h = 1e-6;
                let fd = (l.link(mu + h) - l.link(mu - h)) / (2.0 * h);
                assert_abs_diff_eq!(l.deriv(mu), fd, epsilon = 1e-4);
            }
        }
    }
}

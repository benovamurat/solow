//! Exponential-dispersion families: variance function, unit deviance,
//! log-likelihood, and canonical/default link.

use crate::links::Link;
use solow_distributions::lgamma;

/// `y · ln(y/μ)` with the convention `0 · ln 0 = 0`.
fn xlogy_ratio(y: f64, mu: f64) -> f64 {
    if y == 0.0 {
        0.0
    } else {
        y * (y / mu).ln()
    }
}

/// An exponential-family error distribution for a GLM.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Family {
    /// Normal errors; `V(μ) = 1`.
    Gaussian,
    /// Bernoulli/binomial (binary) errors; `V(μ) = μ(1 − μ)`.
    Binomial,
    /// Poisson counts; `V(μ) = μ`.
    Poisson,
    /// Gamma errors; `V(μ) = μ²`.
    Gamma,
    /// Inverse-Gaussian errors; `V(μ) = μ³`.
    InverseGaussian,
    /// Negative binomial with dispersion `alpha`; `V(μ) = μ + α μ²`.
    NegativeBinomial { alpha: f64 },
}

impl Family {
    /// The family's display name, matching the reference's family class name
    /// (used as the `Model Family:` field of a GLM summary).
    pub fn name(&self) -> &'static str {
        match self {
            Family::Gaussian => "Gaussian",
            Family::Binomial => "Binomial",
            Family::Poisson => "Poisson",
            Family::Gamma => "Gamma",
            Family::InverseGaussian => "InverseGaussian",
            Family::NegativeBinomial { .. } => "NegativeBinomial",
        }
    }

    /// The canonical / default link used when none is specified.
    pub fn default_link(&self) -> Link {
        match self {
            Family::Gaussian => Link::Identity,
            Family::Binomial => Link::Logit,
            Family::Poisson => Link::Log,
            Family::Gamma => Link::InversePower,
            Family::InverseGaussian => Link::InverseSquared,
            Family::NegativeBinomial { .. } => Link::Log,
        }
    }

    /// Whether the dispersion (scale) is fixed at 1 (counts/binary).
    pub fn fixed_scale(&self) -> bool {
        matches!(
            self,
            Family::Binomial | Family::Poisson | Family::NegativeBinomial { .. }
        )
    }

    /// Variance function `V(μ)`.
    pub fn variance(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => 1.0,
            Family::Binomial => mu * (1.0 - mu),
            Family::Poisson => mu,
            Family::Gamma => mu * mu,
            Family::InverseGaussian => mu * mu * mu,
            Family::NegativeBinomial { alpha } => mu + alpha * mu * mu,
        }
    }

    /// Per-observation starting mean for IRLS.
    pub fn starting_mu(&self, y: f64, ybar: f64) -> f64 {
        match self {
            Family::Binomial => (y + 0.5) / 2.0,
            _ => (y + ybar) / 2.0,
        }
    }

    /// Unit deviance `d(y, μ)` (the per-observation deviance contribution).
    pub fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        match self {
            Family::Gaussian => (y - mu) * (y - mu),
            Family::Poisson => 2.0 * (xlogy_ratio(y, mu) - (y - mu)),
            Family::Binomial => 2.0 * (xlogy_ratio(y, mu) + xlogy_ratio(1.0 - y, 1.0 - mu)),
            Family::Gamma => 2.0 * (-(y / mu).ln() + (y - mu) / mu),
            Family::InverseGaussian => (y - mu) * (y - mu) / (y * mu * mu),
            Family::NegativeBinomial { alpha } => {
                let a = *alpha;
                2.0 * (xlogy_ratio(y, mu) - (y + 1.0 / a) * ((1.0 + a * y) / (1.0 + a * mu)).ln())
            }
        }
    }

    /// Total deviance `Σ d(yᵢ, μᵢ)`.
    pub fn deviance(&self, y: &[f64], mu: &[f64]) -> f64 {
        y.iter()
            .zip(mu)
            .map(|(&yi, &mi)| self.unit_deviance(yi, mi))
            .sum()
    }

    /// Per-observation log-likelihood `ℓ(yᵢ; μᵢ, scale)`.
    pub fn loglike_obs(&self, y: f64, mu: f64, scale: f64) -> f64 {
        use std::f64::consts::PI;
        match self {
            Family::Gaussian => -0.5 * ((y - mu).powi(2) / scale + (2.0 * PI * scale).ln()),
            Family::Poisson => y * mu.ln() - mu - lgamma(y + 1.0),
            Family::Binomial => y * mu.ln() + (1.0 - y) * (1.0 - mu).ln(),
            Family::Gamma => {
                let a = 1.0 / scale; // shape
                a * a.ln() - lgamma(a) + (a - 1.0) * y.ln() - a * (y / mu + mu.ln())
            }
            Family::InverseGaussian => {
                -0.5 * ((y - mu).powi(2) / (y * mu * mu * scale)
                    + (scale * 2.0 * PI * y.powi(3)).ln())
            }
            Family::NegativeBinomial { alpha } => {
                let a = *alpha;
                let lin = mu / (1.0 + a * mu);
                // ll = y ln(a μ /(1+a μ)) - (1/a) ln(1+a μ) + ln Γ(y+1/a) - ln Γ(1/a) - ln Γ(y+1)
                y * (a * lin).ln() - (1.0 / a) * (1.0 + a * mu).ln() + lgamma(y + 1.0 / a)
                    - lgamma(1.0 / a)
                    - lgamma(y + 1.0)
            }
        }
    }

    /// Total log-likelihood.
    pub fn loglike(&self, y: &[f64], mu: &[f64], scale: f64) -> f64 {
        y.iter()
            .zip(mu)
            .map(|(&yi, &mi)| self.loglike_obs(yi, mi, scale))
            .sum()
    }
}

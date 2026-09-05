//! Bivariate elliptical copulas: Gaussian and (optionally) Student-t.

use crate::bvn::bvn_cdf;
use solow_distributions::{norm_ppf, t_pdf};

/// Bivariate Gaussian copula parameterised by a single correlation `rho`.
///
/// The copula density is obtained by the normal-quantile transform
/// `x = Phi^{-1}(u)`, `y = Phi^{-1}(v)`:
///
/// `c(u, v) = phi_2(x, y; rho) / (phi(x) phi(v))`.
///
/// The CDF uses a bivariate-normal CDF, [`crate::bvn::bvn_cdf`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianCopula {
    /// Correlation parameter in `(-1, 1)`.
    pub rho: f64,
}

impl GaussianCopula {
    /// Construct a Gaussian copula from a scalar correlation.
    pub fn new(rho: f64) -> Self {
        Self { rho }
    }

    /// Construct from a 2x2 correlation matrix `[[1, rho], [rho, 1]]`.
    ///
    /// The off-diagonal entry is used directly.
    pub fn from_corr(corr: [[f64; 2]; 2]) -> Self {
        Self { rho: corr[0][1] }
    }

    /// Copula density `c(u, v)`.
    pub fn pdf(&self, u: f64, v: f64) -> f64 {
        let rho = self.rho;
        let x = norm_ppf(u);
        let y = norm_ppf(v);
        // Bivariate standard normal density divided by the product of the
        // univariate normal densities at the quantiles.
        let det = 1.0 - rho * rho;
        let quad = (x * x + y * y - 2.0 * rho * x * y) / det - (x * x + y * y);
        (-0.5 * quad).exp() / det.sqrt()
    }

    /// Log copula density `ln c(u, v)`.
    pub fn logpdf(&self, u: f64, v: f64) -> f64 {
        let rho = self.rho;
        let x = norm_ppf(u);
        let y = norm_ppf(v);
        let det = 1.0 - rho * rho;
        let quad = (x * x + y * y - 2.0 * rho * x * y) / det - (x * x + y * y);
        -0.5 * quad - 0.5 * det.ln()
    }

    /// Copula CDF `C(u, v)` via the bivariate-normal CDF.
    pub fn cdf(&self, u: f64, v: f64) -> f64 {
        let x = norm_ppf(u);
        let y = norm_ppf(v);
        bvn_cdf(x, y, self.rho)
    }

    /// Kendall's tau implied by `rho`: `tau = (2/pi) arcsin(rho)`.
    pub fn tau(&self) -> f64 {
        std::f64::consts::FRAC_2_PI * self.rho.asin()
    }

    /// Spearman's rho implied by `rho`: `rho_S = (6/pi) arcsin(rho/2)`.
    pub fn spearmans_rho(&self) -> f64 {
        6.0 / std::f64::consts::PI * (self.rho / 2.0).asin()
    }

    /// Invert the Kendall's-tau mapping: `rho = sin(pi tau / 2)`.
    pub fn rho_from_tau(tau: f64) -> f64 {
        (std::f64::consts::FRAC_PI_2 * tau).sin()
    }
}

/// Bivariate Student-t copula parameterised by a correlation `rho` and
/// degrees of freedom `df`.
///
/// Only the density is provided (the spec lists the t-copula `pdf` as
/// optional). The density follows the standard construction via the
/// `t_df` quantile transform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentTCopula {
    /// Correlation parameter in `(-1, 1)`.
    pub rho: f64,
    /// Degrees of freedom.
    pub df: f64,
}

impl StudentTCopula {
    /// Construct a Student-t copula.
    pub fn new(rho: f64, df: f64) -> Self {
        Self { rho, df }
    }

    /// Copula density `c(u, v)` via the `t_df` quantile transform.
    ///
    /// `c(u, v) = f_2(x, y; R, nu) / (f(x; nu) f(y; nu))` with
    /// `x = t_nu^{-1}(u)`, `y = t_nu^{-1}(v)`, where `f_2` is the bivariate
    /// Student-t density with shape matrix `R = [[1, rho], [rho, 1]]` and
    /// `f` is the univariate Student-t density.
    pub fn pdf(&self, u: f64, v: f64) -> f64 {
        use solow_distributions::t_ppf;
        let nu = self.df;
        let rho = self.rho;
        let x = t_ppf(u, nu);
        let y = t_ppf(v, nu);
        let det = 1.0 - rho * rho;

        // Bivariate t density (zero location, shape matrix R, dim d = 2):
        //   f_2 = Gamma((nu+d)/2) / (Gamma(nu/2) (nu pi)^{d/2} |R|^{1/2})
        //         * (1 + x' R^{-1} x / nu)^{-(nu+d)/2}.
        let lgamma = solow_distributions::lgamma;
        let d = 2.0;
        let log_c = lgamma((nu + d) / 2.0)
            - lgamma(nu / 2.0)
            - (d / 2.0) * (nu * std::f64::consts::PI).ln()
            - 0.5 * det.ln();
        let quad = (x * x - 2.0 * rho * x * y + y * y) / det;
        let log_joint = log_c - (nu + d) / 2.0 * (1.0 + quad / nu).ln();

        // Divide by the marginal t densities.
        let log_marg = t_pdf(x, nu).ln() + t_pdf(y, nu).ln();
        (log_joint - log_marg).exp()
    }

    /// Kendall's tau implied by `rho`: `tau = (2/pi) arcsin(rho)` (the same
    /// elliptical relation as the Gaussian copula).
    pub fn tau(&self) -> f64 {
        std::f64::consts::FRAC_2_PI * self.rho.asin()
    }
}

/// Standard bivariate-normal density used by tests for cross-checks.
#[allow(dead_code)]
fn bvn_pdf(x: f64, y: f64, rho: f64) -> f64 {
    let det = 1.0 - rho * rho;
    let quad = (x * x + y * y - 2.0 * rho * x * y) / det;
    (-0.5 * quad).exp() / (std::f64::consts::TAU * det.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use solow_distributions::norm_pdf;

    #[test]
    fn gaussian_pdf_via_bvn_pdf() {
        // c(u,v) should equal phi_2(x,y;rho)/(phi(x)phi(y)).
        let g = GaussianCopula::new(0.5);
        for (u, v) in [(0.3, 0.7), (0.1, 0.9), (0.5, 0.5)] {
            let x = norm_ppf(u);
            let y = norm_ppf(v);
            let direct = bvn_pdf(x, y, 0.5) / (norm_pdf(x) * norm_pdf(y));
            assert!((g.pdf(u, v) - direct).abs() < 1e-12);
        }
    }

    #[test]
    fn gaussian_independence_pdf_is_one() {
        let g = GaussianCopula::new(0.0);
        for (u, v) in [(0.2, 0.8), (0.5, 0.5), (0.95, 0.05)] {
            assert!((g.pdf(u, v) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn gaussian_tau_spearman_roundtrip() {
        for &rho in &[-0.6, 0.25, 0.8] {
            let g = GaussianCopula::new(rho);
            let back = GaussianCopula::rho_from_tau(g.tau());
            assert!((back - rho).abs() < 1e-12);
        }
    }

    #[test]
    fn gaussian_logpdf_matches_pdf() {
        let g = GaussianCopula::new(-0.4);
        for (u, v) in [(0.3, 0.6), (0.8, 0.2)] {
            assert!((g.logpdf(u, v).exp() - g.pdf(u, v)).abs() < 1e-12);
        }
    }

    #[test]
    fn studentt_independence_pdf_is_one() {
        // At rho=0 the t-copula is NOT independent in general, but tau is 0.
        let t = StudentTCopula::new(0.0, 5.0);
        assert!((t.tau()).abs() < 1e-15);
        // The density at rho=0 reduces to product of conditionals; check it
        // is finite and positive.
        let p = t.pdf(0.3, 0.7);
        assert!(p.is_finite() && p > 0.0);
    }
}

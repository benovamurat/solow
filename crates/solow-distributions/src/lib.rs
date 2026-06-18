//! # solow-distributions
//!
//! Special functions and the continuous distributions used for statistical
//! inference. Everything is implemented from scratch in pure Rust and validated
//! against an authoritative reference.
//!
//! - [`special`] — `lgamma`, `digamma`, incomplete beta/gamma (+ inverses), `erf`
//! - [`continuous`] — [`Normal`], [`StudentT`], [`FDist`], [`ChiSquared`] and the
//!   matching free functions (`norm_cdf`, `t_sf`, `f_sf`, `chi2_sf`, …)

pub mod continuous;
pub mod continuous_ext;
pub mod discrete;
pub mod empirical;
pub mod special;

pub use continuous::{
    chi2_cdf, chi2_isf, chi2_pdf, chi2_ppf, chi2_sf, f_cdf, f_isf, f_pdf, f_ppf, f_sf, norm_cdf,
    norm_isf, norm_pdf, norm_ppf, norm_sf, t_cdf, t_isf, t_pdf, t_ppf, t_sf, ChiSquared, FDist,
    Normal, StudentT,
};
pub use continuous_ext::{
    Beta, Cauchy, Exponential, Gamma, Laplace, LogNormal, Logistic, Pareto, Uniform, WeibullMin,
};
pub use discrete::{Binomial, Geometric, NegativeBinomial, Poisson};
pub use empirical::{Ecdf, Side, StepFunction};
pub use special::{
    betainc, betaincinv, digamma, erf, erfc, erfinv, gamma, gammainc, gammaincc, gammaincinv,
    lbeta, lgamma,
};

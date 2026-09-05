//! # solow-copula
//!
//! Bivariate copulas for the Solow statistical library.
//!
//! Two families are provided:
//!
//! * **Archimedean** copulas in closed form: [`ClaytonCopula`],
//!   [`FrankCopula`], and [`GumbelCopula`]. Each exposes `cdf(u, v)`,
//!   `pdf(u, v)`, and the Kendall's-tau mapping `tau()`.
//! * **Elliptical** copulas: [`GaussianCopula`] (with closed-form `pdf`
//!   via the normal-quantile transform, an analytic `tau`/`spearmans_rho`,
//!   and a `cdf` built on a bivariate-normal CDF) and an optional
//!   [`StudentTCopula`] `pdf`.
//!
//! The free functions [`kendalls_tau`] and [`spearmans_rho`] compute the
//! sample rank correlations for paired data.
//!
//! All quantities are cross-validated against the canonical Python
//! reference (`distributions.copula.api`) in `tests/reference.rs`.

mod archimedean;
mod bvn;
mod elliptical;
mod rank;

pub use archimedean::{ClaytonCopula, FrankCopula, GumbelCopula};
pub use bvn::bvn_cdf;
pub use elliptical::{GaussianCopula, StudentTCopula};
pub use rank::{kendalls_tau, spearmans_rho};

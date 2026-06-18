//! # solow-gam
//!
//! Generalized additive models with penalized B-spline smooth terms.
//!
//! [`BSplines`] builds a quantile-knot B-spline basis for a single covariate
//! together with a curvature ([`BSplines::cov_der2`]) penalty. [`GlmGam`] fits
//! a GLM with one such smooth term plus an intercept by penalized iteratively
//! reweighted least squares (P-IRLS) at a fixed smoothing parameter `alpha`,
//! exposing the parameters, fitted values, effective degrees of freedom, and
//! penalized deviance. Validated against an authoritative reference.
//!
//! ```
//! use ndarray::Array1;
//! use solow_gam::GlmGam;
//! use solow_glm::Family;
//!
//! let x = Array1::linspace(0.0, 1.0, 60);
//! let y = x.mapv(|xi| (2.0 * std::f64::consts::PI * xi).sin());
//! let res = GlmGam::new(y, &x, 10, 3, 0.6, Family::Gaussian)
//!     .unwrap()
//!     .fit()
//!     .unwrap();
//! assert!(res.converged);
//! assert!(res.edf_total > 0.0);
//! ```

mod bspline;
mod cyclic;
mod gam;
mod gam_ext;

pub use bspline::BSplines;
pub use cyclic::CyclicCubicSplines;
pub use gam::{GamResults, GlmGam};
pub use gam_ext::{GamExtResults, GlmGamExt};

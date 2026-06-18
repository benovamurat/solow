//! # solow-mixed
//!
//! Linear mixed-effects models (`MixedLM`) estimated by restricted maximum
//! likelihood (REML) or maximum likelihood (ML).
//!
//! The model is
//!
//! ```text
//! y = X·β + Z·b + ε,    b ~ N(0, Ψ),    ε ~ N(0, σ²·I)
//! ```
//!
//! where observations are partitioned into independent groups and the random
//! effects `b` are shared within a group. This crate implements the
//! **random-intercept** model: a single grouping factor with one random effect
//! per group (`Z` is a column of ones within each group), so `Ψ = ψ·σ²` is a
//! scalar variance.
//!
//! Estimation profiles out the fixed effects `β` (by generalized least squares)
//! and the residual scale `σ²` (closed form), leaving a one-dimensional profile
//! objective over the covariance ratio `ψ` that is maximized numerically. The
//! formulation follows the canonical `MixedLM` REML/ML profile log-likelihood.

mod model;

pub use model::{MixedLm, MixedLmResults, RemlMethod};

//! # solow-optimize
//!
//! Unconstrained optimization and numerical differentiation for maximum-likelihood
//! estimation: [`newton_stationary`] (analytic score/Hessian), [`minimize_bfgs`]
//! (gradient only), and finite-difference [`approx_fprime`] / [`approx_hess`].

mod numdiff;
mod optimizer;

pub use numdiff::{approx_fprime, approx_hess};
pub use optimizer::{minimize_bfgs, newton_stationary, OptimizeResult};

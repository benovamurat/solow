//! # solow-gp
//!
//! Gaussian-process regression and classification — the reference
//! `gaussian_process` family.
//!
//! * [`kernels`] — a small algebra of kernels: `RBF`, `Matern`, `RationalQuadratic`,
//!   `ConstantKernel`, `WhiteKernel`, `DotProduct`, and the sum, product,
//!   and exponentiation composers.
//! * [`GaussianProcessRegressor`] — GP regression with an optional
//!   `WhiteKernel`-style additive noise term, Cholesky-based inference,
//!   and log-marginal-likelihood evaluation.
//! * [`GaussianProcessClassifier`] — Laplace-approximate GP binary
//!   classifier (Rasmussen-Williams Ch. 3).
//!
//! # References
//!
//! * Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes
//!   for Machine Learning.* MIT Press.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod classifier;
pub mod kernels;
pub mod regressor;

pub use classifier::GaussianProcessClassifier;
pub use regressor::GaussianProcessRegressor;

/// Commonly-used imports.
pub mod prelude {
    pub use crate::classifier::GaussianProcessClassifier;
    pub use crate::kernels::{
        ConstantKernel, DotProduct, Exponentiation, Kernel, Matern, Product,
        RationalQuadratic, Rbf, Sum, WhiteKernel,
    };
    pub use crate::regressor::GaussianProcessRegressor;
}

//! # solow-impute
//!
//! The deterministic core of multiple imputation by chained equations (MICE).
//!
//! Multiple imputation has two ingredients: a stochastic step that draws
//! plausible replacements for missing values, and a deterministic step that
//! combines the analyses of the completed data sets back into a single
//! inference. This crate implements the deterministic pieces, which are exactly
//! reproducible and can be validated against an authoritative reference:
//!
//! * [`combine`] applies Rubin's combining rules to per-imputation parameter
//!   estimates and covariance matrices, returning the pooled estimate, the
//!   within/between/total covariance, the fraction of missing information, and
//!   the Barnard–Rubin degrees of freedom.
//! * [`conditional_mean_impute`] performs a single deterministic
//!   conditional-mean (regression) imputation of one variable given the others.
//!
//! The stochastic posterior draws that drive a full MICE run depend on a random
//! number generator and are intentionally out of scope here.
//!
//! ```
//! use ndarray::array;
//! use solow_impute::combine;
//!
//! let p1 = array![1.0, 2.0];
//! let p2 = array![1.2, 1.8];
//! let c1 = array![[0.04, 0.0], [0.0, 0.05]];
//! let c2 = array![[0.05, 0.0], [0.0, 0.06]];
//! let res = combine(&[p1, p2], &[c1, c2], f64::INFINITY).unwrap();
//! assert!((res.params[0] - 1.1).abs() < 1e-12);
//! ```

mod combine;
mod regression;

pub use combine::{combine, CombinedEstimate};
pub use regression::{conditional_mean_impute, ConditionalMeanImputation};

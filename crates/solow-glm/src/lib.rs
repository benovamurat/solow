//! # solow-glm
//!
//! Generalized linear models: exponential-dispersion [`Family`] distributions,
//! [`Link`] functions, and the [`Glm`] estimator (iteratively reweighted least
//! squares). Validated against an authoritative reference.
//!
//! ```
//! use ndarray::{array, Array1};
//! use solow_glm::{Family, Glm};
//!
//! // Poisson regression with an intercept.
//! let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
//! let y: Array1<f64> = array![1.0, 2.0, 4.0, 7.0, 12.0];
//! let res = Glm::new(y, x, Family::Poisson).unwrap().fit().unwrap();
//! assert!(res.converged);
//! ```

mod family;
mod glm;
mod links;
mod tweedie;

pub use family::Family;
pub use glm::{Glm, GlmResults};
pub use links::Link;
pub use tweedie::TweedieGlm;

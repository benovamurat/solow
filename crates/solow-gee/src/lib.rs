//! # solow-gee
//!
//! Generalized estimating equations (GEE) for clustered / longitudinal data.
//! Mean parameters are estimated by Fisher scoring on the estimating
//! equations under a *working* within-cluster correlation ([`CovStruct`]);
//! inference uses the cluster-robust sandwich covariance (with a model-based
//! "naive" covariance also exposed).  Supports the Gaussian, Poisson, and
//! Binomial families with Independence and Exchangeable working correlation.
//! Validated against an authoritative reference.
//!
//! ```
//! use ndarray::array;
//! use solow_gee::{CovStruct, Gee};
//! use solow_glm::Family;
//!
//! let x = array![
//!     [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
//!     [1.0, 3.0], [1.0, 4.0], [1.0, 5.0],
//! ];
//! let y = array![1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
//! let groups = [0i64, 0, 1, 1, 2, 2];
//! let res = Gee::new(y, x, &groups, Family::Poisson, CovStruct::Exchangeable)
//!     .unwrap()
//!     .fit()
//!     .unwrap();
//! assert!(res.converged);
//! ```

mod categorical;
mod gee;

pub use categorical::{CategoricalCov, CategoricalGeeResults, NominalGee, OrdinalGee};
pub use gee::{CovStruct, Gee, GeeResults};

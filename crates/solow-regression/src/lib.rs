//! # solow-regression
//!
//! Linear regression models estimated by least squares, with the full standard
//! battery of results and inference statistics. Validated against an
//! authoritative reference.
//!
//! ```
//! use ndarray::{array, Array1};
//! use solow_core::tools::{add_constant, HasConstant};
//! use solow_regression::LinearModel;
//!
//! let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
//! let y: Array1<f64> = array![1.1, 1.9, 3.2, 3.9, 5.1];
//! let design = add_constant(&x, true, HasConstant::Add).unwrap();
//! let res = LinearModel::ols(y, design).unwrap().fit().unwrap();
//! assert!((res.rsquared - 0.997).abs() < 0.01);
//! ```

mod dimred;
mod glsar;
mod linear;
mod quantile;
mod recursive;
mod robustcov;
mod rolling;

pub use dimred::{SirResults, SlicedInverseReg};
pub use glsar::{Glsar, GlsarResults};
pub use linear::{LinearModel, LinearResults};
pub use quantile::{QuantReg, QuantRegResults};
pub use recursive::{RecursiveLS, RecursiveLSResults};
pub use robustcov::{bse_from_cov, robust_cov, CovType};
pub use rolling::{RollingOLS, RollingOLSResults};

//! # solow-core
//!
//! Foundational types shared across the Solow statistical computing stack:
//! a common [`Error`] type, numeric aliases, and shared data-handling tools.

pub mod error;
pub mod tools;

pub use error::{Error, Result};

use ndarray::{Array1, Array2};

/// A dense column of `f64` observations.
pub type Vector = Array1<f64>;
/// A dense `f64` matrix (rows = observations, columns = variables).
pub type Matrix = Array2<f64>;

/// Commonly used imports.
pub mod prelude {
    pub use crate::error::{Error, Result};
    pub use crate::tools::{
        add_constant, add_constant_default, ensure_all_finite, ensure_all_finite_2d, HasConstant,
    };
    pub use crate::{Matrix, Vector};
}

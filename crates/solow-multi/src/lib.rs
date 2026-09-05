//! # solow-multi
//!
//! Multi-class and multi-output meta-estimators — the reference
//! `multiclass` and `multioutput` families.
//!
//! The trait system is intentionally light: a binary base classifier
//! ([`traits::BinaryClassifier`]) can be lifted to a multi-class one via
//! [`OneVsRestClassifier`], [`OneVsOneClassifier`], or
//! [`OutputCodeClassifier`], and a scalar regressor
//! ([`traits::Regressor`]) can be lifted to multi-output via
//! [`MultiOutputRegressor`] / [`RegressorChain`].
//!
//! # References
//!
//! * Rifkin, R., & Klautau, A. (2004). *In Defense of One-vs-All
//!   Classification.* JMLR 5, 101-141.
//! * Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011).
//!   *Classifier Chains for Multi-label Classification.* Machine
//!   Learning 85, 333-359.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod chains;
pub mod multi_output;
pub mod one_vs_one;
pub mod one_vs_rest;
pub mod output_code;
pub mod traits;

pub use chains::{ClassifierChain, RegressorChain};
pub use multi_output::{MultiOutputClassifier, MultiOutputRegressor};
pub use one_vs_one::OneVsOneClassifier;
pub use one_vs_rest::OneVsRestClassifier;
pub use output_code::OutputCodeClassifier;
pub use traits::{BinaryClassifier, MultiClassifier, Regressor};

/// Commonly-used imports.
pub mod prelude {
    pub use crate::chains::{ClassifierChain, RegressorChain};
    pub use crate::multi_output::{MultiOutputClassifier, MultiOutputRegressor};
    pub use crate::one_vs_one::OneVsOneClassifier;
    pub use crate::one_vs_rest::OneVsRestClassifier;
    pub use crate::output_code::OutputCodeClassifier;
    pub use crate::traits::{BinaryClassifier, MultiClassifier, Regressor};
}

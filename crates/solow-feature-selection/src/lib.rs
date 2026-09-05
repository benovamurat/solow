//! # solow-feature-selection
//!
//! Univariate and model-based feature-selection routines.
//!
//! * [`VarianceThreshold`] — drops columns whose variance falls below
//!   a caller-specified floor (the classical "constant-features" filter).
//! * [`SelectKBest`] — keeps the `k` columns with the highest score
//!   under a caller-supplied scoring function; ships with
//!   [`score_f_classif`] (one-way ANOVA F-statistic per feature) and
//!   [`score_f_regression`] (correlation-based F-statistic against a
//!   continuous target).
//! * [`Rfe`] — Recursive Feature Elimination (Guyon et al. 2002)
//!   around a caller-supplied ranker that maps `(x, y)` to a
//!   per-feature importance vector.
//!
//! All selectors expose `fit` / `transform` / `fit_transform` and
//! report the selected column indices. Transformations preserve
//! column order relative to the input.
//!
//! # References
//!
//! * Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). *Gene
//!   selection for cancer classification using support vector machines.*
//!   Machine Learning 46(1-3), 389-422.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod filter;
pub mod percentile;
pub mod rfe;
pub mod scores;
pub mod sequential;

pub use filter::{SelectKBest, VarianceThreshold};
pub use percentile::{SelectFdr, SelectFpr, SelectFwe, SelectPercentile};
pub use rfe::Rfe;
pub use scores::{score_f_classif, score_f_regression};
pub use sequential::{SequentialFeatureSelector, SfsDirection};

/// Commonly-used imports.
pub mod prelude {
    pub use crate::filter::{SelectKBest, VarianceThreshold};
    pub use crate::percentile::{SelectFdr, SelectFpr, SelectFwe, SelectPercentile};
    pub use crate::rfe::Rfe;
    pub use crate::scores::{score_f_classif, score_f_regression};
    pub use crate::sequential::{SequentialFeatureSelector, SfsDirection};
}

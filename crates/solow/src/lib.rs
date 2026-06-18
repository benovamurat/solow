//! # Solow
//!
//! A complete statistical modeling, econometrics, and data-visualization toolkit.
//!
//! This umbrella crate re-exports the full public API. Most users can work
//! entirely through [`prelude`].

pub use solow_bayes as bayes;
pub use solow_copula as copula;
pub use solow_core as core;
pub use solow_discrete as discrete;
pub use solow_distributions as distributions;
pub use solow_duration as duration;
pub use solow_emplike as emplike;
pub use solow_fit as fit;
pub use solow_formula as formula;
pub use solow_gam as gam;
pub use solow_gee as gee;
pub use solow_glm as glm;
pub use solow_graphics as graphics;
pub use solow_impute as impute;
pub use solow_linalg as linalg;
pub use solow_mixed as mixed;
pub use solow_multivariate as multivariate;
pub use solow_nonparametric as nonparametric;
pub use solow_optimize as optimize;
pub use solow_othermod as othermod;
pub use solow_regime as regime;
pub use solow_regression as regression;
pub use solow_robust as robust;
pub use solow_statespace as statespace;
pub use solow_stats as stats;
pub use solow_summary as summary;
pub use solow_tsa as tsa;
pub use solow_var as var;
pub use solow_viz as viz;

pub use ndarray;

/// The recommended glob-import for day-to-day use.
pub mod prelude {
    pub use ndarray::{array, Array1, Array2, Axis};
    pub use solow_core::prelude::*;
}

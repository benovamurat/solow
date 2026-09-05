//! # solow-pipeline
//!
//! Composable preprocessing / estimator pipelines and hyperparameter
//! search, tuned to the Rust idioms of the Solow stack.
//!
//! Where the reference relies on Python's duck typing, this crate
//! uses **erased closures** — every stage is a boxed function taking
//! the current matrix and returning the next matrix (for
//! transformers) or the final scalar score (for the terminal
//! scorer). This keeps the type surface small while allowing any
//! Solow estimator to plug in.
//!
//! Two workhorses ship here:
//!
//! * [`GridSearchCV`] — full-factorial grid search over an arbitrary
//!   parameter grid, scored by a user-supplied CV callback. Records
//!   every point's mean CV score and standard deviation.
//! * [`RandomizedSearchCV`] — samples `n_iter` parameter dictionaries
//!   uniformly at random from a caller-supplied `samplers` list under
//!   a portable MMIX-LCG PRNG (bit-for-bit reproducible under a seed).
//!
//! Both accept CV folds from [`solow-cv`](https://docs.rs/solow-cv)
//! so the whole tuning loop remains deterministic when the folds and
//! the scorer are.
//!
//! ```
//! use solow_cv::KFold;
//! use solow_pipeline::{GridSearchCV, ParamGrid};
//!
//! # fn score(_: &[usize], _: &[usize], _p: &std::collections::BTreeMap<String, f64>) -> Result<f64, solow_core::Error> { Ok(0.0) }
//! let kf = KFold::new(3)?;
//! let grid = ParamGrid::default()
//!     .add("alpha", vec![0.01, 0.1, 1.0])
//!     .add("l1_ratio", vec![0.0, 0.5, 1.0]);
//! let gs = GridSearchCV::run(&kf, 30, grid, score)?;
//! println!("best alpha = {}", gs.best_params["alpha"]);
//! # Ok::<_, solow_core::Error>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod composite;
pub mod halving;
pub mod pipeline;
pub mod search;
pub mod target;

pub use composite::{ColumnTransformer, ColumnTransformerStep, FeatureUnion, FeatureUnionStep};
pub use halving::{HalvingConfig, HalvingGridSearchCV, HalvingRandomSearchCV};
pub use pipeline::{Pipeline, Step};
pub use search::{GridSearchCV, ParamGrid, RandomizedSearchCV, SearchResult};
pub use target::TransformedTargetRegressor;

/// Commonly-used imports.
pub mod prelude {
    pub use crate::halving::{HalvingConfig, HalvingGridSearchCV, HalvingRandomSearchCV};
    pub use crate::pipeline::{Pipeline, Step};
    pub use crate::search::{GridSearchCV, ParamGrid, RandomizedSearchCV, SearchResult};
    pub use crate::target::TransformedTargetRegressor;
}

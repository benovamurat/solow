//! # solow-cv
//!
//! Cross-validation, resampling, and bootstrap for the Solow statistical
//! stack.
//!
//! The crate provides three composable layers:
//!
//! 1. **Splitters** — [`KFold`], [`StratifiedKFold`], [`TimeSeriesSplit`],
//!    [`LeaveOneOut`], and [`ShuffleSplit`] behind the shared [`Splitter`]
//!    trait. Every splitter emits validated `Vec<Split>` fold lists that a
//!    caller can iterate, index, or run in parallel.
//!
//! 2. **Cross-validated scoring** — [`cross_val_score`] runs a
//!    `fit_and_score` callback across every fold and returns the per-fold
//!    scores with mean, unbiased standard deviation, and standard error.
//!
//! 3. **Bootstrap confidence intervals** — [`bootstrap_ci`] resamples with
//!    replacement and returns a percentile, reverse-percentile ("basic"), or
//!    Efron BCa confidence interval for any scalar statistic of a sample.
//!
//! ```
//! use ndarray::array;
//! use solow_cv::{KFold, Splitter};
//!
//! let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
//! let kf = KFold::new(5)?.shuffle(false);
//! for split in kf.split(y.len())? {
//!     let (train, test) = (split.train, split.test);
//!     assert_eq!(train.len() + test.len(), y.len());
//!     // Fit on `y.select(Axis(0), &train)`; score on the test rows.
//! }
//! # Ok::<_, solow_core::Error>(())
//! ```
//!
//! ## Determinism
//!
//! Random splits and bootstrap resamples use a well-tested 64-bit
//! linear-congruential generator (Numerical Recipes' MMIX constants),
//! seeded by the caller. The stream is bit-for-bit portable across
//! platforms; a fixed seed always produces the same folds and replicates.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod block_bootstrap;
mod bootstrap;
mod extras;
mod group_splitters;
mod scoring;
mod shuffle;
mod splitters;

pub use block_bootstrap::{
    circular_block_bootstrap_indices, moving_block_bootstrap_indices,
    stationary_bootstrap_indices,
};
pub use bootstrap::{bootstrap_ci, BootstrapCi, BootstrapMethod};
pub use extras::{
    learning_curve, permutation_test_score, validation_curve, LeavePOut, RepeatedKFold,
    RepeatedStratifiedKFold,
};
pub use group_splitters::{
    CombinatorialPurgedKFold, GroupKFold, PurgedKFold, StratifiedGroupKFold,
};
pub use shuffle::{GroupShuffleSplit, StratifiedShuffleSplit};
#[cfg(feature = "parallel")]
pub use scoring::cross_val_score_parallel;
pub use scoring::{cross_val_score, cross_val_score_from_folds, CrossValScores};
pub use splitters::{
    KFold, LeaveOneOut, ShuffleSplit, Split, Splitter, StratifiedKFold, TimeSeriesSplit,
};

/// Commonly used imports.
pub mod prelude {
    pub use crate::{
        bootstrap_ci, cross_val_score, cross_val_score_from_folds, learning_curve,
        permutation_test_score, validation_curve, BootstrapCi, BootstrapMethod,
        CombinatorialPurgedKFold, CrossValScores, GroupKFold, GroupShuffleSplit, KFold,
        LeaveOneOut, LeavePOut, PurgedKFold, RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit,
        Split, Splitter, StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit,
        TimeSeriesSplit,
    };
}

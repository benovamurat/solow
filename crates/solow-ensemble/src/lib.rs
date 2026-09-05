//! # solow-ensemble
//!
//! Ensemble estimators for the Solow statistical stack.
//!
//! * [`RandomForestClassifier`] / [`RandomForestRegressor`] —
//!   Breiman (2001) bagging of CART trees with per-node random
//!   feature-subset selection (`max_features`).
//! * [`GradientBoostingRegressor`] — Friedman (2001) stagewise
//!   additive regression with a fixed learning rate and per-stage
//!   least-squares CART trees.
//! * [`AdaBoostClassifier`] — Freund-Schapire (1997) SAMME.R additive
//!   real-valued AdaBoost for two-class problems.
//! * [`IsolationForest`] — Liu-Ting-Zhou (2008) unsupervised anomaly
//!   detector built from randomised isolation trees. Reports a
//!   normalised anomaly score in `[0, 1]`.
//!
//! All ensembles are deterministic under a caller-supplied seed via a
//! portable MMIX-LCG. Fitting a bag of `T` trees on `n` samples of
//! `d` features is `O(T · n · d · log n)` — the classical bagging
//! bound.
//!
//! # References
//!
//! * Breiman, L. (2001). *Random Forests.* Machine Learning 45(1),
//!   5-32.
//! * Friedman, J. H. (2001). *Greedy function approximation: A
//!   gradient boosting machine.* Annals of Statistics 29(5),
//!   1189-1232.
//! * Freund, Y., & Schapire, R. E. (1997). *A decision-theoretic
//!   generalization of on-line learning and an application to boosting.*
//!   Journal of Computer and System Sciences 55(1), 119-139.
//! * Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation
//!   forest.* ICDM 2008, 413-422.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod adaboost;
pub mod adaboost_regressor;
pub mod bagging;
pub mod extra_trees;
pub mod gradient_boosting;
pub mod gradient_boosting_classifier;
pub mod hist_gradient_boosting;
pub mod isolation_forest;
pub mod random_forest;
pub mod stacking;
pub mod voting;

pub use adaboost::AdaBoostClassifier;
pub use adaboost_regressor::AdaBoostRegressor;
pub use bagging::{BaggingClassifier, BaggingRegressor};
pub use extra_trees::{ExtraTreesClassifier, ExtraTreesRegressor};
pub use gradient_boosting::GradientBoostingRegressor;
pub use gradient_boosting_classifier::GradientBoostingClassifier;
pub use hist_gradient_boosting::{
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
};
pub use isolation_forest::IsolationForest;
pub use random_forest::{RandomForestClassifier, RandomForestRegressor};
pub use stacking::{StackingClassifier, StackingRegressor};
pub use voting::{VotingClassifier, VotingMode, VotingRegressor};

/// Commonly-used imports.
pub mod prelude {
    pub use crate::adaboost::AdaBoostClassifier;
    pub use crate::bagging::{BaggingClassifier, BaggingRegressor};
    pub use crate::extra_trees::{ExtraTreesClassifier, ExtraTreesRegressor};
    pub use crate::gradient_boosting::GradientBoostingRegressor;
    pub use crate::hist_gradient_boosting::{
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    };
    pub use crate::isolation_forest::IsolationForest;
    pub use crate::random_forest::{RandomForestClassifier, RandomForestRegressor};
    pub use crate::stacking::{StackingClassifier, StackingRegressor};
    pub use crate::voting::{VotingClassifier, VotingMode, VotingRegressor};
}

// ---------------------------------------------------------------------------
// Shared deterministic PRNG (MMIX 64-bit LCG)
// ---------------------------------------------------------------------------

pub(crate) fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

pub(crate) fn uniform_index(state: &mut u64, n: u64) -> usize {
    let max = u64::MAX - (u64::MAX % n);
    loop {
        let r = lcg_next(state);
        if r < max {
            return (r % n) as usize;
        }
    }
}

pub(crate) fn uniform_f64(state: &mut u64) -> f64 {
    (lcg_next(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

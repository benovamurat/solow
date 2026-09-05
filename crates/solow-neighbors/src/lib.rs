//! # solow-neighbors
//!
//! Nearest-neighbor data structures and estimators for the Solow
//! statistical stack.
//!
//! The crate ships a **KDTree** — the balanced k-dimensional binary
//! space partition tree — for `O(log n)` amortised nearest-neighbour
//! and radius queries in low-to-moderate dimensions, together with two
//! estimators that consume it:
//!
//! * [`KNeighborsClassifier`] — majority-vote classification over the
//!   `k` nearest labelled neighbours, with optional distance-weighted
//!   voting (uniform vs `1/d`).
//! * [`KNeighborsRegressor`] — mean of the `k` nearest neighbour target
//!   values (uniform or distance-weighted).
//!
//! Every estimator exposes the classical the reference `fit` /
//! `predict` / `predict_proba` shape. The tree is built once at fit
//! time; predict on `m` new points is `O(m · k · log n)` in expectation.
//!
//! ## Complexity
//!
//! * [`KdTree::build`]: `O(n · log n)` time, `O(n)` space, using the
//!   median-split classical construction with an in-place bounding-box
//!   pruning key.
//! * [`KdTree::k_nearest`]: `O(k · log n)` expected time in low `d`,
//!   degrades to `O(n)` as `d` grows (the classical KD-tree curse of
//!   dimensionality).
//! * [`KdTree::radius`]: `O(log n + m)` where `m` is the answer size.
//!
//! For high-dimensional data (`d ≥ 30`) prefer brute-force scanning —
//! the KDTree still returns the correct answer but no longer beats an
//! `O(n · d)` sweep.
//!
//! ## Determinism
//!
//! Construction is fully deterministic: the median split picks the
//! lower-index sample on ties, so `KdTree::build` on the same input
//! always produces the same tree. Queries are deterministic given the
//! tree.
//!
//! ## References
//!
//! * Bentley, J. L. (1975). *Multidimensional binary search trees used
//!   for associative searching.* Communications of the ACM, 18(9),
//!   509-517.
//! * Friedman, J. H., Bentley, J. L., & Finkel, R. A. (1977). *An
//!   algorithm for finding best matches in logarithmic expected time.*
//!   ACM Transactions on Mathematical Software, 3(3), 209-226.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod ball_tree;
pub mod centroid;
pub mod kde;
pub mod kdtree;
pub mod knn;
pub mod lof;
pub mod radius;

pub use ball_tree::BallTree;
pub use centroid::NearestCentroid;
pub use kde::{KdeKernel, KernelDensity};
pub use kdtree::{KdTree, Neighbor};
pub use knn::{KNeighborsClassifier, KNeighborsRegressor, WeightKind};
pub use lof::LocalOutlierFactor;
pub use radius::{RadiusNeighborsClassifier, RadiusNeighborsRegressor};

/// Commonly-used imports.
pub mod prelude {
    pub use crate::ball_tree::BallTree;
    pub use crate::centroid::NearestCentroid;
    pub use crate::kde::{KdeKernel, KernelDensity};
    pub use crate::kdtree::{KdTree, Neighbor};
    pub use crate::knn::{KNeighborsClassifier, KNeighborsRegressor, WeightKind};
    pub use crate::lof::LocalOutlierFactor;
    pub use crate::radius::{RadiusNeighborsClassifier, RadiusNeighborsRegressor};
}

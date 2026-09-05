//! # solow-tree
//!
//! Decision-tree learners for the Solow statistical stack.
//!
//! Implements the **CART** (Classification And Regression Trees;
//! Breiman-Friedman-Olshen-Stone 1984) algorithm in its axis-aligned
//! binary-split form:
//!
//! * [`DecisionTreeClassifier`] with `Gini` or `Entropy` splitters,
//!   returning class-probability leaves.
//! * [`DecisionTreeRegressor`] with `Mse` or `Mae` splitters, returning
//!   mean-response leaves.
//!
//! # Split selection
//!
//! At each internal node the learner scans every feature and every
//! order-statistic-derived split threshold, choosing the split that
//! maximises impurity decrease (Breiman 1984, §2.4):
//!
//! ```text
//! ΔI(t) = I(t) − (N_L / N_t) · I(t_L) − (N_R / N_t) · I(t_R)
//! ```
//!
//! Ties on impurity gain break on the smaller feature index and then
//! the smaller threshold, so a fit is bit-for-bit reproducible on a
//! given input.
//!
//! # Complexity
//!
//! Sorted-scan CART is `O(n · d · log n)` per level with `O(log n)`
//! expected tree depth on balanced targets; the worst case is `O(n²)`.
//! Space is `O(n · d)` during the fit and `O(nodes)` for the fitted
//! tree.
//!
//! # References
//!
//! * Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984).
//!   *Classification and Regression Trees.* Wadsworth.
//! * Quinlan, J. R. (1986). *Induction of decision trees.*
//!   Machine Learning 1(1), 81-106.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod classifier;
pub mod extra;
pub mod regressor;
pub(crate) mod tree;

pub use classifier::{ClassificationCriterion, DecisionTreeClassifier};
pub use extra::{ExtraTreeClassifier, ExtraTreeRegressor};
pub use regressor::{DecisionTreeRegressor, RegressionCriterion};
pub use tree::{Node, TreeParams};

/// Commonly-used imports.
pub mod prelude {
    pub use crate::classifier::{ClassificationCriterion, DecisionTreeClassifier};
    pub use crate::extra::{ExtraTreeClassifier, ExtraTreeRegressor};
    pub use crate::regressor::{DecisionTreeRegressor, RegressionCriterion};
    pub use crate::tree::TreeParams;
}

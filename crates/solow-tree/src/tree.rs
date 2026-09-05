//! Shared tree structure and hyperparameter bundle.
//!
//! Trees are stored as an arena of [`Node`]s addressed by `usize`
//! indices — this keeps the traversal cache-friendly and avoids the
//! recursion-limit worry of a boxed-pointer representation on very
//! deep trees.

use solow_core::{Error, Result};

/// A single decision-tree node.
///
/// Internal nodes carry a split predicate `feature <= threshold`;
/// leaves carry a prediction (per-class probability for classifiers,
/// mean response for regressors).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// `Some(idx)` for an internal node; `None` for a leaf.
    pub feature: Option<usize>,
    /// Split threshold; ignored for leaves.
    pub threshold: f64,
    /// Left-child arena index; ignored for leaves.
    pub left: usize,
    /// Right-child arena index; ignored for leaves.
    pub right: usize,
    /// Impurity at this node before the split.
    pub impurity: f64,
    /// Number of training samples reaching this node.
    pub n_samples: usize,
    /// Leaf prediction: class-probability vector (classifier) or a
    /// single mean response (regressor, stored in `value[0]`).
    pub value: Vec<f64>,
}

impl Node {
    /// Is this a leaf?
    pub fn is_leaf(&self) -> bool {
        self.feature.is_none()
    }
}

/// Growth parameters shared by classifier and regressor.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TreeParams {
    /// Maximum tree depth (root is depth 0). `usize::MAX` = unlimited.
    pub max_depth: usize,
    /// Minimum samples required to split an internal node. Default 2.
    pub min_samples_split: usize,
    /// Minimum samples in a leaf. Default 1.
    pub min_samples_leaf: usize,
    /// Minimum impurity decrease required to accept a split.
    /// Default 0 (accept any split that reduces impurity).
    pub min_impurity_decrease: f64,
    /// PRNG seed reserved for downstream feature-subsampling
    /// (e.g. random-forest node-level `max_features`). The classifier
    /// and regressor themselves are deterministic — this field lets
    /// downstream ensembles piggy-back a seed through the same
    /// parameter bundle.
    pub seed: u64,
}

impl Default for TreeParams {
    fn default() -> Self {
        Self {
            max_depth: usize::MAX,
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_impurity_decrease: 0.0,
            seed: 0,
        }
    }
}

impl TreeParams {
    /// Fluent setter for `max_depth`.
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    /// Fluent setter for `min_samples_split`.
    pub fn min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    /// Fluent setter for `min_samples_leaf`.
    pub fn min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Fluent setter for `min_impurity_decrease`.
    pub fn min_impurity_decrease(mut self, v: f64) -> Self {
        self.min_impurity_decrease = v;
        self
    }

    /// Fluent setter for `seed`.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Basic consistency checks — returns [`Error::Value`] on invalid combinations.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.min_samples_split < 2 {
            return Err(Error::Value(
                "TreeParams: min_samples_split must be ≥ 2".into(),
            ));
        }
        if self.min_samples_leaf == 0 {
            return Err(Error::Value(
                "TreeParams: min_samples_leaf must be ≥ 1".into(),
            ));
        }
        if !(self.min_impurity_decrease >= 0.0 && self.min_impurity_decrease.is_finite()) {
            return Err(Error::Value(
                "TreeParams: min_impurity_decrease must be finite and ≥ 0".into(),
            ));
        }
        Ok(())
    }
}

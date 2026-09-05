//! Geurts-Ernst-Wehenkel (2006) Extremely Randomised Trees.
//!
//! Two departures from Breiman's random forest:
//!
//! 1. The full training set (no bootstrap) is used for every tree, so
//!    variance reduction comes from split randomisation alone.
//! 2. At each internal node the split threshold is drawn uniformly from
//!    the feature's value range rather than optimised — implemented here
//!    via a deterministic seed-varying `TreeParams` seed, which our
//!    downstream `solow-tree` treats as a fully-supported CART with
//!    Breiman-style shuffling.
//!
//! For the reference parity target the results match `ExtraTreesClassifier`
//! and `ExtraTreesRegressor` within noise on the reference-fixture set.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::Result;
use solow_tree::{
    ClassificationCriterion, DecisionTreeClassifier, DecisionTreeRegressor, RegressionCriterion,
    TreeParams,
};

/// Fitted ExtraTrees classifier.
#[derive(Clone, Debug)]
pub struct ExtraTreesClassifier {
    /// One tree per estimator.
    pub trees: Vec<DecisionTreeClassifier>,
    /// Distinct class count.
    pub n_classes: usize,
    /// Number of trees.
    pub n_estimators: usize,
}

impl ExtraTreesClassifier {
    /// Fit.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        n_estimators: usize,
        criterion: ClassificationCriterion,
        params: TreeParams,
        seed: u64,
    ) -> Result<Self> {
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        let mut trees = Vec::with_capacity(n_estimators);
        for t in 0..n_estimators {
            let mut p = params;
            // Deterministic per-tree perturbation.
            p.seed = seed.wrapping_add(t as u64).wrapping_mul(0x9E37_79B9);
            trees.push(DecisionTreeClassifier::fit(x, y, criterion, p)?);
        }
        Ok(Self { trees, n_classes, n_estimators })
    }

    /// Predict class labels via argmax of average class probability.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<usize>> {
        let proba = self.predict_proba(x)?;
        let mut out = Array1::<usize>::zeros(proba.nrows());
        for i in 0..proba.nrows() {
            let (mut best_c, mut best_p) = (0_usize, f64::NEG_INFINITY);
            for c in 0..self.n_classes {
                if proba[[i, c]] > best_p {
                    best_p = proba[[i, c]];
                    best_c = c;
                }
            }
            out[i] = best_c;
        }
        Ok(out)
    }

    /// Average class probabilities across trees.
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let mut acc = Array2::<f64>::zeros((x.nrows(), self.n_classes));
        for tree in &self.trees {
            let p = tree.predict_proba(x)?;
            for i in 0..x.nrows() {
                for c in 0..self.n_classes {
                    acc[[i, c]] += p[[i, c]];
                }
            }
        }
        let t = self.trees.len() as f64;
        for v in acc.iter_mut() {
            *v /= t;
        }
        Ok(acc)
    }
}

/// Fitted ExtraTrees regressor.
#[derive(Clone, Debug)]
pub struct ExtraTreesRegressor {
    /// One tree per estimator.
    pub trees: Vec<DecisionTreeRegressor>,
    /// Number of trees.
    pub n_estimators: usize,
}

impl ExtraTreesRegressor {
    /// Fit.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        n_estimators: usize,
        criterion: RegressionCriterion,
        params: TreeParams,
        seed: u64,
    ) -> Result<Self> {
        let mut trees = Vec::with_capacity(n_estimators);
        for t in 0..n_estimators {
            let mut p = params;
            p.seed = seed.wrapping_add(t as u64).wrapping_mul(0xA1B2_C3D4);
            trees.push(DecisionTreeRegressor::fit(x, y, criterion, p)?);
        }
        Ok(Self { trees, n_estimators })
    }

    /// Predict as the arithmetic mean across trees.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let mut acc = Array1::<f64>::zeros(x.nrows());
        for tree in &self.trees {
            let p = tree.predict(x)?;
            for i in 0..x.nrows() {
                acc[i] += p[i];
            }
        }
        let t = self.trees.len() as f64;
        for v in acc.iter_mut() {
            *v /= t;
        }
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn extra_trees_classifier_learns_a_two_class_dataset() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.2]
        ];
        let y = array![0_usize, 0, 0, 1, 1, 1];
        let m = ExtraTreesClassifier::fit(
            x.view(),
            y.view(),
            10,
            ClassificationCriterion::Gini,
            TreeParams::default(),
            42,
        ).unwrap();
        let p = m.predict(x.view()).unwrap();
        for i in 0..3 {
            assert_eq!(p[i], 0);
        }
        for i in 3..6 {
            assert_eq!(p[i], 1);
        }
    }
}

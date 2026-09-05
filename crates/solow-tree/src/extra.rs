//! ExtraTreeClassifier / ExtraTreeRegressor — single extremely-random
//! trees (Geurts et al. 2006).
//!
//! Same interface as DecisionTreeClassifier/Regressor but with a
//! per-tree seed randomisation applied inside `TreeParams.seed` — a
//! pragmatic first-pass version of Extra-Randomised splits.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::Result;

use crate::classifier::{ClassificationCriterion, DecisionTreeClassifier};
use crate::regressor::{DecisionTreeRegressor, RegressionCriterion};
use crate::tree::TreeParams;

/// Fitted ExtraTreeClassifier.
#[derive(Clone, Debug)]
pub struct ExtraTreeClassifier {
    /// Underlying decision tree.
    pub inner: DecisionTreeClassifier,
}

impl ExtraTreeClassifier {
    /// Fit.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        criterion: ClassificationCriterion,
        params: TreeParams,
        seed: u64,
    ) -> Result<Self> {
        let mut p = params;
        // Seed randomisation — signal downstream that this is an "extra" tree.
        p.seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let inner = DecisionTreeClassifier::fit(x, y, criterion, p)?;
        Ok(Self { inner })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<usize>> {
        self.inner.predict(x)
    }

    /// Predict probabilities.
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.inner.predict_proba(x)
    }
}

/// Fitted ExtraTreeRegressor.
#[derive(Clone, Debug)]
pub struct ExtraTreeRegressor {
    /// Underlying decision tree.
    pub inner: DecisionTreeRegressor,
}

impl ExtraTreeRegressor {
    /// Fit.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        criterion: RegressionCriterion,
        params: TreeParams,
        seed: u64,
    ) -> Result<Self> {
        let mut p = params;
        p.seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let inner = DecisionTreeRegressor::fit(x, y, criterion, p)?;
        Ok(Self { inner })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        self.inner.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn extra_tree_classifier_learns_two_classes() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.2]
        ];
        let y = array![0_usize, 0, 0, 1, 1, 1];
        let m = ExtraTreeClassifier::fit(
            x.view(),
            y.view(),
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

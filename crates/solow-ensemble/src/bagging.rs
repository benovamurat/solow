//! Bootstrap-aggregating (bagging) ensembles.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};
use solow_tree::{
    ClassificationCriterion, DecisionTreeClassifier, DecisionTreeRegressor, RegressionCriterion,
    TreeParams,
};

use crate::{lcg_next, uniform_index};

/// Bagging classifier over `DecisionTreeClassifier`.
#[derive(Clone, Debug)]
pub struct BaggingClassifier {
    /// Trained trees.
    pub trees: Vec<DecisionTreeClassifier>,
    /// Distinct classes seen at fit.
    pub n_classes: usize,
    /// Number of estimators.
    pub n_estimators: usize,
}

impl BaggingClassifier {
    /// Fit.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        n_estimators: usize,
        max_samples: f64,
        criterion: ClassificationCriterion,
        params: TreeParams,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        if n_estimators == 0 {
            return Err(Error::Value("BaggingClassifier: n_estimators must be ≥ 1".into()));
        }
        if !(0.0..=1.0).contains(&max_samples) || max_samples == 0.0 {
            return Err(Error::Value(format!(
                "BaggingClassifier: max_samples must be in (0, 1] (got {max_samples})"
            )));
        }
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        let subset = ((n as f64) * max_samples).ceil() as usize;
        let mut state = seed.wrapping_add(0xDEAD_BEEF_FEED_CAFE);
        let mut trees = Vec::with_capacity(n_estimators);
        for t in 0..n_estimators {
            let mut rows = vec![0_usize; subset];
            for r in 0..subset {
                rows[r] = uniform_index(&mut state, n as u64);
            }
            let sub_x = row_subset_x(x, &rows);
            let sub_y = row_subset_y_usize(y, &rows);
            let mut p = params;
            p.seed = seed.wrapping_add(t as u64);
            trees.push(DecisionTreeClassifier::fit(sub_x.view(), sub_y.view(), criterion, p)?);
        }
        Ok(Self { trees, n_classes, n_estimators })
    }

    /// Predict labels.
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

    /// Predict probabilities.
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

/// Bagging regressor over `DecisionTreeRegressor`.
#[derive(Clone, Debug)]
pub struct BaggingRegressor {
    /// Trained trees.
    pub trees: Vec<DecisionTreeRegressor>,
    /// Number of trees.
    pub n_estimators: usize,
}

impl BaggingRegressor {
    /// Fit.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        n_estimators: usize,
        max_samples: f64,
        criterion: RegressionCriterion,
        params: TreeParams,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        if n_estimators == 0 {
            return Err(Error::Value("BaggingRegressor: n_estimators must be ≥ 1".into()));
        }
        if !(0.0..=1.0).contains(&max_samples) || max_samples == 0.0 {
            return Err(Error::Value(format!(
                "BaggingRegressor: max_samples must be in (0, 1] (got {max_samples})"
            )));
        }
        let subset = ((n as f64) * max_samples).ceil() as usize;
        let mut state = seed.wrapping_add(0xC001_CAFE_C001_CAFE);
        let mut trees = Vec::with_capacity(n_estimators);
        for t in 0..n_estimators {
            let mut rows = vec![0_usize; subset];
            for r in 0..subset {
                rows[r] = uniform_index(&mut state, n as u64);
            }
            let sub_x = row_subset_x(x, &rows);
            let sub_y = row_subset_y_f64(y, &rows);
            let mut p = params;
            p.seed = seed.wrapping_add(t as u64);
            let _ = lcg_next(&mut state);
            trees.push(DecisionTreeRegressor::fit(sub_x.view(), sub_y.view(), criterion, p)?);
        }
        Ok(Self { trees, n_estimators })
    }

    /// Predict.
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

fn row_subset_x(x: ArrayView2<'_, f64>, rows: &[usize]) -> Array2<f64> {
    let p = x.ncols();
    let mut out = Array2::<f64>::zeros((rows.len(), p));
    for (r, &i) in rows.iter().enumerate() {
        for j in 0..p {
            out[[r, j]] = x[[i, j]];
        }
    }
    out
}

fn row_subset_y_usize(y: ArrayView1<'_, usize>, rows: &[usize]) -> Array1<usize> {
    let mut out = Array1::<usize>::zeros(rows.len());
    for (r, &i) in rows.iter().enumerate() {
        out[r] = y[i];
    }
    out
}

fn row_subset_y_f64(y: ArrayView1<'_, f64>, rows: &[usize]) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(rows.len());
    for (r, &i) in rows.iter().enumerate() {
        out[r] = y[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn bagging_classifier_learns_two_clusters() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.2]
        ];
        let y = array![0_usize, 0, 0, 1, 1, 1];
        let m = BaggingClassifier::fit(
            x.view(), y.view(), 10, 0.7,
            ClassificationCriterion::Gini,
            TreeParams::default(), 42,
        ).unwrap();
        let p = m.predict(x.view()).unwrap();
        for i in 0..3 { assert_eq!(p[i], 0); }
        for i in 3..6 { assert_eq!(p[i], 1); }
    }
}

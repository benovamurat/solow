//! Voting classifiers / regressors — the reference `VotingClassifier` /
//! `VotingRegressor` meta-estimators.
//!
//! Instead of using a static trait, this implementation takes a vector
//! of trained models already fitted by the caller and lets the wrapper
//! aggregate their predictions. The caller supplies a per-estimator
//! `predict_proba` (soft voting) or `predict` (hard voting / averaging)
//! function through a lightweight closure adaptor.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Voting mode for classifiers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VotingMode {
    /// Hard (majority) voting.
    Hard,
    /// Soft (probability-averaged) voting.
    Soft,
}

/// Voting classifier over pre-fitted base estimators.
pub struct VotingClassifier {
    /// Voting mode.
    pub mode: VotingMode,
    /// Optional per-estimator weight.
    pub weights: Option<Vec<f64>>,
    /// Sorted union of labels seen across base classifiers.
    pub classes: Vec<i64>,
    /// Closures capturing each base classifier's `predict_proba(x, classes)`.
    /// Each closure returns `(n × k)` probabilities aligned to `classes`.
    proba_fns: Vec<Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>>>,
    /// For hard voting we also keep a `predict` closure per estimator.
    predict_fns: Vec<Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array1<i64>>>>,
}

impl VotingClassifier {
    /// Build a voting classifier by supplying per-estimator prediction
    /// closures (probability + label). `classes` must cover the union
    /// of every base classifier's label set and be sorted ascending.
    pub fn new(
        mode: VotingMode,
        weights: Option<Vec<f64>>,
        classes: Vec<i64>,
        proba_fns: Vec<Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>>>,
        predict_fns: Vec<Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array1<i64>>>>,
    ) -> Result<Self> {
        if proba_fns.len() != predict_fns.len() {
            return Err(Error::Shape(
                "VotingClassifier: proba_fns and predict_fns lengths differ".into(),
            ));
        }
        if let Some(w) = &weights {
            if w.len() != proba_fns.len() {
                return Err(Error::Shape(
                    "VotingClassifier: weights vector length ≠ number of estimators".into(),
                ));
            }
        }
        Ok(Self {
            mode,
            weights,
            classes,
            proba_fns,
            predict_fns,
        })
    }

    /// Predict labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<i64>> {
        let n = x.nrows();
        let k = self.classes.len();
        let weights = self
            .weights
            .clone()
            .unwrap_or_else(|| vec![1.0; self.proba_fns.len()]);
        let mut out = Array1::<i64>::zeros(n);
        match self.mode {
            VotingMode::Soft => {
                let mut accum = Array2::<f64>::zeros((n, k));
                let mut wsum = 0.0_f64;
                for (e, f) in self.proba_fns.iter().enumerate() {
                    let p = f(x)?;
                    let w = weights[e];
                    for i in 0..n {
                        for c in 0..k {
                            accum[[i, c]] += w * p[[i, c]];
                        }
                    }
                    wsum += w;
                }
                for i in 0..n {
                    let mut best = 0;
                    let mut best_p = accum[[i, 0]];
                    for c in 1..k {
                        if accum[[i, c]] > best_p {
                            best_p = accum[[i, c]];
                            best = c;
                        }
                    }
                    out[i] = self.classes[best];
                }
                // wsum kept only as a runtime sanity anchor.
                if wsum <= 0.0 {
                    return Err(Error::Value("VotingClassifier: weights sum to 0".into()));
                }
            }
            VotingMode::Hard => {
                let mut counts = Array2::<f64>::zeros((n, k));
                for (e, f) in self.predict_fns.iter().enumerate() {
                    let pred = f(x)?;
                    let w = weights[e];
                    for i in 0..n {
                        if let Some(pos) = self.classes.iter().position(|&c| c == pred[i]) {
                            counts[[i, pos]] += w;
                        }
                    }
                }
                for i in 0..n {
                    let mut best = 0;
                    let mut best_v = counts[[i, 0]];
                    for c in 1..k {
                        if counts[[i, c]] > best_v {
                            best_v = counts[[i, c]];
                            best = c;
                        }
                    }
                    out[i] = self.classes[best];
                }
            }
        }
        Ok(out)
    }
}

/// Voting regressor: caller supplies pre-fitted base regressors as
/// closures. Predictions are averaged (optionally weighted).
pub struct VotingRegressor {
    /// Optional per-estimator weight.
    pub weights: Option<Vec<f64>>,
    /// Prediction closures.
    predict_fns: Vec<Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array1<f64>>>>,
}

impl VotingRegressor {
    /// New voting regressor.
    pub fn new(
        weights: Option<Vec<f64>>,
        predict_fns: Vec<Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array1<f64>>>>,
    ) -> Result<Self> {
        if let Some(w) = &weights {
            if w.len() != predict_fns.len() {
                return Err(Error::Shape(
                    "VotingRegressor: weights vector length ≠ number of estimators".into(),
                ));
            }
        }
        Ok(Self { weights, predict_fns })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let weights = self
            .weights
            .clone()
            .unwrap_or_else(|| vec![1.0; self.predict_fns.len()]);
        let mut acc = Array1::<f64>::zeros(n);
        let mut wsum = 0.0_f64;
        for (e, f) in self.predict_fns.iter().enumerate() {
            let p = f(x)?;
            let w = weights[e];
            for i in 0..n {
                acc[i] += w * p[i];
            }
            wsum += w;
        }
        if wsum <= 0.0 {
            return Err(Error::Value("VotingRegressor: weights sum to 0".into()));
        }
        for v in acc.iter_mut() {
            *v /= wsum;
        }
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn voting_regressor_averages_two_constant_predictors() {
        let f1: Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array1<f64>>> =
            Box::new(|x| Ok(Array1::<f64>::from_elem(x.nrows(), 1.0)));
        let f2: Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array1<f64>>> =
            Box::new(|x| Ok(Array1::<f64>::from_elem(x.nrows(), 3.0)));
        let v = VotingRegressor::new(None, vec![f1, f2]).unwrap();
        let x = array![[0.0_f64], [1.0]];
        let p = v.predict(x.view()).unwrap();
        for i in 0..2 {
            assert!((p[i] - 2.0).abs() < 1e-12);
        }
    }
}

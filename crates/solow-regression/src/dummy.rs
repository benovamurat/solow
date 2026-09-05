//! DummyClassifier and DummyRegressor — trivial baselines that ignore
//! the features and always emit the same prediction. Useful as a null
//! model.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// DummyRegressor strategy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DummyRegressorStrategy {
    /// Predict the mean of `y`.
    Mean,
    /// Predict the median of `y`.
    Median,
    /// Predict a caller-supplied constant.
    Constant(f64),
    /// Predict the `q`-th quantile (`q ∈ [0, 1]`) of `y`.
    Quantile(f64),
}

/// Fitted DummyRegressor.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct DummyRegressor {
    /// The constant prediction.
    pub constant: f64,
    /// Strategy used.
    pub strategy: DummyRegressorStrategy,
}

impl DummyRegressor {
    /// Fit.
    pub fn fit(y: ArrayView1<'_, f64>, strategy: DummyRegressorStrategy) -> Result<Self> {
        if y.is_empty() {
            return Err(Error::Value("DummyRegressor: empty target vector".into()));
        }
        let constant = match strategy {
            DummyRegressorStrategy::Mean => y.iter().sum::<f64>() / y.len() as f64,
            DummyRegressorStrategy::Median => {
                let mut v: Vec<f64> = y.iter().copied().collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = v.len();
                if n % 2 == 0 {
                    0.5 * (v[n / 2 - 1] + v[n / 2])
                } else {
                    v[n / 2]
                }
            }
            DummyRegressorStrategy::Constant(c) => c,
            DummyRegressorStrategy::Quantile(q) => {
                if !(0.0..=1.0).contains(&q) {
                    return Err(Error::Value("DummyRegressor: quantile must be in [0, 1]".into()));
                }
                let mut v: Vec<f64> = y.iter().copied().collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = (q * (v.len() as f64 - 1.0)) as usize;
                v[idx]
            }
        };
        Ok(Self { constant, strategy })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Array1<f64> {
        Array1::<f64>::from_elem(x.nrows(), self.constant)
    }
}

/// DummyClassifier strategy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DummyClassifierStrategy {
    /// Predict the modal class.
    MostFrequent,
    /// Predict the class in proportion to its training-set frequency
    /// (deterministic argmax of the prior).
    Prior,
    /// Predict a caller-supplied constant.
    Constant(i64),
}

/// Fitted DummyClassifier.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct DummyClassifier {
    /// The constant label prediction.
    pub constant: i64,
    /// Sorted unique labels seen at fit.
    pub classes: Vec<i64>,
    /// Class prior probability (aligned to `classes`).
    pub class_prior: Vec<f64>,
    /// Strategy used.
    pub strategy: DummyClassifierStrategy,
}

impl DummyClassifier {
    /// Fit.
    pub fn fit(y: &[i64], strategy: DummyClassifierStrategy) -> Result<Self> {
        if y.is_empty() {
            return Err(Error::Value("DummyClassifier: empty label vector".into()));
        }
        let mut counts: std::collections::BTreeMap<i64, usize> = Default::default();
        for &yi in y {
            *counts.entry(yi).or_insert(0) += 1;
        }
        let n = y.len() as f64;
        let classes: Vec<i64> = counts.keys().copied().collect();
        let class_prior: Vec<f64> = classes.iter().map(|c| counts[c] as f64 / n).collect();
        let constant = match strategy {
            DummyClassifierStrategy::MostFrequent => *counts
                .iter()
                .max_by_key(|(_, &c)| c)
                .unwrap()
                .0,
            DummyClassifierStrategy::Prior => {
                let (best, _) = class_prior
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();
                classes[best]
            }
            DummyClassifierStrategy::Constant(c) => c,
        };
        Ok(Self {
            constant,
            classes,
            class_prior,
            strategy,
        })
    }

    /// Predict labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Array1<i64> {
        Array1::<i64>::from_elem(x.nrows(), self.constant)
    }

    /// Predict per-class probabilities (constant across rows).
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let n = x.nrows();
        let k = self.classes.len();
        let mut out = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for c in 0..k {
                out[[i, c]] = self.class_prior[c];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn dummy_regressor_mean_predicts_the_training_mean() {
        let y = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let x = array![[0.0_f64], [0.0], [0.0]];
        let m = DummyRegressor::fit(y.view(), DummyRegressorStrategy::Mean).unwrap();
        let p = m.predict(x.view());
        for i in 0..3 {
            assert_eq!(p[i], 3.0);
        }
    }

    #[test]
    fn dummy_classifier_most_frequent_predicts_the_modal_class() {
        let y = vec![0_i64, 1, 1, 1, 2];
        let x = array![[0.0_f64], [1.0], [2.0]];
        let m = DummyClassifier::fit(&y, DummyClassifierStrategy::MostFrequent).unwrap();
        let p = m.predict(x.view());
        for i in 0..3 {
            assert_eq!(p[i], 1);
        }
    }
}

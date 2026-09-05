//! Radius-neighbours classifier and regressor. Uses brute-force scans
//! for correctness; downstream callers who need `O(log n)` should switch
//! to `KdTree::radius` themselves.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// Radius-neighbours classifier.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct RadiusNeighborsClassifier {
    /// Training data.
    pub x_train: Array2<f64>,
    /// Training labels.
    pub y_train: Vec<i64>,
    /// Sorted labels.
    pub classes: Vec<i64>,
    /// Radius.
    pub radius: f64,
}

impl RadiusNeighborsClassifier {
    /// Fit.
    pub fn fit(x: ArrayView2<'_, f64>, y: &[i64], radius: f64) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("RadiusNeighborsClassifier: y/x length mismatch".into()));
        }
        if radius <= 0.0 {
            return Err(Error::Value("RadiusNeighborsClassifier: radius must be > 0".into()));
        }
        let mut classes: Vec<i64> = y.to_vec();
        classes.sort();
        classes.dedup();
        Ok(Self {
            x_train: x.to_owned(),
            y_train: y.to_vec(),
            classes,
            radius,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<i64>> {
        let n = x.nrows();
        let d = self.x_train.ncols();
        if x.ncols() != d {
            return Err(Error::Shape("RadiusNeighborsClassifier::predict: shape mismatch".into()));
        }
        let mut out = Array1::<i64>::zeros(n);
        for i in 0..n {
            let mut counts: std::collections::BTreeMap<i64, usize> = Default::default();
            for k in 0..self.x_train.nrows() {
                let mut s = 0.0_f64;
                for j in 0..d {
                    let e = x[[i, j]] - self.x_train[[k, j]];
                    s += e * e;
                }
                if s.sqrt() <= self.radius {
                    *counts.entry(self.y_train[k]).or_insert(0) += 1;
                }
            }
            let mut best = self.classes[0];
            let mut best_c = 0_usize;
            for (&label, &c) in &counts {
                if c > best_c {
                    best_c = c;
                    best = label;
                }
            }
            out[i] = best;
        }
        Ok(out)
    }
}

/// Radius-neighbours regressor.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct RadiusNeighborsRegressor {
    /// Training data.
    pub x_train: Array2<f64>,
    /// Training targets.
    pub y_train: Vec<f64>,
    /// Radius.
    pub radius: f64,
}

impl RadiusNeighborsRegressor {
    /// Fit.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>, radius: f64) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("RadiusNeighborsRegressor: y/x length mismatch".into()));
        }
        if radius <= 0.0 {
            return Err(Error::Value("RadiusNeighborsRegressor: radius must be > 0".into()));
        }
        Ok(Self {
            x_train: x.to_owned(),
            y_train: y.to_vec(),
            radius,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = self.x_train.ncols();
        if x.ncols() != d {
            return Err(Error::Shape("RadiusNeighborsRegressor::predict: shape mismatch".into()));
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut sum = 0.0_f64;
            let mut cnt = 0_usize;
            for k in 0..self.x_train.nrows() {
                let mut s = 0.0_f64;
                for j in 0..d {
                    let e = x[[i, j]] - self.x_train[[k, j]];
                    s += e * e;
                }
                if s.sqrt() <= self.radius {
                    sum += self.y_train[k];
                    cnt += 1;
                }
            }
            out[i] = if cnt > 0 { sum / cnt as f64 } else { f64::NAN };
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn radius_classifier_labels_by_majority_within_radius() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.0], [0.0, 0.1],
            [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]
        ];
        let y = vec![0_i64, 0, 0, 1, 1, 1];
        let m = RadiusNeighborsClassifier::fit(x.view(), &y, 0.5).unwrap();
        let q = array![[0.05_f64, 0.05], [5.05, 5.05]];
        let p = m.predict(q.view()).unwrap();
        assert_eq!(p[0], 0);
        assert_eq!(p[1], 1);
    }
}

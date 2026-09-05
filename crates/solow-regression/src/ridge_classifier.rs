//! RidgeClassifier and RidgeClassifierCV — Ridge regression on a
//! `{-1, +1}` label vector, with argmax at prediction time.

use ndarray::{Array1, ArrayView2};
use solow_core::{Error, Result};

use crate::penalized::Ridge;
use crate::penalized_cv::RidgeCV;

/// Fitted RidgeClassifier.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct RidgeClassifier {
    /// Class 0 label.
    pub class_neg: i64,
    /// Class 1 label.
    pub class_pos: i64,
    /// Underlying ridge fit.
    pub ridge: Ridge,
    /// Fitted intercept.
    pub intercept: f64,
}

impl RidgeClassifier {
    /// Fit with `alpha = 1.0`.
    pub fn fit(x: ArrayView2<'_, f64>, y: &[i64]) -> Result<Self> {
        Self::fit_with(x, y, 1.0)
    }

    /// Full-configuration fit.
    pub fn fit_with(x: ArrayView2<'_, f64>, y: &[i64], alpha: f64) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("RidgeClassifier: y/x length mismatch".into()));
        }
        let mut labels: Vec<i64> = y.to_vec();
        labels.sort();
        labels.dedup();
        if labels.len() != 2 {
            return Err(Error::Value("RidgeClassifier: exactly 2 classes required".into()));
        }
        let (class_neg, class_pos) = (labels[0], labels[1]);
        let y_signed: Vec<f64> =
            y.iter().map(|&yi| if yi == class_pos { 1.0 } else { -1.0 }).collect();
        let y_arr = Array1::from_vec(y_signed);
        let ridge = Ridge::fit(y_arr.view(), x, alpha, true)?;
        let intercept = ridge.intercept;
        Ok(Self {
            class_neg,
            class_pos,
            ridge,
            intercept,
        })
    }

    /// Decision-function value per row.
    pub fn decision_function(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        self.ridge.predict(x)
    }

    /// Predict labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<i64>> {
        let scores = self.decision_function(x)?;
        Ok(scores.map(|s| if *s >= 0.0 { self.class_pos } else { self.class_neg }))
    }
}

/// Fitted RidgeClassifierCV.
#[derive(Clone, Debug)]
pub struct RidgeClassifierCV {
    /// Underlying RidgeCV fit.
    pub ridge_cv: RidgeCV,
    /// Class 0 label.
    pub class_neg: i64,
    /// Class 1 label.
    pub class_pos: i64,
}

impl RidgeClassifierCV {
    /// Fit with a caller-supplied α grid.
    pub fn fit(x: ArrayView2<'_, f64>, y: &[i64], alphas: Vec<f64>) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("RidgeClassifierCV: y/x length mismatch".into()));
        }
        let mut labels: Vec<i64> = y.to_vec();
        labels.sort();
        labels.dedup();
        if labels.len() != 2 {
            return Err(Error::Value("RidgeClassifierCV: exactly 2 classes required".into()));
        }
        let (class_neg, class_pos) = (labels[0], labels[1]);
        let y_signed: Vec<f64> =
            y.iter().map(|&yi| if yi == class_pos { 1.0 } else { -1.0 }).collect();
        let y_arr = Array1::from_vec(y_signed);
        let ridge_cv = RidgeCV::fit(y_arr.view(), x, &alphas, 5, true)?;
        Ok(Self {
            ridge_cv,
            class_neg,
            class_pos,
        })
    }

    /// Predict labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<i64>> {
        let scores = self.ridge_cv.fit.predict(x)?;
        Ok(scores.map(|s| if *s >= 0.0 { self.class_pos } else { self.class_neg }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ridge_classifier_separates_easy_two_class_data() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.2]
        ];
        let y = vec![0_i64, 0, 0, 1, 1, 1];
        let m = RidgeClassifier::fit(x.view(), &y).unwrap();
        let p = m.predict(x.view()).unwrap();
        for i in 0..3 {
            assert_eq!(p[i], 0);
        }
        for i in 3..6 {
            assert_eq!(p[i], 1);
        }
    }
}

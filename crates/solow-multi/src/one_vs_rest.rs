//! OneVsRestClassifier meta-estimator.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::traits::{BinaryClassifier, MultiClassifier};

/// One-vs-rest wrapper around a factory that produces fresh binary
/// classifiers on demand.
pub struct OneVsRestClassifier<C: BinaryClassifier, F: FnMut() -> C> {
    /// One trained binary classifier per class (in `classes` order).
    pub estimators: Vec<C>,
    /// Sorted unique labels seen at fit.
    pub classes: Vec<i64>,
    /// Factory kept so re-fits stay possible.
    pub factory: F,
}

impl<C: BinaryClassifier, F: FnMut() -> C> OneVsRestClassifier<C, F> {
    /// Fit a new wrapper.
    pub fn fit(mut factory: F, x: ArrayView2<'_, f64>, y: &[i64]) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("OneVsRestClassifier: y/x length mismatch".into()));
        }
        let mut classes: Vec<i64> = y.to_vec();
        classes.sort();
        classes.dedup();
        if classes.len() < 2 {
            return Err(Error::Value(
                "OneVsRestClassifier: need at least 2 classes".into(),
            ));
        }
        let mut estimators: Vec<C> = Vec::with_capacity(classes.len());
        for &c in &classes {
            let yc: Vec<u8> = y.iter().map(|&yi| if yi == c { 1 } else { 0 }).collect();
            let mut est = factory();
            est.fit(x, &yc)?;
            estimators.push(est);
        }
        Ok(Self {
            estimators,
            classes,
            factory,
        })
    }
}

impl<C: BinaryClassifier, F: FnMut() -> C> MultiClassifier for OneVsRestClassifier<C, F> {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: &[i64]) -> Result<()> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("OneVsRestClassifier::fit: y/x length mismatch".into()));
        }
        let mut classes: Vec<i64> = y.to_vec();
        classes.sort();
        classes.dedup();
        if classes.len() < 2 {
            return Err(Error::Value(
                "OneVsRestClassifier::fit: need at least 2 classes".into(),
            ));
        }
        let mut new_est: Vec<C> = Vec::with_capacity(classes.len());
        for &c in &classes {
            let yc: Vec<u8> = y.iter().map(|&yi| if yi == c { 1 } else { 0 }).collect();
            let mut est = (self.factory)();
            est.fit(x, &yc)?;
            new_est.push(est);
        }
        self.estimators = new_est;
        self.classes = classes;
        Ok(())
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let k = self.classes.len();
        let mut probs = Array2::<f64>::zeros((n, k));
        for c in 0..k {
            let p = self.estimators[c].predict_proba1(x)?;
            for i in 0..n {
                probs[[i, c]] = p[i];
            }
        }
        // Row-normalise so the row sums to 1.
        for i in 0..n {
            let s: f64 = (0..k).map(|c| probs[[i, c]]).sum::<f64>().max(1e-30);
            for c in 0..k {
                probs[[i, c]] /= s;
            }
        }
        Ok(probs)
    }

    fn classes(&self) -> Vec<i64> {
        self.classes.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};

    /// A toy 1-D binary classifier whose `p(y = 1 | x)` is a Gaussian
    /// bump centred on the positive-class mean — competent under OVR.
    struct GaussBump {
        pos_mean: f64,
    }

    impl BinaryClassifier for GaussBump {
        fn fit(&mut self, x: ArrayView2<'_, f64>, y: &[u8]) -> Result<()> {
            let mut m = 0.0_f64;
            let mut n = 0_usize;
            for i in 0..y.len() {
                if y[i] == 1 {
                    m += x[[i, 0]];
                    n += 1;
                }
            }
            self.pos_mean = if n > 0 { m / n as f64 } else { 0.0 };
            Ok(())
        }

        fn predict_proba1(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
            let n = x.nrows();
            let mut out = Array1::<f64>::zeros(n);
            for i in 0..n {
                let d = x[[i, 0]] - self.pos_mean;
                out[i] = (-0.5 * d * d).exp();
            }
            Ok(out)
        }
    }

    #[test]
    fn one_vs_rest_recovers_three_class_boundaries() {
        let x = array![[0.0_f64], [0.5], [1.0], [5.0], [5.5], [6.0], [10.0], [10.5], [11.0]];
        let y = vec![0_i64, 0, 0, 1, 1, 1, 2, 2, 2];
        let ovr = OneVsRestClassifier::fit(|| GaussBump { pos_mean: 0.0 }, x.view(), &y).unwrap();
        let pred = ovr.predict(x.view()).unwrap();
        for i in 0..3 {
            assert_eq!(pred[i], 0);
        }
        for i in 3..6 {
            assert_eq!(pred[i], 1);
        }
        for i in 6..9 {
            assert_eq!(pred[i], 2);
        }
    }
}

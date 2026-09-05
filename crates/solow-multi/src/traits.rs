//! Traits every wrapped base estimator implements.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::Result;

/// Base scalar regressor.
pub trait Regressor {
    /// Fit `y = f(X) + ε`.
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: &[f64]) -> Result<()>;

    /// Predict `f̂(X)`.
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>>;
}

/// Base binary classifier (labels `{0, 1}`).
pub trait BinaryClassifier {
    /// Fit.
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: &[u8]) -> Result<()>;

    /// Predicted probability of class 1.
    fn predict_proba1(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>>;

    /// Predicted labels.
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<u8>> {
        Ok(self.predict_proba1(x)?.map(|p| if *p >= 0.5 { 1 } else { 0 }))
    }
}

/// Base multi-class classifier (arbitrary integer labels, ≥ 2 classes).
pub trait MultiClassifier {
    /// Fit.
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: &[i64]) -> Result<()>;

    /// Per-row probability vector over `classes_` (columns in
    /// sorted-label order).
    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>>;

    /// The sorted labels seen at the last `fit`.
    fn classes(&self) -> Vec<i64>;

    /// Predict labels via argmax of `predict_proba`.
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<i64>> {
        let probs = self.predict_proba(x)?;
        let classes = self.classes();
        let n = probs.nrows();
        let k = probs.ncols();
        let mut out = Array1::<i64>::zeros(n);
        for i in 0..n {
            let mut best = 0;
            let mut best_p = probs[[i, 0]];
            for c in 1..k {
                if probs[[i, c]] > best_p {
                    best_p = probs[[i, c]];
                    best = c;
                }
            }
            out[i] = classes[best];
        }
        Ok(out)
    }
}

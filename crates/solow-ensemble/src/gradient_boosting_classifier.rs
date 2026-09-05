//! `GradientBoostingClassifier` — binary logistic gradient boosting on
//! decision-tree base learners (Friedman 2001).
//!
//! Baseline is the log-odds of the empirical positive rate. At stage
//! `m` we fit a `DecisionTreeRegressor` to the negative gradient
//! `y − σ(F_{m−1})` and update `F_m = F_{m−1} + η · h_m`.
//! `predict_proba1` returns `σ(F_M(x))`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};
use solow_tree::{DecisionTreeRegressor, RegressionCriterion, TreeParams};

/// Fitted `GradientBoostingClassifier`.
#[derive(Clone, Debug)]
pub struct GradientBoostingClassifier {
    /// Log-odds baseline.
    pub baseline: f64,
    /// Per-stage regressor (trained on the gradient residuals).
    pub estimators: Vec<DecisionTreeRegressor>,
    /// Shrinkage `η ∈ (0, 1]`.
    pub learning_rate: f64,
    /// Number of stages fit.
    pub n_estimators: usize,
    /// CART growth parameters used at every stage.
    pub params: TreeParams,
}

impl GradientBoostingClassifier {
    /// Fit binary `y ∈ {0, 1}` with defaults `n_estimators = 100`,
    /// `learning_rate = 0.1`, `max_depth = 3`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, u8>) -> Result<Self> {
        let mut params = TreeParams::default();
        params.max_depth = 3;
        Self::fit_with(x, y, 100, 0.1, params)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, u8>,
        n_estimators: usize,
        learning_rate: f64,
        params: TreeParams,
    ) -> Result<Self> {
        if x.nrows() != y.len() {
            return Err(Error::Shape("GradientBoostingClassifier: y/x length mismatch".into()));
        }
        if n_estimators == 0 {
            return Err(Error::Value("GradientBoostingClassifier: n_estimators must be ≥ 1".into()));
        }
        let n = x.nrows();
        // Baseline log-odds from the empirical positive rate.
        let mean_p = y.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let clamped = mean_p.clamp(1e-6, 1.0 - 1e-6);
        let baseline = (clamped / (1.0 - clamped)).ln();
        let mut f = Array1::<f64>::from_elem(n, baseline);
        let mut estimators: Vec<DecisionTreeRegressor> = Vec::with_capacity(n_estimators);
        for _ in 0..n_estimators {
            let mut residual = Array1::<f64>::zeros(n);
            for i in 0..n {
                let p = sigmoid(f[i]);
                residual[i] = y[i] as f64 - p;
            }
            let tree = DecisionTreeRegressor::fit(x, residual.view(), RegressionCriterion::Mse, params)?;
            let pred = tree.predict(x)?;
            for i in 0..n {
                f[i] += learning_rate * pred[i];
            }
            estimators.push(tree);
        }
        Ok(Self { baseline, estimators, learning_rate, n_estimators, params })
    }

    /// Predicted probability of class 1.
    pub fn predict_proba1(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let logits = self.decision_function(x)?;
        Ok(logits.map(|z| sigmoid(*z)))
    }

    /// Decision-function value (log-odds) per row.
    pub fn decision_function(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let mut out = Array1::<f64>::from_elem(n, self.baseline);
        for tree in &self.estimators {
            let p = tree.predict(x)?;
            for i in 0..n {
                out[i] += self.learning_rate * p[i];
            }
        }
        Ok(out)
    }

    /// Predicted labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<u8>> {
        Ok(self.predict_proba1(x)?.map(|p| if *p >= 0.5 { 1 } else { 0 }))
    }
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

// Suppress unused-import warning if Array2 isn't used downstream.
#[allow(dead_code)]
fn _touch(m: Array2<f64>) -> Array2<f64> {
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn gradient_boosting_classifier_separates_two_clusters() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.2], [5.3, 5.3]
        ];
        let y = array![0_u8, 0, 0, 0, 1, 1, 1, 1];
        let m = GradientBoostingClassifier::fit(x.view(), y.view()).unwrap();
        let p = m.predict(x.view()).unwrap();
        for i in 0..4 {
            assert_eq!(p[i], 0);
        }
        for i in 4..8 {
            assert_eq!(p[i], 1);
        }
    }
}

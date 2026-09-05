//! [`GradientBoostingRegressor`] — Friedman (2001) stagewise additive
//! regression on the least-squares loss.
//!
//! The initial model is the constant `F₀(x) = mean(y)`. At stage `m`,
//! we fit a CART regressor `h_m` to the current residuals
//! `y − F_{m−1}(x)` and update
//!
//! ```text
//! F_m(x) = F_{m−1}(x) + η · h_m(x)
//! ```
//!
//! where `η ∈ (0, 1]` is the `learning_rate`. Under the least-squares
//! loss the pseudo-residuals coincide with the ordinary residuals, so
//! the estimator reduces to the classical LS boosting algorithm.
//!
//! # Complexity
//!
//! `O(M · n · d · log n)` for `M` stages, `n` samples, `d` features.
//! Space: one tree per stage plus the constant.

use ndarray::{Array1, ArrayView1, ArrayView2};
use solow_core::{Error, Result};
use solow_tree::{DecisionTreeRegressor, RegressionCriterion, TreeParams};

/// Stagewise gradient-boosted regressor for the least-squares loss.
#[derive(Clone, Debug)]
pub struct GradientBoostingRegressor {
    /// Baseline (mean of y under LS).
    pub baseline: f64,
    /// Per-stage regressor.
    pub estimators: Vec<DecisionTreeRegressor>,
    /// Shrinkage factor `η ∈ (0, 1]`.
    pub learning_rate: f64,
    /// Number of stages fit.
    pub n_estimators: usize,
    /// CART growth parameters used at every stage.
    pub params: TreeParams,
}

impl GradientBoostingRegressor {
    /// Fit `M = n_estimators` shrunk trees on the running residuals.
    ///
    /// * `params.max_depth` — controls per-stage complexity. `3` is the
    ///   Friedman-recommended default; the reference defaults to `3` as well.
    /// * `learning_rate` — shrinkage. `0.1` is the standard trade-off
    ///   with `M ≈ 100`; smaller `η` needs a proportionally larger `M`.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        n_estimators: usize,
        learning_rate: f64,
        params: TreeParams,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "GradientBoostingRegressor::fit: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        if n_estimators == 0 {
            return Err(Error::Value(
                "GradientBoostingRegressor::fit: n_estimators must be ≥ 1".into(),
            ));
        }
        if !(learning_rate > 0.0 && learning_rate <= 1.0) {
            return Err(Error::Value(format!(
                "GradientBoostingRegressor::fit: learning_rate must be in (0, 1] (got {learning_rate})"
            )));
        }
        let baseline: f64 = y.iter().sum::<f64>() / y.len() as f64;
        let mut residuals: Array1<f64> = y.mapv(|v| v - baseline);
        let mut estimators = Vec::with_capacity(n_estimators);
        for _ in 0..n_estimators {
            let tree =
                DecisionTreeRegressor::fit(x, residuals.view(), RegressionCriterion::Mse, params)?;
            let pred = tree.predict(x)?;
            for i in 0..residuals.len() {
                residuals[i] -= learning_rate * pred[i];
            }
            estimators.push(tree);
        }
        Ok(Self {
            baseline,
            estimators,
            learning_rate,
            n_estimators,
            params,
        })
    }

    /// Predict `F_M(x) = F_0 + η · Σ_m h_m(x)`.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let mut out = Array1::<f64>::from_elem(x.nrows(), self.baseline);
        for est in &self.estimators {
            let p = est.predict(x)?;
            for i in 0..x.nrows() {
                out[i] += self.learning_rate * p[i];
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn boosted_fit_beats_baseline_mse() {
        // Ground truth y = 2x + noise.
        let n = 80usize;
        let x: ndarray::Array2<f64> =
            ndarray::Array2::from_shape_vec((n, 1), (0..n).map(|i| i as f64 * 0.1).collect())
                .unwrap();
        let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v + ((v * 7.3).sin()) * 0.05);
        let params = TreeParams::default().max_depth(3);
        let gb = GradientBoostingRegressor::fit(x.view(), y.view(), 60, 0.1, params).unwrap();
        let pred = gb.predict(x.view()).unwrap();
        let mse: f64 = pred
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;
        // Baseline MSE is Var(y); boosted MSE must be much smaller.
        let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
        let var_y: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / n as f64;
        assert!(
            mse < 0.1 * var_y,
            "boosted MSE {mse} vs baseline var {var_y}"
        );
    }

    #[test]
    fn zero_estimators_rejected() {
        let x = array![[1.0]];
        let y = array![1.0];
        assert!(
            GradientBoostingRegressor::fit(x.view(), y.view(), 0, 0.1, TreeParams::default())
                .is_err()
        );
    }
}

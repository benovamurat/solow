//! `LassoLarsIC` — LARS path with an information-criterion stop rule
//! (Efron-Hastie-Johnstone-Tibshirani 2004, §7). Picks the sparsity
//! level that minimises AIC or BIC on the training residual.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

use crate::lars::Lars;

/// Information criterion used for model selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InformationCriterion {
    /// Akaike Information Criterion.
    Aic,
    /// Bayesian Information Criterion.
    Bic,
}

/// Fitted `LassoLarsIC`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LassoLarsIC {
    /// Coefficients at the winning sparsity level.
    pub coef: Array1<f64>,
    /// Intercept.
    pub intercept: f64,
    /// Information-criterion trajectory (one value per LARS step).
    pub criterion_path: Array1<f64>,
    /// Winning step (index into `coef_path`).
    pub best_step: usize,
    /// Criterion used.
    pub criterion: InformationCriterion,
}

impl LassoLarsIC {
    /// Fit with the given criterion.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        criterion: InformationCriterion,
    ) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("LassoLarsIC: y/x row mismatch".into()));
        }
        let lars = Lars::fit(x, y)?;
        // Estimate σ² from the full-model residual.
        let mut best_step = 0_usize;
        let mut best_ic = f64::INFINITY;
        let mut ic_path = Array1::<f64>::zeros(lars.coef_path.len());
        let (_x_mean, _y_mean, y_var) = center_stats(x, y);
        // σ² estimate from OLS on all features (last LARS step).
        let sigma2 = residual_variance(x, y, &lars.coef_path[lars.coef_path.len() - 1]);
        let sigma2 = sigma2.max(1e-30);
        for (step, beta) in lars.coef_path.iter().enumerate() {
            let rss = residual_ss(x, y, beta);
            let k = beta.iter().filter(|c| c.abs() > 1e-10).count() as f64;
            let ic = match criterion {
                InformationCriterion::Aic => rss / sigma2 + 2.0 * k,
                InformationCriterion::Bic => rss / sigma2 + (n as f64).ln() * k,
            };
            ic_path[step] = ic;
            if ic < best_ic {
                best_ic = ic;
                best_step = step;
            }
        }
        // Compute intercept from the winning coefficients on centred data.
        let coef = lars.coef_path[best_step].clone();
        let mut mean_x = vec![0.0_f64; x.ncols()];
        for j in 0..x.ncols() {
            let mut s = 0.0_f64;
            for i in 0..n {
                s += x[[i, j]];
            }
            mean_x[j] = s / n as f64;
        }
        let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
        let mut intercept = mean_y;
        for j in 0..x.ncols() {
            intercept -= mean_x[j] * coef[j];
        }
        // Suppress unused warnings on the helper.
        let _ = y_var;
        Ok(Self {
            coef,
            intercept,
            criterion_path: ic_path,
            best_step,
            criterion,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        if d != self.coef.len() {
            return Err(Error::Shape("LassoLarsIC::predict: shape mismatch".into()));
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = self.intercept;
            for j in 0..d {
                s += x[[i, j]] * self.coef[j];
            }
            out[i] = s;
        }
        Ok(out)
    }
}

fn center_stats(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> (Vec<f64>, f64, f64) {
    let n = x.nrows() as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    let var_y: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / n.max(1.0);
    let mut mean_x = vec![0.0_f64; x.ncols()];
    for j in 0..x.ncols() {
        for i in 0..(n as usize) {
            mean_x[j] += x[[i, j]];
        }
        mean_x[j] /= n;
    }
    (mean_x, mean_y, var_y)
}

fn residual_ss(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>, beta: &Array1<f64>) -> f64 {
    let n = x.nrows();
    let d = x.ncols();
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
    // Compute residual using the constant fit intercept = mean(y) − Σ mean(x_j) β_j.
    let mut mean_x = vec![0.0_f64; d];
    for j in 0..d {
        let mut s = 0.0_f64;
        for i in 0..n {
            s += x[[i, j]];
        }
        mean_x[j] = s / n as f64;
    }
    let mut intercept = mean_y;
    for j in 0..d {
        intercept -= mean_x[j] * beta[j];
    }
    let mut rss = 0.0_f64;
    for i in 0..n {
        let mut yhat = intercept;
        for j in 0..d {
            yhat += x[[i, j]] * beta[j];
        }
        rss += (y[i] - yhat).powi(2);
    }
    rss
}

fn residual_variance(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>, beta: &Array1<f64>) -> f64 {
    let n = x.nrows() as f64;
    let d = beta.iter().filter(|c| c.abs() > 1e-10).count() as f64;
    let rss = residual_ss(x, y, beta);
    rss / (n - d - 1.0).max(1.0)
}

// Prevent unused-import warning.
#[allow(dead_code)]
fn _touch(a: Array2<f64>) -> Array2<f64> {
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn lasso_lars_ic_returns_a_valid_step() {
        let x = array![
            [1.0_f64, 2.0, 3.0], [2.0, 3.0, 5.0], [3.0, 5.0, 8.0],
            [4.0, 7.0, 11.0], [5.0, 9.0, 14.0]
        ];
        let y = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let m = LassoLarsIC::fit(x.view(), y.view(), InformationCriterion::Bic).unwrap();
        assert!(m.best_step < m.criterion_path.len());
    }
}

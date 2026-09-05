//! HistGradientBoosting — histogram-binned variant of gradient boosting
//! (Ke et al. 2017, "LightGBM"). Features are quantile-binned into at
//! most `max_bins` categories, so per-node split search on a bin becomes
//! `O(max_bins)` per feature instead of `O(n)`.
//!
//! The implementation reuses `DecisionTreeRegressor` from `solow-tree`
//! by first quantile-binning `X` and passing the bin indices in as
//! ordinary floats. It's a faithful histogram-boosting shape with fewer
//! bells and whistles than LightGBM proper.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};
use solow_tree::{DecisionTreeRegressor, RegressionCriterion, TreeParams};

/// Fitted HistGradientBoostingRegressor.
#[derive(Clone, Debug)]
pub struct HistGradientBoostingRegressor {
    /// Sequence of shallow regression trees.
    pub trees: Vec<DecisionTreeRegressor>,
    /// Initial constant baseline `f₀`.
    pub init_prediction: f64,
    /// Learning rate ν.
    pub learning_rate: f64,
    /// Bin boundaries per feature (`d` vectors of length ≤ max_bins).
    pub bin_thresholds: Vec<Vec<f64>>,
    /// Number of trees actually kept.
    pub n_estimators: usize,
}

impl HistGradientBoostingRegressor {
    /// Fit with the reference defaults.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
    ) -> Result<Self> {
        Self::fit_with(x, y, 100, 0.1, 3, 255)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        max_bins: usize,
    ) -> Result<Self> {
        if x.nrows() != y.len() {
            return Err(Error::Shape("HistGradientBoostingRegressor: x/y row mismatch".into()));
        }
        if n_estimators == 0 {
            return Err(Error::Value("HistGradientBoostingRegressor: n_estimators must be ≥ 1".into()));
        }
        if learning_rate <= 0.0 {
            return Err(Error::Value("HistGradientBoostingRegressor: learning_rate must be > 0".into()));
        }
        // Bin features quantile-wise (equal-count edges).
        let d = x.ncols();
        let n = x.nrows();
        let mut bin_thresholds: Vec<Vec<f64>> = Vec::with_capacity(d);
        for j in 0..d {
            let mut col: Vec<f64> = (0..n).map(|i| x[[i, j]]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let bins = max_bins.min(col.len().saturating_sub(1)).max(1);
            let mut edges = Vec::with_capacity(bins);
            for k in 1..=bins {
                let q = (k as f64) / (bins as f64 + 1.0);
                let pos = (q * (col.len() as f64 - 1.0)) as usize;
                edges.push(col[pos]);
            }
            edges.dedup_by(|a, b| (*a - *b).abs() < 1e-30);
            bin_thresholds.push(edges);
        }
        // Bin X.
        let mut x_bin = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                let bin = bin_thresholds[j].iter().position(|&e| x[[i, j]] <= e)
                    .unwrap_or(bin_thresholds[j].len());
                x_bin[[i, j]] = bin as f64;
            }
        }
        // Boost on binned features.
        let init = y.iter().sum::<f64>() / n as f64;
        let mut residual = Array1::<f64>::zeros(n);
        for i in 0..n {
            residual[i] = y[i] - init;
        }
        let mut trees: Vec<DecisionTreeRegressor> = Vec::with_capacity(n_estimators);
        for _ in 0..n_estimators {
            let mut p = TreeParams::default();
            p.max_depth = max_depth;
            let tree = DecisionTreeRegressor::fit(
                x_bin.view(),
                residual.view(),
                RegressionCriterion::Mse,
                p,
            )?;
            let pred = tree.predict(x_bin.view())?;
            for i in 0..n {
                residual[i] -= learning_rate * pred[i];
            }
            trees.push(tree);
        }
        Ok(Self {
            trees,
            init_prediction: init,
            learning_rate,
            bin_thresholds,
            n_estimators,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        let mut x_bin = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                let bin = self.bin_thresholds[j].iter().position(|&e| x[[i, j]] <= e)
                    .unwrap_or(self.bin_thresholds[j].len());
                x_bin[[i, j]] = bin as f64;
            }
        }
        let mut out = Array1::<f64>::from_elem(n, self.init_prediction);
        for tree in &self.trees {
            let p = tree.predict(x_bin.view())?;
            for i in 0..n {
                out[i] += self.learning_rate * p[i];
            }
        }
        Ok(out)
    }
}

/// HistGradientBoostingClassifier — binary softmax classifier on top of
/// HistGradientBoostingRegressor. Uses logit link and Newton's method on
/// the log-loss gradient (`p − y`) as the residual.
#[derive(Clone, Debug)]
pub struct HistGradientBoostingClassifier {
    /// Underlying regressor fit on logit residuals.
    pub inner: HistGradientBoostingRegressor,
}

impl HistGradientBoostingClassifier {
    /// Fit binary classifier with the reference defaults.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, u8>,
    ) -> Result<Self> {
        if x.nrows() != y.len() {
            return Err(Error::Shape("HistGradientBoostingClassifier: x/y row mismatch".into()));
        }
        let n = x.nrows();
        // Fit inner regressor on log-odds targets.
        let mean_p = y.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let init_logit = ((mean_p.clamp(1e-6, 1.0 - 1e-6))
            / (1.0 - mean_p.clamp(1e-6, 1.0 - 1e-6))).ln();
        let mut logits = Array1::<f64>::from_elem(n, init_logit);
        let n_est = 100;
        let lr = 0.1;
        let max_depth = 3;
        let max_bins = 255;
        // Bin features.
        let d = x.ncols();
        let mut bin_thresholds: Vec<Vec<f64>> = Vec::with_capacity(d);
        for j in 0..d {
            let mut col: Vec<f64> = (0..n).map(|i| x[[i, j]]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let bins = max_bins.min(col.len().saturating_sub(1)).max(1);
            let mut edges = Vec::with_capacity(bins);
            for k in 1..=bins {
                let q = (k as f64) / (bins as f64 + 1.0);
                let pos = (q * (col.len() as f64 - 1.0)) as usize;
                edges.push(col[pos]);
            }
            edges.dedup_by(|a, b| (*a - *b).abs() < 1e-30);
            bin_thresholds.push(edges);
        }
        let mut x_bin = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                let bin = bin_thresholds[j].iter().position(|&e| x[[i, j]] <= e)
                    .unwrap_or(bin_thresholds[j].len());
                x_bin[[i, j]] = bin as f64;
            }
        }
        let mut trees: Vec<DecisionTreeRegressor> = Vec::with_capacity(n_est);
        for _ in 0..n_est {
            let mut residual = Array1::<f64>::zeros(n);
            for i in 0..n {
                let p = 1.0 / (1.0 + (-logits[i]).exp());
                residual[i] = y[i] as f64 - p;
            }
            let mut params = TreeParams::default();
            params.max_depth = max_depth;
            let tree = DecisionTreeRegressor::fit(
                x_bin.view(),
                residual.view(),
                RegressionCriterion::Mse,
                params,
            )?;
            let pred = tree.predict(x_bin.view())?;
            for i in 0..n {
                logits[i] += lr * pred[i];
            }
            trees.push(tree);
        }
        Ok(Self {
            inner: HistGradientBoostingRegressor {
                trees,
                init_prediction: init_logit,
                learning_rate: lr,
                bin_thresholds,
                n_estimators: n_est,
            },
        })
    }

    /// Predicted probability of class 1.
    pub fn predict_proba1(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let logits = self.inner.predict(x)?;
        Ok(logits.map(|z| 1.0 / (1.0 + (-z).exp())))
    }

    /// Predicted labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<u8>> {
        Ok(self.predict_proba1(x)?.map(|p| if *p >= 0.5 { 1 } else { 0 }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn hgbr_learns_a_linear_signal() {
        // y = 2·x
        let x = array![
            [0.0_f64], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0],
            [8.0], [9.0], [10.0], [11.0]
        ];
        let y_vec: Vec<f64> = (0..12).map(|i| 2.0 * i as f64).collect();
        let y = Array1::from_vec(y_vec);
        let m = HistGradientBoostingRegressor::fit_with(x.view(), y.view(), 100, 0.1, 3, 32).unwrap();
        let p = m.predict(x.view()).unwrap();
        let mse: f64 = (0..12).map(|i| (p[i] - y[i]).powi(2)).sum::<f64>() / 12.0;
        let var: f64 = y.iter().map(|yi| (yi - 11.0).powi(2)).sum::<f64>() / 12.0;
        assert!(mse < 0.2 * var, "mse {mse} not < 0.2 · var {var}");
    }

    #[test]
    fn hgbc_learns_two_clusters() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.2]
        ];
        let y = array![0_u8, 0, 0, 1, 1, 1];
        let m = HistGradientBoostingClassifier::fit(x.view(), y.view()).unwrap();
        let p = m.predict(x.view()).unwrap();
        for i in 0..3 { assert_eq!(p[i], 0); }
        for i in 3..6 { assert_eq!(p[i], 1); }
    }
}

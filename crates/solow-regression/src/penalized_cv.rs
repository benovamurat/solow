//! Cross-validated penalised-regression variants:
//! [`RidgeCV`], [`LassoCV`], [`ElasticNetCV`].
//!
//! Each searches over a user-supplied `alpha` grid (and, for
//! `ElasticNetCV`, an `l1_ratio` grid) using solow-cv-style K-fold
//! cross-validation with the mean-squared-error criterion, refits the
//! full estimator at the best-CV `alpha`, and reports both the
//! selected hyperparameter and the per-alpha CV score.
//!
//! # References
//!
//! * Hoerl-Kennard (1970), Tibshirani (1996), Zou-Hastie (2005) — see
//!   [`crate::penalized`] for the underlying estimators.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

use crate::penalized::{ElasticNet, Lasso, Ridge};

/// Cross-validated Ridge regression.
#[derive(Clone, Debug)]
pub struct RidgeCV {
    /// The fitted [`Ridge`] at the CV-optimal α.
    pub fit: Ridge,
    /// α values searched, in the order supplied.
    pub alphas: Vec<f64>,
    /// Mean CV MSE per α (lower is better).
    pub mean_cv_mse: Vec<f64>,
    /// Best α (minimises mean CV MSE).
    pub best_alpha: f64,
}

impl RidgeCV {
    /// Fit RidgeCV over the given `alphas` with `k`-fold cross-validation.
    pub fn fit(
        y: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
        alphas: &[f64],
        k: usize,
        fit_intercept: bool,
    ) -> Result<Self> {
        if alphas.is_empty() {
            return Err(Error::Value(
                "RidgeCV::fit: alphas must be non-empty".into(),
            ));
        }
        if k < 2 || k > y.len() {
            return Err(Error::Value(format!(
                "RidgeCV::fit: k must be in [2, n] (got {k})"
            )));
        }
        let folds = kfold_indices(y.len(), k);
        let mut mean_cv_mse = Vec::with_capacity(alphas.len());
        for &alpha in alphas {
            let mut sum_mse = 0.0_f64;
            for (train, test) in &folds {
                let (y_tr, x_tr) = subset(y, x, train);
                let (y_te, x_te) = subset(y, x, test);
                let fit = Ridge::fit(y_tr.view(), x_tr.view(), alpha, fit_intercept)?;
                let pred = fit.predict(x_te.view())?;
                let mse: f64 = pred
                    .iter()
                    .zip(y_te.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / y_te.len() as f64;
                sum_mse += mse;
            }
            mean_cv_mse.push(sum_mse / folds.len() as f64);
        }
        let (best_i, _) = argmin(&mean_cv_mse);
        let best_alpha = alphas[best_i];
        let fit = Ridge::fit(y, x, best_alpha, fit_intercept)?;
        Ok(Self {
            fit,
            alphas: alphas.to_vec(),
            mean_cv_mse,
            best_alpha,
        })
    }
}

/// Cross-validated Lasso.
#[derive(Clone, Debug)]
pub struct LassoCV {
    /// Fitted Lasso at the best α.
    pub fit: Lasso,
    /// α values searched.
    pub alphas: Vec<f64>,
    /// Mean CV MSE per α.
    pub mean_cv_mse: Vec<f64>,
    /// Best α.
    pub best_alpha: f64,
}

impl LassoCV {
    /// Fit LassoCV.
    pub fn fit(
        y: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
        alphas: &[f64],
        k: usize,
        fit_intercept: bool,
    ) -> Result<Self> {
        if alphas.is_empty() {
            return Err(Error::Value(
                "LassoCV::fit: alphas must be non-empty".into(),
            ));
        }
        if k < 2 || k > y.len() {
            return Err(Error::Value(format!(
                "LassoCV::fit: k must be in [2, n] (got {k})"
            )));
        }
        let folds = kfold_indices(y.len(), k);
        let mut mean_cv_mse = Vec::with_capacity(alphas.len());
        for &alpha in alphas {
            let mut sum_mse = 0.0_f64;
            for (train, test) in &folds {
                let (y_tr, x_tr) = subset(y, x, train);
                let (y_te, x_te) = subset(y, x, test);
                let fit = Lasso::fit(y_tr.view(), x_tr.view(), alpha, fit_intercept)?;
                let pred = fit.predict(x_te.view())?;
                let mse: f64 = pred
                    .iter()
                    .zip(y_te.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / y_te.len() as f64;
                sum_mse += mse;
            }
            mean_cv_mse.push(sum_mse / folds.len() as f64);
        }
        let (best_i, _) = argmin(&mean_cv_mse);
        let best_alpha = alphas[best_i];
        let fit = Lasso::fit(y, x, best_alpha, fit_intercept)?;
        Ok(Self {
            fit,
            alphas: alphas.to_vec(),
            mean_cv_mse,
            best_alpha,
        })
    }
}

/// Cross-validated ElasticNet with a 2-D `(alpha, l1_ratio)` grid.
#[derive(Clone, Debug)]
pub struct ElasticNetCV {
    /// Fitted ElasticNet at the best `(α, ρ)`.
    pub fit: ElasticNet,
    /// α grid searched.
    pub alphas: Vec<f64>,
    /// `l1_ratio` grid searched.
    pub l1_ratios: Vec<f64>,
    /// Mean CV MSE per `(alpha_idx, l1_ratio_idx)` — row-major.
    pub mean_cv_mse: Vec<Vec<f64>>,
    /// Best `(alpha, l1_ratio)` pair.
    pub best_alpha: f64,
    /// See [`Self::best_alpha`].
    pub best_l1_ratio: f64,
}

impl ElasticNetCV {
    /// Fit ElasticNetCV over the product grid.
    pub fn fit(
        y: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
        alphas: &[f64],
        l1_ratios: &[f64],
        k: usize,
        fit_intercept: bool,
    ) -> Result<Self> {
        if alphas.is_empty() || l1_ratios.is_empty() {
            return Err(Error::Value(
                "ElasticNetCV::fit: alphas and l1_ratios must be non-empty".into(),
            ));
        }
        if k < 2 || k > y.len() {
            return Err(Error::Value(format!(
                "ElasticNetCV::fit: k must be in [2, n] (got {k})"
            )));
        }
        let folds = kfold_indices(y.len(), k);
        let mut grid: Vec<Vec<f64>> = Vec::with_capacity(alphas.len());
        let (mut best_i, mut best_j) = (0usize, 0usize);
        let mut best_score = f64::INFINITY;
        for (i, &alpha) in alphas.iter().enumerate() {
            let mut row = Vec::with_capacity(l1_ratios.len());
            for (j, &rho) in l1_ratios.iter().enumerate() {
                let mut sum_mse = 0.0_f64;
                for (train, test) in &folds {
                    let (y_tr, x_tr) = subset(y, x, train);
                    let (y_te, x_te) = subset(y, x, test);
                    let fit = ElasticNet::fit(y_tr.view(), x_tr.view(), alpha, rho, fit_intercept)?;
                    let pred = fit.predict(x_te.view())?;
                    let mse: f64 = pred
                        .iter()
                        .zip(y_te.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        / y_te.len() as f64;
                    sum_mse += mse;
                }
                let mean = sum_mse / folds.len() as f64;
                row.push(mean);
                if mean < best_score {
                    best_score = mean;
                    best_i = i;
                    best_j = j;
                }
            }
            grid.push(row);
        }
        let best_alpha = alphas[best_i];
        let best_l1_ratio = l1_ratios[best_j];
        let fit = ElasticNet::fit(y, x, best_alpha, best_l1_ratio, fit_intercept)?;
        Ok(Self {
            fit,
            alphas: alphas.to_vec(),
            l1_ratios: l1_ratios.to_vec(),
            mean_cv_mse: grid,
            best_alpha,
            best_l1_ratio,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers — a self-contained K-fold splitter so this module doesn't need a
// dependency on solow-cv (which would be a cycle: solow-cv would then depend
// on solow-regression via consumers).
// ---------------------------------------------------------------------------

fn kfold_indices(n: usize, k: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut out = Vec::with_capacity(k);
    for fold in 0..k {
        let start = fold * n / k;
        let end = (fold + 1) * n / k;
        let test: Vec<usize> = (start..end).collect();
        let train: Vec<usize> = (0..n).filter(|i| !(start..end).contains(i)).collect();
        out.push((train, test));
    }
    out
}

fn subset(
    y: ArrayView1<'_, f64>,
    x: ArrayView2<'_, f64>,
    idx: &[usize],
) -> (Array1<f64>, Array2<f64>) {
    let d = x.ncols();
    let mut sub_y = Array1::<f64>::zeros(idx.len());
    let mut sub_x = Array2::<f64>::zeros((idx.len(), d));
    for (r, &i) in idx.iter().enumerate() {
        sub_y[r] = y[i];
        for j in 0..d {
            sub_x[[r, j]] = x[[i, j]];
        }
    }
    (sub_y, sub_x)
}

fn argmin(v: &[f64]) -> (usize, f64) {
    let mut best_i = 0usize;
    let mut best_v = f64::INFINITY;
    for (i, &val) in v.iter().enumerate() {
        if val < best_v {
            best_v = val;
            best_i = i;
        }
    }
    (best_i, best_v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ridgecv_picks_a_reasonable_alpha() {
        // Small linear signal — RidgeCV over a wide α grid should land
        // near α = 0 (small regularisation).
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0]
        ];
        let y = x.column(0).mapv(|v| 2.0 * v + 1.0);
        let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
        let cv = RidgeCV::fit(y.view(), x.view(), &alphas, 3, true).unwrap();
        assert!(alphas.contains(&cv.best_alpha));
        // Fit predictions are close to y.
        let pred = cv.fit.predict(x.view()).unwrap();
        let mse: f64 = pred
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;
        let var_y: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / y.len() as f64;
        assert!(mse < 0.1 * var_y, "MSE = {mse}");
    }

    #[test]
    fn lassocv_returns_a_valid_alpha_from_the_grid() {
        let x = array![
            [1.0, 5.0],
            [2.0, -3.0],
            [3.0, 4.0],
            [4.0, 0.0],
            [5.0, 2.0],
            [6.0, -1.0],
            [7.0, 3.0],
            [8.0, 0.5],
            [9.0, -2.0]
        ];
        let y = x.column(0).mapv(|v| 3.0 * v + 1.0);
        let alphas = vec![0.01, 0.1, 1.0];
        let cv = LassoCV::fit(y.view(), x.view(), &alphas, 3, true).unwrap();
        assert!(alphas.contains(&cv.best_alpha));
        // We have `alphas.len()` mean-MSE entries.
        assert_eq!(cv.mean_cv_mse.len(), alphas.len());
    }

    #[test]
    fn elasticnetcv_searches_the_full_grid() {
        let x = array![
            [1.0, 5.0],
            [2.0, -3.0],
            [3.0, 4.0],
            [4.0, 0.0],
            [5.0, 2.0],
            [6.0, -1.0],
            [7.0, 3.0],
            [8.0, 0.5],
            [9.0, -2.0]
        ];
        let y = x.column(0).mapv(|v| 3.0 * v + 1.0);
        let alphas = vec![0.01, 0.1];
        let l1s = vec![0.0, 0.5, 1.0];
        let cv = ElasticNetCV::fit(y.view(), x.view(), &alphas, &l1s, 3, true).unwrap();
        assert!(alphas.contains(&cv.best_alpha));
        assert!(l1s.contains(&cv.best_l1_ratio));
        assert_eq!(cv.mean_cv_mse.len(), alphas.len());
        assert_eq!(cv.mean_cv_mse[0].len(), l1s.len());
    }
}

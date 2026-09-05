//! `AdaBoostRegressor` — Drucker's (1997) AdaBoost.R2 for regression.
//!
//! At each round we fit a `DecisionTreeRegressor` on the current sample
//! weights, compute per-sample loss ratios, derive the round weight β,
//! and reweight samples toward the harder ones. Predictions are the
//! weighted median of the base predictions (Drucker's recipe).

use ndarray::{Array1, ArrayView1, ArrayView2};
use solow_core::{Error, Result};
use solow_tree::{DecisionTreeRegressor, RegressionCriterion, TreeParams};

use crate::{lcg_next, uniform_index};

/// A single AdaBoost.R2 round's contribution.
#[derive(Clone, Debug)]
struct Round {
    tree: DecisionTreeRegressor,
    beta: f64,
}

/// Fitted `AdaBoostRegressor`.
#[derive(Clone, Debug)]
pub struct AdaBoostRegressor {
    rounds: Vec<Round>,
    /// Number of stages fit.
    pub n_estimators: usize,
    /// Learning rate `η ∈ (0, 1]` applied to `log(1/β)` at combine time.
    pub learning_rate: f64,
    /// Random seed used to draw weighted samples.
    pub seed: u64,
}

impl AdaBoostRegressor {
    /// Fit with the reference defaults `n_estimators = 50`, `learning_rate = 1.0`,
    /// `max_depth = 3`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>, seed: u64) -> Result<Self> {
        let mut params = TreeParams::default();
        params.max_depth = 3;
        Self::fit_with(x, y, 50, 1.0, params, seed)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        n_estimators: usize,
        learning_rate: f64,
        params: TreeParams,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("AdaBoostRegressor: y/x length mismatch".into()));
        }
        if n_estimators == 0 {
            return Err(Error::Value("AdaBoostRegressor: n_estimators must be ≥ 1".into()));
        }
        let mut weights = vec![1.0_f64 / n as f64; n];
        let mut rounds: Vec<Round> = Vec::new();
        let mut state = seed.wrapping_add(0xADA_D00D);
        for _ in 0..n_estimators {
            // Weighted resample via inverse-CDF on the cumulative weights.
            let mut cdf = vec![0.0_f64; n];
            let mut s = 0.0_f64;
            for i in 0..n {
                s += weights[i];
                cdf[i] = s;
            }
            let mut sub_rows = Vec::with_capacity(n);
            for _ in 0..n {
                let u = uniform_f64(&mut state) * cdf[n - 1];
                // Binary search on the CDF.
                let mut lo = 0_usize;
                let mut hi = n - 1;
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if cdf[mid] < u {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                sub_rows.push(lo);
            }
            let (sx, sy) = row_subset(x, y, &sub_rows);
            let tree = DecisionTreeRegressor::fit(sx.view(), sy.view(), RegressionCriterion::Mse, params)?;
            let pred = tree.predict(x)?;
            let mut abs_err = vec![0.0_f64; n];
            let mut max_err = 0.0_f64;
            for i in 0..n {
                let e = (y[i] - pred[i]).abs();
                abs_err[i] = e;
                if e > max_err {
                    max_err = e;
                }
            }
            if max_err <= 1e-30 {
                rounds.push(Round { tree, beta: 1e-30 });
                break;
            }
            let mut e_bar = 0.0_f64;
            for i in 0..n {
                let loss = abs_err[i] / max_err;
                e_bar += weights[i] * loss;
            }
            if e_bar >= 0.5 {
                // Discard round; keep training-set weights.
                if rounds.is_empty() {
                    rounds.push(Round { tree, beta: 1.0 });
                }
                break;
            }
            let beta = e_bar / (1.0 - e_bar);
            let mut sum_w = 0.0_f64;
            for i in 0..n {
                let loss = abs_err[i] / max_err;
                weights[i] *= beta.powf(learning_rate * (1.0 - loss));
                sum_w += weights[i];
            }
            for w in weights.iter_mut() {
                *w /= sum_w.max(1e-30);
            }
            rounds.push(Round { tree, beta });
        }
        let actual = rounds.len();
        // Suppress unused import warning at the workspace level.
        let _ = lcg_next;
        let _ = uniform_index;
        Ok(Self {
            rounds,
            n_estimators: actual,
            learning_rate,
            seed,
        })
    }

    /// Predict via the weighted median (Drucker 1997).
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let m = self.rounds.len();
        if m == 0 {
            return Err(Error::Value("AdaBoostRegressor::predict: model has no rounds".into()));
        }
        // Per-round predictions and log(1/β) weights.
        let mut per_round_pred = Vec::with_capacity(m);
        let mut log_inv_beta = Vec::with_capacity(m);
        for r in &self.rounds {
            per_round_pred.push(r.tree.predict(x)?);
            log_inv_beta.push((1.0 / r.beta.max(1e-30)).ln());
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            // Sort (prediction, weight) pairs by prediction; pick the
            // weighted median.
            let mut pairs: Vec<(f64, f64)> = (0..m)
                .map(|k| (per_round_pred[k][i], log_inv_beta[k]))
                .collect();
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let total: f64 = pairs.iter().map(|(_, w)| *w).sum();
            let mut acc = 0.0_f64;
            let mut median = pairs[0].0;
            for (p, w) in &pairs {
                acc += *w;
                if acc >= 0.5 * total {
                    median = *p;
                    break;
                }
            }
            out[i] = median;
        }
        Ok(out)
    }
}

fn row_subset(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    rows: &[usize],
) -> (ndarray::Array2<f64>, Array1<f64>) {
    let d = x.ncols();
    let mut xs = ndarray::Array2::<f64>::zeros((rows.len(), d));
    let mut ys = Array1::<f64>::zeros(rows.len());
    for (r, &i) in rows.iter().enumerate() {
        for j in 0..d {
            xs[[r, j]] = x[[i, j]];
        }
        ys[r] = y[i];
    }
    (xs, ys)
}

fn uniform_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let r = *state >> 11;
    (r as f64) * f64::from_bits(0x3CA0_0000_0000_0000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn adaboost_regressor_recovers_a_linear_signal() {
        let x = array![
            [1.0_f64], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]
        ];
        let y = array![2.0_f64, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let m = AdaBoostRegressor::fit(x.view(), y.view(), 42).unwrap();
        let p = m.predict(x.view()).unwrap();
        let mse: f64 = (0..10).map(|i| (p[i] - y[i]).powi(2)).sum::<f64>() / 10.0;
        assert!(mse < 5.0, "mse = {mse}");
    }
}

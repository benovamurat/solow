//! [`AdaBoostClassifier`] — Freund & Schapire (1997) SAMME adaptive
//! boosting for binary and multiclass targets, using CART stumps as
//! weak learners.
//!
//! # Algorithm (SAMME, Zhu-Zou-Rosset-Hastie 2009)
//!
//! Initialise sample weights `w_i = 1/n`. For `m = 1..M`:
//!
//! 1. Fit a weak learner `h_m` on the weighted sample (a CART
//!    decision stump — a tree of depth 1 — is the classical choice).
//! 2. Compute the weighted error `err_m = Σ_i w_i · [h_m(x_i) ≠ y_i]
//!    / Σ_i w_i`.
//! 3. If `err_m ≥ (K − 1) / K` for `K` classes, stop (worse than
//!    random).
//! 4. Set `α_m = log((1 − err_m) / err_m) + log(K − 1)`.
//! 5. Update `w_i ← w_i · exp(α_m · [h_m(x_i) ≠ y_i])` and renormalise.
//!
//! At predict time, the estimator sums per-class `α_m` votes and
//! returns the argmax. This reduces to real AdaBoost.M1 for `K = 2`.
//!
//! Sample weights are realised by re-drawing bootstrap indices from
//! the multinomial `w` at each stage (matches the reference default when
//! the weak learner does not natively accept `sample_weight`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};
use solow_tree::{ClassificationCriterion, DecisionTreeClassifier, TreeParams};

/// AdaBoost (SAMME) classifier with CART weak learners.
#[derive(Clone, Debug)]
pub struct AdaBoostClassifier {
    /// Per-stage weak learner.
    pub estimators: Vec<DecisionTreeClassifier>,
    /// Per-stage `α_m` coefficient.
    pub alphas: Vec<f64>,
    /// Number of distinct classes.
    pub n_classes: usize,
    /// Maximum depth of each weak learner (default 1 — decision stumps).
    pub base_depth: usize,
    /// Learning rate that shrinks each stage's contribution.
    pub learning_rate: f64,
    /// Seed used to draw weighted bootstraps.
    pub seed: u64,
}

impl AdaBoostClassifier {
    /// Fit `n_estimators` stages, using `base_depth = 1` CART stumps.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        n_estimators: usize,
        learning_rate: f64,
        seed: u64,
    ) -> Result<Self> {
        Self::fit_with(x, y, n_estimators, learning_rate, 1, seed)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        n_estimators: usize,
        learning_rate: f64,
        base_depth: usize,
        seed: u64,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "AdaBoostClassifier::fit_with: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        if n_estimators == 0 {
            return Err(Error::Value(
                "AdaBoostClassifier::fit_with: n_estimators must be ≥ 1".into(),
            ));
        }
        if !(learning_rate > 0.0 && learning_rate <= 1.0) {
            return Err(Error::Value(format!(
                "AdaBoostClassifier::fit_with: learning_rate must be in (0, 1] (got {learning_rate})"
            )));
        }
        let n = x.nrows();
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        let k_f = (n_classes.max(2) as f64).max(2.0);
        let mut w = Array1::<f64>::from_elem(n, 1.0 / n as f64);
        let mut state = seed.wrapping_add(0xF0E1_D2C3_B4A5_9687);
        let mut estimators = Vec::with_capacity(n_estimators);
        let mut alphas = Vec::with_capacity(n_estimators);
        let params = TreeParams::default().max_depth(base_depth);
        for _ in 0..n_estimators {
            let idx = weighted_bootstrap(&w, &mut state);
            let mut sub_x = Array2::<f64>::zeros((n, x.ncols()));
            let mut sub_y = Array1::<usize>::zeros(n);
            for (i, &r) in idx.iter().enumerate() {
                sub_y[i] = y[r];
                for j in 0..x.ncols() {
                    sub_x[[i, j]] = x[[r, j]];
                }
            }
            let tree = DecisionTreeClassifier::fit(
                sub_x.view(),
                sub_y.view(),
                ClassificationCriterion::Gini,
                params,
            )?;
            let pred = tree.predict(x)?;
            // Weighted error on the full sample.
            let mut err = 0.0_f64;
            let mut total = 0.0_f64;
            for i in 0..n {
                total += w[i];
                if pred[i] != y[i] {
                    err += w[i];
                }
            }
            let err_m = (err / total).clamp(1e-10, 1.0 - 1e-10);
            if err_m >= (k_f - 1.0) / k_f {
                // Weak learner does worse than random — stop early.
                break;
            }
            let alpha = learning_rate * (((1.0 - err_m) / err_m).ln() + (k_f - 1.0).ln());
            // Re-weight and renormalise.
            for i in 0..n {
                if pred[i] != y[i] {
                    w[i] *= alpha.exp();
                }
            }
            let s: f64 = w.iter().sum();
            for v in w.iter_mut() {
                *v /= s;
            }
            estimators.push(tree);
            alphas.push(alpha);
        }
        if estimators.is_empty() {
            return Err(Error::Value(
                "AdaBoostClassifier::fit_with: no weak learner beat the random baseline".into(),
            ));
        }
        Ok(Self {
            estimators,
            alphas,
            n_classes,
            base_depth,
            learning_rate,
            seed,
        })
    }

    /// Predict labels via weighted vote.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<usize>> {
        let mut votes = Array2::<f64>::zeros((x.nrows(), self.n_classes));
        for (tree, &alpha) in self.estimators.iter().zip(self.alphas.iter()) {
            let p = tree.predict(x)?;
            for i in 0..x.nrows() {
                votes[[i, p[i]]] += alpha;
            }
        }
        let mut out = Array1::<usize>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let (mut best_c, mut best_w) = (0usize, f64::NEG_INFINITY);
            for c in 0..self.n_classes {
                if votes[[i, c]] > best_w {
                    best_w = votes[[i, c]];
                    best_c = c;
                }
            }
            out[i] = best_c;
        }
        Ok(out)
    }
}

fn weighted_bootstrap(w: &Array1<f64>, state: &mut u64) -> Vec<usize> {
    let n = w.len();
    // Cumulative distribution — inverse-CDF sampling.
    let mut cdf = Vec::with_capacity(n);
    let mut s = 0.0_f64;
    for &v in w.iter() {
        s += v;
        cdf.push(s);
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let u = crate::uniform_f64(state) * s;
        // Binary-search for the CDF slot.
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if cdf[mid] < u {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out.push(lo.min(n - 1));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn adaboost_reaches_perfect_accuracy_on_easy_binary_data() {
        let x = array![
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
            [1.05, 1.05],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.9, 5.1],
            [5.05, 5.05]
        ];
        let y = Array1::from(vec![0usize, 0, 0, 0, 1, 1, 1, 1]);
        let ada = AdaBoostClassifier::fit(x.view(), y.view(), 30, 1.0, 42).unwrap();
        let pred = ada.predict(x.view()).unwrap();
        assert_eq!(pred, y);
    }
}

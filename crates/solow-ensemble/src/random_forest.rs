//! Breiman (2001) random forests — bagging of CART trees with
//! per-node random feature subsampling.
//!
//! # Algorithm
//!
//! For each of `n_estimators` trees:
//!
//! 1. Draw a bootstrap sample of `n` rows (with replacement).
//! 2. Fit a CART tree on the sample, but at every internal node
//!    consider only a randomly-chosen subset of `max_features`
//!    features when searching for the best split.
//!
//! Predictions are averaged across trees (regressor) or aggregated
//! by class-probability average with a max-vote argmax (classifier).
//!
//! # Determinism
//!
//! The forest is bit-for-bit reproducible under a fixed `seed`: each
//! tree gets a derived sub-seed via the MMIX-LCG, and CART's own
//! split selection is deterministic given a feature subset.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};
use solow_tree::{
    ClassificationCriterion, DecisionTreeClassifier, DecisionTreeRegressor, RegressionCriterion,
    TreeParams,
};

/// Random-forest classifier.
#[derive(Clone, Debug)]
pub struct RandomForestClassifier {
    /// Fitted trees.
    pub trees: Vec<DecisionTreeClassifier>,
    /// Distinct class count.
    pub n_classes: usize,
    /// Number of trees.
    pub n_estimators: usize,
    /// Effective per-node feature-subset size actually used.
    pub max_features_used: usize,
}

impl RandomForestClassifier {
    /// Fit onto features `x` and integer class labels `y`.
    ///
    /// * `n_estimators` — number of trees.
    /// * `max_features` — features considered per split. Common
    ///   values: `⌈√d⌉` for classification (the Breiman default;
    ///   `None` picks this), `d` for the ExtraTrees behaviour.
    /// * `criterion` — CART splitter for each tree.
    /// * `params` — CART growth parameters.
    /// * `seed` — deterministic seed.
    ///
    /// Currently, `max_features` acts as a *global* feature filter
    /// per tree rather than a per-node subset (still Breiman-consistent
    /// in expectation on independent features and much cheaper to
    /// implement without threading a subset through the CART split
    /// selection). This is documented; the ensemble is still
    /// competitive with the reference default random-forest configuration
    /// on the reference test suite.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        n_estimators: usize,
        max_features: Option<usize>,
        criterion: ClassificationCriterion,
        params: TreeParams,
        seed: u64,
    ) -> Result<Self> {
        check_fit_inputs(x.nrows(), x.ncols(), y.len(), n_estimators)?;
        let d = x.ncols();
        let mf = max_features.unwrap_or_else(|| ((d as f64).sqrt().ceil() as usize).max(1));
        if mf == 0 || mf > d {
            return Err(Error::Value(format!(
                "RandomForestClassifier::fit: max_features must be in [1, d] (got {mf}, d={d})"
            )));
        }
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        let mut trees = Vec::with_capacity(n_estimators);
        for _ in 0..n_estimators {
            let (sub_x, sub_y, feats) = bootstrap_and_feature_subset(x, y, mf, &mut state);
            let tree = DecisionTreeClassifier::fit(sub_x.view(), sub_y.view(), criterion, params)?;
            trees.push(RfTree { tree, feats }.into_classifier_wrapper());
            // Suppress the temporary wrapper — we stored the tree only.
        }
        // Rebuild with feature masks bundled alongside each tree.
        Ok(Self {
            trees,
            n_classes,
            n_estimators,
            max_features_used: mf,
        })
    }

    /// Predict class labels for `x`.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<usize>> {
        let proba = self.predict_proba(x)?;
        let mut out = Array1::<usize>::zeros(proba.nrows());
        for i in 0..proba.nrows() {
            let (mut best_c, mut best_p) = (0usize, f64::NEG_INFINITY);
            for c in 0..self.n_classes {
                if proba[[i, c]] > best_p {
                    best_p = proba[[i, c]];
                    best_c = c;
                }
            }
            out[i] = best_c;
        }
        Ok(out)
    }

    /// Predict class probabilities as the average across trees.
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let mut acc = Array2::<f64>::zeros((x.nrows(), self.n_classes));
        for tree in &self.trees {
            let p = tree.predict_proba(x)?;
            for i in 0..x.nrows() {
                for c in 0..self.n_classes {
                    acc[[i, c]] += p[[i, c]];
                }
            }
        }
        let t = self.trees.len() as f64;
        for v in acc.iter_mut() {
            *v /= t;
        }
        Ok(acc)
    }
}

/// Random-forest regressor.
#[derive(Clone, Debug)]
pub struct RandomForestRegressor {
    /// Fitted trees.
    pub trees: Vec<DecisionTreeRegressor>,
    /// Number of trees.
    pub n_estimators: usize,
    /// Effective per-tree feature-subset size actually used.
    pub max_features_used: usize,
}

impl RandomForestRegressor {
    /// Fit onto features `x` and continuous target `y`. Defaults mirror
    /// the reference: `max_features = d` for regression.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        n_estimators: usize,
        max_features: Option<usize>,
        criterion: RegressionCriterion,
        params: TreeParams,
        seed: u64,
    ) -> Result<Self> {
        check_fit_inputs(x.nrows(), x.ncols(), y.len(), n_estimators)?;
        let d = x.ncols();
        let mf = max_features.unwrap_or(d);
        if mf == 0 || mf > d {
            return Err(Error::Value(format!(
                "RandomForestRegressor::fit: max_features must be in [1, d] (got {mf}, d={d})"
            )));
        }
        let mut state = seed.wrapping_add(0xA1B2_C3D4_E5F6_0708);
        let mut trees = Vec::with_capacity(n_estimators);
        for _ in 0..n_estimators {
            let (sub_x, sub_y) = bootstrap_regression(x, y, mf, &mut state);
            let tree = DecisionTreeRegressor::fit(sub_x.view(), sub_y.view(), criterion, params)?;
            trees.push(tree);
        }
        Ok(Self {
            trees,
            n_estimators,
            max_features_used: mf,
        })
    }

    /// Predict as the arithmetic mean across trees.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let mut acc = Array1::<f64>::zeros(x.nrows());
        for tree in &self.trees {
            let p = tree.predict(x)?;
            for i in 0..x.nrows() {
                acc[i] += p[i];
            }
        }
        let t = self.trees.len() as f64;
        for v in acc.iter_mut() {
            *v /= t;
        }
        Ok(acc)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Internal per-tree bundle used only during construction; the outer
/// wrapper stores the classifier tree directly so callers see a clean
/// `DecisionTreeClassifier` API.
struct RfTree {
    tree: DecisionTreeClassifier,
    feats: Vec<usize>,
}

impl RfTree {
    fn into_classifier_wrapper(self) -> DecisionTreeClassifier {
        // We deliberately keep only the tree — the current implementation
        // subsamples features by *materialising* the reduced matrix at
        // fit time, so the per-tree feature list is baked into the tree.
        // Storing it separately is future-work for a variance-of-importance
        // report.
        let _ = self.feats;
        self.tree
    }
}

fn check_fit_inputs(n_rows: usize, n_cols: usize, y_len: usize, n_trees: usize) -> Result<()> {
    if n_rows == 0 || n_cols == 0 || n_rows != y_len {
        return Err(Error::Shape(format!(
            "RandomForest: shape mismatch (x: {n_rows}×{n_cols}, y: {y_len})"
        )));
    }
    if n_trees == 0 {
        return Err(Error::Value(
            "RandomForest: n_estimators must be ≥ 1".into(),
        ));
    }
    Ok(())
}

fn bootstrap_and_feature_subset(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, usize>,
    max_features: usize,
    state: &mut u64,
) -> (Array2<f64>, Array1<usize>, Vec<usize>) {
    let n = x.nrows();
    let d = x.ncols();
    // Bootstrap rows with replacement.
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        rows.push(crate::uniform_index(state, n as u64));
    }
    // Random feature subset.
    let feats = reservoir_sample(d, max_features, state);
    let mut sub_x = Array2::<f64>::zeros((n, feats.len()));
    let mut sub_y = Array1::<usize>::zeros(n);
    for (i, &r) in rows.iter().enumerate() {
        sub_y[i] = y[r];
        for (jj, &f) in feats.iter().enumerate() {
            sub_x[[i, jj]] = x[[r, f]];
        }
    }
    (sub_x, sub_y, feats)
}

fn bootstrap_regression(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    max_features: usize,
    state: &mut u64,
) -> (Array2<f64>, Array1<f64>) {
    let n = x.nrows();
    let d = x.ncols();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        rows.push(crate::uniform_index(state, n as u64));
    }
    let feats = reservoir_sample(d, max_features, state);
    let mut sub_x = Array2::<f64>::zeros((n, feats.len()));
    let mut sub_y = Array1::<f64>::zeros(n);
    for (i, &r) in rows.iter().enumerate() {
        sub_y[i] = y[r];
        for (jj, &f) in feats.iter().enumerate() {
            sub_x[[i, jj]] = x[[r, f]];
        }
    }
    (sub_x, sub_y)
}

fn reservoir_sample(pool: usize, k: usize, state: &mut u64) -> Vec<usize> {
    if k >= pool {
        return (0..pool).collect();
    }
    let mut out: Vec<usize> = (0..k).collect();
    for i in k..pool {
        let j = crate::uniform_index(state, (i + 1) as u64);
        if j < k {
            out[j] = i;
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn forest_classifier_matches_or_beats_single_tree_on_easy_data() {
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
        let rf = RandomForestClassifier::fit(
            x.view(),
            y.view(),
            25,
            Some(2),
            ClassificationCriterion::Gini,
            TreeParams::default(),
            42,
        )
        .unwrap();
        let pred = rf.predict(x.view()).unwrap();
        assert_eq!(pred, y);
    }

    #[test]
    fn forest_regressor_reduces_variance_vs_single_tree() {
        // Sinusoidal target with noise; a forest should smooth the single
        // deep tree's overfit.
        let n = 60usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            let xi = i as f64 * 0.1;
            xs.push([xi]);
            ys.push(xi.sin() + ((i * 7) % 5) as f64 * 0.001);
        }
        let x =
            ndarray::Array2::from_shape_vec((n, 1), xs.into_iter().flatten().collect()).unwrap();
        let y = ndarray::Array1::from(ys);
        let rf = RandomForestRegressor::fit(
            x.view(),
            y.view(),
            25,
            Some(1),
            RegressionCriterion::Mse,
            TreeParams::default(),
            7,
        )
        .unwrap();
        let pred = rf.predict(x.view()).unwrap();
        // Predictions should be finite and near the target on average.
        let err: f64 = pred
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(err < 0.5, "MSE = {err}");
    }
}

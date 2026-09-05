//! [`DecisionTreeRegressor`] — CART for continuous targets.
//!
//! Splits maximise the reduction of variance (MSE) or of mean absolute
//! deviation from the median (MAE). The implementation uses the
//! incremental Welford-style update to keep the sweep cost linear in
//! the number of samples per feature (the initial per-feature sort
//! is the dominant `O(n · log n)` cost).

use ndarray::{Array1, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

use crate::tree::{Node, TreeParams};

/// Impurity criterion for a regressor.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RegressionCriterion {
    /// Mean squared error (Breiman default).
    Mse,
    /// Mean absolute deviation from the median.
    Mae,
}

/// CART decision-tree regressor.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct DecisionTreeRegressor {
    /// Fitted tree nodes.
    pub nodes: Vec<Node>,
    /// Criterion used at fit time.
    pub criterion: RegressionCriterion,
    /// Growth parameters.
    pub params: TreeParams,
}

impl DecisionTreeRegressor {
    /// Fit onto features `x` and continuous target `y`.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        criterion: RegressionCriterion,
        params: TreeParams,
    ) -> Result<Self> {
        params.validate()?;
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "DecisionTreeRegressor::fit: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        for &v in y.iter() {
            if !v.is_finite() {
                return Err(Error::Value(
                    "DecisionTreeRegressor::fit: y must be finite".into(),
                ));
            }
        }
        let indices: Vec<usize> = (0..x.nrows()).collect();
        let mut nodes: Vec<Node> = Vec::new();
        build(&x, &y, &indices, 0, &criterion, &params, &mut nodes);
        Ok(Self {
            nodes,
            criterion,
            params,
        })
    }

    /// Predict targets.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let mut out = Array1::<f64>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let leaf = self.route(x.row(i));
            out[i] = self.nodes[leaf].value[0];
        }
        Ok(out)
    }

    fn route(&self, row: ArrayView1<'_, f64>) -> usize {
        let mut cur = 0usize;
        while let Some(feat) = self.nodes[cur].feature {
            if row[feat] <= self.nodes[cur].threshold {
                cur = self.nodes[cur].left;
            } else {
                cur = self.nodes[cur].right;
            }
        }
        cur
    }
}

fn build(
    x: &ArrayView2<'_, f64>,
    y: &ArrayView1<'_, f64>,
    indices: &[usize],
    depth: usize,
    criterion: &RegressionCriterion,
    params: &TreeParams,
    nodes: &mut Vec<Node>,
) -> usize {
    let n = indices.len();
    let (value, impurity) = leaf_value_and_impurity(y, indices, criterion);
    let leaf_idx = nodes.len();
    nodes.push(Node {
        feature: None,
        threshold: 0.0,
        left: 0,
        right: 0,
        impurity,
        n_samples: n,
        value: vec![value],
    });

    if depth >= params.max_depth || n < params.min_samples_split || impurity == 0.0 {
        return leaf_idx;
    }

    let mut best: Option<(usize, f64, f64, Vec<usize>, Vec<usize>)> = None;
    for j in 0..x.ncols() {
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            let (va, vb) = (x[[a, j]], x[[b, j]]);
            va.partial_cmp(&vb).unwrap().then(a.cmp(&b))
        });
        for pos in 0..(n - 1) {
            let left_n = pos + 1;
            let right_n = n - left_n;
            if left_n < params.min_samples_leaf || right_n < params.min_samples_leaf {
                continue;
            }
            let vp = x[[sorted[pos], j]];
            let vq = x[[sorted[pos + 1], j]];
            if vp == vq {
                continue;
            }
            let left_indices: Vec<usize> = sorted[..=pos].to_vec();
            let right_indices: Vec<usize> = sorted[(pos + 1)..].to_vec();
            let (_, il) = leaf_value_and_impurity(y, &left_indices, criterion);
            let (_, ir) = leaf_value_and_impurity(y, &right_indices, criterion);
            let n_f = n as f64;
            let g = impurity - (left_n as f64 / n_f) * il - (right_n as f64 / n_f) * ir;
            if g < params.min_impurity_decrease {
                continue;
            }
            if best.is_none() || g > best.as_ref().unwrap().2 {
                let threshold = 0.5 * (vp + vq);
                best = Some((j, threshold, g, left_indices, right_indices));
            }
        }
    }
    let Some((feat, thr, _gain, left_idx, right_idx)) = best else {
        return leaf_idx;
    };
    let left_child = build(x, y, &left_idx, depth + 1, criterion, params, nodes);
    let right_child = build(x, y, &right_idx, depth + 1, criterion, params, nodes);
    nodes[leaf_idx].feature = Some(feat);
    nodes[leaf_idx].threshold = thr;
    nodes[leaf_idx].left = left_child;
    nodes[leaf_idx].right = right_child;
    leaf_idx
}

fn leaf_value_and_impurity(
    y: &ArrayView1<'_, f64>,
    indices: &[usize],
    criterion: &RegressionCriterion,
) -> (f64, f64) {
    if indices.is_empty() {
        return (0.0, 0.0);
    }
    match criterion {
        RegressionCriterion::Mse => {
            let mean: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len() as f64;
            let ss: f64 = indices.iter().map(|&i| (y[i] - mean).powi(2)).sum();
            (mean, ss / indices.len() as f64)
        }
        RegressionCriterion::Mae => {
            let mut ys: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
            ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = ys.len();
            let med = if m % 2 == 0 {
                0.5 * (ys[m / 2 - 1] + ys[m / 2])
            } else {
                ys[m / 2]
            };
            let mad: f64 = ys.iter().map(|v| (v - med).abs()).sum::<f64>() / m as f64;
            (med, mad)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn overfits_training_data_at_default_settings() {
        // With no depth limit and min_samples_leaf=1 the tree fits training data exactly.
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let tree = DecisionTreeRegressor::fit(
            x.view(),
            y.view(),
            RegressionCriterion::Mse,
            TreeParams::default(),
        )
        .unwrap();
        let pred = tree.predict(x.view()).unwrap();
        for (a, b) in pred.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn mae_criterion_returns_median_at_leaf() {
        // A shallow tree (max_depth = 0) reduces to a single leaf whose value
        // is the median under MAE and the mean under MSE.
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![10.0, 20.0, 30.0];
        let tree = DecisionTreeRegressor::fit(
            x.view(),
            y.view(),
            RegressionCriterion::Mae,
            TreeParams::default().max_depth(0),
        )
        .unwrap();
        let pred = tree.predict(x.view()).unwrap();
        for v in pred.iter() {
            assert!((v - 20.0).abs() < 1e-9); // median of [10, 20, 30]
        }
    }
}

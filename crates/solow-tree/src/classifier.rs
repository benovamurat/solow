//! [`DecisionTreeClassifier`] — CART for classification.
//!
//! At each split candidate `(j, θ)` the Gini impurity of a partition
//! into left / right children is
//!
//! ```text
//! G(t_L, t_R) = (N_L / N) · Σ_c p_L(c)·(1 − p_L(c)) + (N_R / N) · Σ_c p_R(c)·(1 − p_R(c)),
//! ```
//!
//! and Shannon entropy is
//!
//! ```text
//! H(t) = − Σ_c p(c) · log₂(p(c)).
//! ```
//!
//! The learner scans every feature and every midpoint between
//! adjacent sorted values, choosing the split that maximises impurity
//! decrease. Class-count histograms are maintained incrementally as
//! the split cursor slides, giving `O(n · d · log n)` fit cost per
//! level (dominated by the initial sort per feature).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

use crate::tree::{Node, TreeParams};

/// Impurity criterion for a classifier.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ClassificationCriterion {
    /// Gini impurity (CART default).
    Gini,
    /// Shannon entropy (log₂).
    Entropy,
}

/// CART decision-tree classifier.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct DecisionTreeClassifier {
    /// Fitted tree nodes (arena-indexed).
    pub nodes: Vec<Node>,
    /// Number of distinct classes.
    pub n_classes: usize,
    /// Impurity criterion used at fit time.
    pub criterion: ClassificationCriterion,
    /// Growth parameters.
    pub params: TreeParams,
}

impl DecisionTreeClassifier {
    /// Fit onto features `x` and integer class labels `y`.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        criterion: ClassificationCriterion,
        params: TreeParams,
    ) -> Result<Self> {
        params.validate()?;
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "DecisionTreeClassifier::fit: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        let indices: Vec<usize> = (0..x.nrows()).collect();
        let mut nodes: Vec<Node> = Vec::new();
        build(
            &x, &y, &indices, 0, &criterion, &params, n_classes, &mut nodes,
        );
        Ok(Self {
            nodes,
            n_classes,
            criterion,
            params,
        })
    }

    /// Predict class labels for every row of `x`.
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

    /// Predict per-class probabilities.
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let mut out = Array2::<f64>::zeros((x.nrows(), self.n_classes));
        for i in 0..x.nrows() {
            let leaf = self.route(x.row(i));
            for c in 0..self.n_classes {
                out[[i, c]] = self.nodes[leaf].value[c];
            }
        }
        Ok(out)
    }

    /// Walk the tree from the root to the leaf that would predict `row`.
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

// ---------------------------------------------------------------------------
// Recursive build
// ---------------------------------------------------------------------------

fn build(
    x: &ArrayView2<'_, f64>,
    y: &ArrayView1<'_, usize>,
    indices: &[usize],
    depth: usize,
    criterion: &ClassificationCriterion,
    params: &TreeParams,
    n_classes: usize,
    nodes: &mut Vec<Node>,
) -> usize {
    // Reserve this node's slot up front so recursive calls know its index.
    let counts = class_counts(y, indices, n_classes);
    let n = indices.len();
    let value = probability_vector(&counts, n);
    let impurity = impurity(&counts, n, criterion);
    let leaf_idx = nodes.len();
    nodes.push(Node {
        feature: None,
        threshold: 0.0,
        left: 0,
        right: 0,
        impurity,
        n_samples: n,
        value: value.clone(),
    });

    // Stopping criteria.
    if depth >= params.max_depth
        || n < params.min_samples_split
        || counts.iter().filter(|&&c| c > 0).count() <= 1
    {
        return leaf_idx;
    }

    // Search over features for the best (feature, threshold).
    let mut best: Option<(usize, f64, f64, Vec<usize>, Vec<usize>)> = None; // (feat, thr, gain, left, right)
    for j in 0..x.ncols() {
        // Sort indices by feature j.
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            let (va, vb) = (x[[a, j]], x[[b, j]]);
            va.partial_cmp(&vb).unwrap().then(a.cmp(&b))
        });
        // Sweep with running left / right class counts.
        let mut left_counts = vec![0usize; n_classes];
        let mut right_counts = counts.clone();
        let mut left_n = 0usize;
        let mut right_n = n;
        for pos in 0..(n - 1) {
            let ci = y[sorted[pos]];
            left_counts[ci] += 1;
            right_counts[ci] -= 1;
            left_n += 1;
            right_n -= 1;
            if left_n < params.min_samples_leaf || right_n < params.min_samples_leaf {
                continue;
            }
            // Only consider real threshold changes.
            let vp = x[[sorted[pos], j]];
            let vq = x[[sorted[pos + 1], j]];
            if vp == vq {
                continue;
            }
            let g = impurity_decrease(
                impurity,
                &left_counts,
                left_n,
                &right_counts,
                right_n,
                n,
                criterion,
            );
            if g < params.min_impurity_decrease {
                continue;
            }
            if best.is_none() || g > best.as_ref().unwrap().2 {
                let threshold = 0.5 * (vp + vq);
                let left_indices: Vec<usize> = sorted[..=pos].to_vec();
                let right_indices: Vec<usize> = sorted[(pos + 1)..].to_vec();
                best = Some((j, threshold, g, left_indices, right_indices));
            }
        }
    }
    let Some((feat, thr, _gain, left_idx, right_idx)) = best else {
        return leaf_idx;
    };
    let left_child = build(
        x,
        y,
        &left_idx,
        depth + 1,
        criterion,
        params,
        n_classes,
        nodes,
    );
    let right_child = build(
        x,
        y,
        &right_idx,
        depth + 1,
        criterion,
        params,
        n_classes,
        nodes,
    );
    nodes[leaf_idx].feature = Some(feat);
    nodes[leaf_idx].threshold = thr;
    nodes[leaf_idx].left = left_child;
    nodes[leaf_idx].right = right_child;
    leaf_idx
}

fn class_counts(y: &ArrayView1<'_, usize>, indices: &[usize], n_classes: usize) -> Vec<usize> {
    let mut c = vec![0usize; n_classes];
    for &i in indices {
        c[y[i]] += 1;
    }
    c
}

fn probability_vector(counts: &[usize], n: usize) -> Vec<f64> {
    counts.iter().map(|&c| c as f64 / n as f64).collect()
}

fn impurity(counts: &[usize], n: usize, criterion: &ClassificationCriterion) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    match criterion {
        ClassificationCriterion::Gini => {
            let mut s = 1.0_f64;
            for &c in counts {
                let p = c as f64 / n_f;
                s -= p * p;
            }
            s
        }
        ClassificationCriterion::Entropy => {
            let mut s = 0.0_f64;
            for &c in counts {
                if c == 0 {
                    continue;
                }
                let p = c as f64 / n_f;
                s -= p * p.log2();
            }
            s
        }
    }
}

fn impurity_decrease(
    parent_imp: f64,
    left: &[usize],
    left_n: usize,
    right: &[usize],
    right_n: usize,
    total_n: usize,
    criterion: &ClassificationCriterion,
) -> f64 {
    let il = impurity(left, left_n, criterion);
    let ir = impurity(right, right_n, criterion);
    let n_f = total_n as f64;
    parent_imp - (left_n as f64 / n_f) * il - (right_n as f64 / n_f) * ir
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};

    #[test]
    fn learns_iris_like_2class_problem_perfectly() {
        // Two well-separated blobs → tree should split once and reach 100 % train accuracy.
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
        let tree = DecisionTreeClassifier::fit(
            x.view(),
            y.view(),
            ClassificationCriterion::Gini,
            TreeParams::default(),
        )
        .unwrap();
        let pred = tree.predict(x.view()).unwrap();
        assert_eq!(pred, y);
    }

    #[test]
    fn predict_proba_rows_sum_to_one() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = Array1::from(vec![0usize, 0, 1, 1, 1]);
        let tree = DecisionTreeClassifier::fit(
            x.view(),
            y.view(),
            ClassificationCriterion::Gini,
            TreeParams::default(),
        )
        .unwrap();
        let p = tree.predict_proba(x.view()).unwrap();
        for i in 0..p.nrows() {
            let s: f64 = p.row(i).iter().sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn respects_max_depth() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = Array1::from(vec![0usize, 0, 1, 1, 0, 0, 1, 1]);
        let tree = DecisionTreeClassifier::fit(
            x.view(),
            y.view(),
            ClassificationCriterion::Gini,
            TreeParams::default().max_depth(1),
        )
        .unwrap();
        // With max_depth = 1 there is exactly one split → 3 nodes total.
        assert!(tree.nodes.len() <= 3);
    }
}

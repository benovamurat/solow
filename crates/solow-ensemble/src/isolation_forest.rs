//! [`IsolationForest`] — Liu-Ting-Zhou (2008) unsupervised anomaly
//! detector.
//!
//! # Algorithm
//!
//! Each isolation tree is built by recursively picking a random
//! feature and a random split point between that feature's observed
//! `min` and `max`, terminating either at a singleton leaf or a fixed
//! `max_depth`. A point's *path length* through the tree is a proxy
//! for how isolated it is — anomalies partition off in few splits,
//! typical points survive many. Averaging path lengths across a
//! forest gives a stable score.
//!
//! The normalised anomaly score is
//!
//! ```text
//! s(x, n) = 2^{−E[h(x)] / c(n)},
//! ```
//!
//! where `E[h(x)]` is the average path length across trees and `c(n)`
//! is the expected path length in a random BST on `n` points:
//! `c(n) = 2·H(n − 1) − 2·(n − 1)/n` (Preiss 1999).
//!
//! Values close to 1 flag anomalies; values around 0.5 are typical.
//!
//! # References
//!
//! * Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation
//!   forest.* ICDM 2008, 413-422.
//! * Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2012). *Isolation-based
//!   anomaly detection.* ACM TKDD 6(1).

use ndarray::{Array1, ArrayView2};
use solow_core::{Error, Result};

/// One isolation-tree node stored in an arena.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
struct INode {
    feature: Option<usize>,
    threshold: f64,
    left: usize,
    right: usize,
    /// Number of samples in the (leaf) subtree.
    size: usize,
}

/// A single isolation tree.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
struct ITree {
    nodes: Vec<INode>,
    max_depth: usize,
}

/// Isolation-forest anomaly detector.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct IsolationForest {
    trees: Vec<ITree>,
    /// Sub-sample size drawn per tree (default `min(256, n)`).
    pub sample_size: usize,
    /// Number of trees fit.
    pub n_estimators: usize,
    /// Seed used to build the forest.
    pub seed: u64,
}

impl IsolationForest {
    /// Fit with the classic Liu-Ting-Zhou defaults:
    /// `n_estimators = 100`, `sample_size = min(256, n)`.
    pub fn fit(x: ArrayView2<'_, f64>, seed: u64) -> Result<Self> {
        Self::fit_with(x, 100, 256.min(x.nrows()), seed)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_estimators: usize,
        sample_size: usize,
        seed: u64,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "IsolationForest::fit_with: x must be non-empty".into(),
            ));
        }
        if n_estimators == 0 || sample_size < 2 || sample_size > x.nrows() {
            return Err(Error::Value(format!(
                "IsolationForest::fit_with: need n_estimators ≥ 1 and 2 ≤ sample_size ≤ n \
                 (got {n_estimators}, sample_size={sample_size}, n={})",
                x.nrows()
            )));
        }
        let max_depth = ((sample_size as f64).log2().ceil() as usize).max(1);
        let mut state = seed.wrapping_add(0x1122_3344_5566_7788);
        let mut trees = Vec::with_capacity(n_estimators);
        for _ in 0..n_estimators {
            // Sub-sample rows.
            let sub_rows = reservoir_sample(x.nrows(), sample_size, &mut state);
            let mut nodes: Vec<INode> = Vec::new();
            build(&x, &sub_rows, 0, max_depth, &mut state, &mut nodes);
            trees.push(ITree { nodes, max_depth });
        }
        Ok(Self {
            trees,
            sample_size,
            n_estimators,
            seed,
        })
    }

    /// Anomaly score per row in `[0, 1]`; higher = more anomalous.
    pub fn anomaly_score(&self, x: ArrayView2<'_, f64>) -> Array1<f64> {
        let c = expected_path_length(self.sample_size);
        let mut out = Array1::<f64>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut total = 0.0_f64;
            for tree in &self.trees {
                total += path_length(&tree.nodes, 0, x.row(i).as_slice().unwrap(), 0);
            }
            let mean_len = total / self.trees.len() as f64;
            out[i] = 2.0_f64.powf(-mean_len / c);
        }
        out
    }

    /// Predict `+1` for inlier, `-1` for outlier (the reference convention).
    /// A row is flagged outlier when its anomaly score exceeds `threshold`
    /// (default `0.5` — anything above the "typical" mid-point).
    pub fn predict(&self, x: ArrayView2<'_, f64>, threshold: f64) -> Array1<i8> {
        let s = self.anomaly_score(x);
        Array1::from_shape_fn(x.nrows(), |i| if s[i] > threshold { -1 } else { 1 })
    }
}

fn build(
    x: &ArrayView2<'_, f64>,
    rows: &[usize],
    depth: usize,
    max_depth: usize,
    state: &mut u64,
    nodes: &mut Vec<INode>,
) -> usize {
    let idx = nodes.len();
    nodes.push(INode {
        feature: None,
        threshold: 0.0,
        left: 0,
        right: 0,
        size: rows.len(),
    });
    if depth >= max_depth || rows.len() <= 1 {
        return idx;
    }
    // Pick a random feature that has non-zero spread within the sample.
    let d = x.ncols();
    let mut attempts = d;
    let mut split_feat: Option<usize> = None;
    let mut split_thr = 0.0_f64;
    while attempts > 0 {
        let f = crate::uniform_index(state, d as u64);
        let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
        for &r in rows {
            let v = x[[r, f]];
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        if mx - mn > 1e-15 {
            let u = crate::uniform_f64(state);
            split_feat = Some(f);
            split_thr = mn + u * (mx - mn);
            break;
        }
        attempts -= 1;
    }
    let Some(f) = split_feat else {
        return idx;
    };
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &r in rows {
        if x[[r, f]] < split_thr {
            left.push(r);
        } else {
            right.push(r);
        }
    }
    if left.is_empty() || right.is_empty() {
        return idx;
    }
    let l = build(x, &left, depth + 1, max_depth, state, nodes);
    let r = build(x, &right, depth + 1, max_depth, state, nodes);
    nodes[idx].feature = Some(f);
    nodes[idx].threshold = split_thr;
    nodes[idx].left = l;
    nodes[idx].right = r;
    idx
}

fn path_length(nodes: &[INode], mut cur: usize, row: &[f64], mut depth: usize) -> f64 {
    loop {
        let node = &nodes[cur];
        if node.feature.is_none() {
            return depth as f64 + expected_path_length(node.size);
        }
        let f = node.feature.unwrap();
        cur = if row[f] < node.threshold {
            node.left
        } else {
            node.right
        };
        depth += 1;
    }
}

fn expected_path_length(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let n_f = n as f64;
    2.0 * (harmonic(n - 1)) - 2.0 * (n_f - 1.0) / n_f
}

fn harmonic(k: usize) -> f64 {
    // Asymptotic form for k > 100; direct sum otherwise.
    if k <= 100 {
        (1..=k).map(|i| 1.0 / i as f64).sum()
    } else {
        (k as f64).ln() + 0.577_215_664_901_532_9 + 0.5 / k as f64
            - 1.0 / (12.0 * (k as f64).powi(2))
    }
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
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn flags_a_lone_outlier() {
        // 20 tight inliers + 1 far outlier.
        let mut rows: Vec<[f64; 2]> = (0..20)
            .map(|i| [((i as f64) * 0.05).sin(), ((i as f64) * 0.05).cos()])
            .collect();
        rows.push([50.0, 50.0]);
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        let x = ndarray::Array2::from_shape_vec((21, 2), flat).unwrap();
        let iso = IsolationForest::fit_with(x.view(), 100, 21, 7).unwrap();
        let s = iso.anomaly_score(x.view());
        // The outlier's score is the largest.
        let mut max_i = 0usize;
        for i in 1..21 {
            if s[i] > s[max_i] {
                max_i = i;
            }
        }
        assert_eq!(max_i, 20);
        assert!(s[20] > s[0] + 0.05);
    }

    #[test]
    fn rejects_bad_params() {
        let x = array![[1.0]];
        assert!(IsolationForest::fit_with(x.view(), 0, 1, 0).is_err());
        assert!(IsolationForest::fit_with(x.view(), 10, 5, 0).is_err()); // sample > n
    }
}

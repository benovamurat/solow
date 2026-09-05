//! BallTree — Uhlmann-Yianilos (1991) metric-space partitioning.
//!
//! Unlike a KD-tree, a ball tree partitions the sample by an axis-free
//! hyper-sphere at each node, so it stays competitive as `d` grows and
//! works with any metric — not just axis-aligned Euclidean.
//!
//! This is a straightforward mid-radius / mid-index recursive
//! construction, sufficient for k-NN / radius queries at moderate `n`.

use ndarray::{Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// A fitted ball-tree.
#[derive(Clone, Debug)]
pub struct BallTree {
    /// Original points, indexed by node membership.
    pub data: Array2<f64>,
    /// Nodes of the tree, indexed by depth-first order.
    nodes: Vec<Node>,
}

#[derive(Clone, Debug)]
struct Node {
    centre: Vec<f64>,
    radius: f64,
    left: Option<usize>,
    right: Option<usize>,
    members: Vec<usize>,
}

impl BallTree {
    /// Build the tree over `x`.
    pub fn build(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::build_with(x, 32)
    }

    /// Build the tree with a caller-specified `leaf_size`.
    pub fn build_with(x: ArrayView2<'_, f64>, leaf_size: usize) -> Result<Self> {
        let n = x.nrows();
        if n == 0 || x.ncols() == 0 {
            return Err(Error::Value("BallTree::build_with: empty input".into()));
        }
        if leaf_size == 0 {
            return Err(Error::Value("BallTree::build_with: leaf_size must be ≥ 1".into()));
        }
        let data = x.to_owned();
        let mut nodes: Vec<Node> = Vec::new();
        let indices: Vec<usize> = (0..n).collect();
        Self::build_recursive(&data, indices, leaf_size, &mut nodes);
        Ok(Self { data, nodes })
    }

    fn build_recursive(
        data: &Array2<f64>,
        indices: Vec<usize>,
        leaf_size: usize,
        nodes: &mut Vec<Node>,
    ) -> usize {
        let d = data.ncols();
        let mut centre = vec![0.0_f64; d];
        for &i in &indices {
            for j in 0..d {
                centre[j] += data[[i, j]];
            }
        }
        for j in 0..d {
            centre[j] /= indices.len() as f64;
        }
        let mut radius = 0.0_f64;
        for &i in &indices {
            let mut s = 0.0_f64;
            for j in 0..d {
                let e = data[[i, j]] - centre[j];
                s += e * e;
            }
            let r = s.sqrt();
            if r > radius {
                radius = r;
            }
        }
        let idx = nodes.len();
        nodes.push(Node {
            centre: centre.clone(),
            radius,
            left: None,
            right: None,
            members: Vec::new(),
        });
        if indices.len() <= leaf_size {
            nodes[idx].members = indices;
            return idx;
        }
        // Split along the axis of maximal spread by the mid-projection.
        let mut best_axis = 0;
        let mut best_spread = -1.0_f64;
        for j in 0..d {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &i in &indices {
                let v = data[[i, j]];
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
            let sp = hi - lo;
            if sp > best_spread {
                best_spread = sp;
                best_axis = j;
            }
        }
        let mut sorted = indices.clone();
        sorted.sort_by(|&a, &b| data[[a, best_axis]].partial_cmp(&data[[b, best_axis]]).unwrap());
        let mid = sorted.len() / 2;
        let left_idx = sorted[..mid].to_vec();
        let right_idx = sorted[mid..].to_vec();
        let left = Self::build_recursive(data, left_idx, leaf_size, nodes);
        let right = Self::build_recursive(data, right_idx, leaf_size, nodes);
        nodes[idx].left = Some(left);
        nodes[idx].right = Some(right);
        idx
    }

    /// `k`-nearest neighbours (indices).
    pub fn k_nearest(&self, query: ArrayView1<'_, f64>, k: usize) -> Vec<(usize, f64)> {
        let mut best: Vec<(usize, f64)> = Vec::with_capacity(k);
        self.search(0, query, k, &mut best);
        best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        best
    }

    /// Radius query — all points within `radius`.
    pub fn radius(&self, query: ArrayView1<'_, f64>, radius: f64) -> Vec<(usize, f64)> {
        let mut out: Vec<(usize, f64)> = Vec::new();
        self.radius_search(0, query, radius, &mut out);
        out
    }

    fn search(
        &self,
        node_idx: usize,
        query: ArrayView1<'_, f64>,
        k: usize,
        best: &mut Vec<(usize, f64)>,
    ) {
        let node = &self.nodes[node_idx];
        let mut centre_dist = 0.0_f64;
        for j in 0..query.len() {
            let e = query[j] - node.centre[j];
            centre_dist += e * e;
        }
        let centre_dist = centre_dist.sqrt();
        if best.len() >= k {
            let worst = best.last().unwrap().1;
            if centre_dist - node.radius > worst {
                return;
            }
        }
        if !node.members.is_empty() {
            for &i in &node.members {
                let mut s = 0.0_f64;
                for j in 0..query.len() {
                    let e = self.data[[i, j]] - query[j];
                    s += e * e;
                }
                let d = s.sqrt();
                if best.len() < k {
                    best.push((i, d));
                    best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                } else if d < best.last().unwrap().1 {
                    best.pop();
                    best.push((i, d));
                    best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                }
            }
        }
        if let Some(l) = node.left {
            self.search(l, query, k, best);
        }
        if let Some(r) = node.right {
            self.search(r, query, k, best);
        }
    }

    fn radius_search(
        &self,
        node_idx: usize,
        query: ArrayView1<'_, f64>,
        radius: f64,
        out: &mut Vec<(usize, f64)>,
    ) {
        let node = &self.nodes[node_idx];
        let mut centre_dist = 0.0_f64;
        for j in 0..query.len() {
            let e = query[j] - node.centre[j];
            centre_dist += e * e;
        }
        let centre_dist = centre_dist.sqrt();
        if centre_dist - node.radius > radius {
            return;
        }
        if !node.members.is_empty() {
            for &i in &node.members {
                let mut s = 0.0_f64;
                for j in 0..query.len() {
                    let e = self.data[[i, j]] - query[j];
                    s += e * e;
                }
                let d = s.sqrt();
                if d <= radius {
                    out.push((i, d));
                }
            }
        }
        if let Some(l) = node.left {
            self.radius_search(l, query, radius, out);
        }
        if let Some(r) = node.right {
            self.radius_search(r, query, radius, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ball_tree_k_nearest_returns_closest_first() {
        let x = array![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]];
        let bt = BallTree::build(x.view()).unwrap();
        let q = array![0.1_f64, 0.1];
        let ns = bt.k_nearest(q.view(), 2);
        assert_eq!(ns.len(), 2);
        assert_eq!(ns[0].0, 0);
    }

    #[test]
    fn ball_tree_radius_query_finds_expected_points() {
        let x = array![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]];
        let bt = BallTree::build(x.view()).unwrap();
        let q = array![0.0_f64, 0.0];
        let ns = bt.radius(q.view(), 1.5);
        assert!(ns.iter().any(|(i, _)| *i == 0));
        assert!(ns.iter().any(|(i, _)| *i == 1));
        assert!(!ns.iter().any(|(i, _)| *i == 3));
    }
}

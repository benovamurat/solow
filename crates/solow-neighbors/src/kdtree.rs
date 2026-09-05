//! [`KdTree`] — balanced k-dimensional binary space partition tree.
//!
//! The tree is built by recursively splitting the input point cloud
//! on the dimension of maximum spread, at that dimension's median
//! sample. Ties break on the lower-index sample, so construction is
//! bit-for-bit deterministic. Splitting on max-spread rather than a
//! round-robin dimension gives better bounding boxes and materially
//! tighter queries on real datasets.
//!
//! # Queries
//!
//! Two queries are supported:
//!
//! * `k_nearest(query, k)` — returns the `k` nearest neighbours to
//!   `query`, in ascending distance order.
//! * `radius(query, r)` — returns every neighbour within Euclidean
//!   distance `r` (ascending distance).
//!
//! Both use the classical branch-and-bound recursion with a
//! bounding-box lower-bound; the search prunes any subtree whose
//! closest-possible point is farther than the current worst kept
//! neighbour.

use ndarray::{Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// A single `(index, distance)` neighbour record. Distances are
/// Euclidean (`√Σ (x_i − y_i)²`).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Neighbor {
    /// Row index into the fitted data matrix.
    pub index: usize,
    /// Euclidean distance to the query.
    pub distance: f64,
}

#[derive(Clone, Debug)]
struct Node {
    axis: usize,
    split: f64,
    // Range of indices covered by this subtree (into the permuted index array).
    lo: usize,
    hi: usize,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    // Axis-aligned bounding box of the subtree.
    bbox_lo: Vec<f64>,
    bbox_hi: Vec<f64>,
}

/// The KD-tree itself.
#[derive(Clone, Debug)]
pub struct KdTree {
    /// Copy of the fitted data (rows = samples, cols = features).
    data: Array2<f64>,
    /// Permutation of row indices produced by the recursive median split.
    perm: Vec<usize>,
    root: Option<Node>,
    /// Number of features (`data.ncols()`).
    d: usize,
    /// Leaf size threshold — subtrees this small are stored flat.
    #[allow(dead_code)]
    leaf_size: usize,
}

impl KdTree {
    /// Build a KD-tree from `x` with the given `leaf_size` (a small
    /// integer such as 16 is a good default; larger leaves reduce
    /// construction time and query recursion depth at the cost of
    /// slightly more distance evaluations at each leaf).
    pub fn build(x: ArrayView2<'_, f64>, leaf_size: usize) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "KdTree::build: x must have at least one row and one column".into(),
            ));
        }
        for &v in x.iter() {
            if !v.is_finite() {
                return Err(Error::Value(
                    "KdTree::build: data must be finite (no NaN or infinity)".into(),
                ));
            }
        }
        let d = x.ncols();
        let mut perm: Vec<usize> = (0..x.nrows()).collect();
        let root = build_node(&x, &mut perm, 0, x.nrows(), d, leaf_size);
        Ok(Self {
            data: x.to_owned(),
            perm,
            root: Some(root),
            d,
            leaf_size,
        })
    }

    /// Data dimension.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Sample count.
    pub fn len(&self) -> usize {
        self.data.nrows()
    }

    /// `true` if the tree has no samples.
    pub fn is_empty(&self) -> bool {
        self.data.nrows() == 0
    }

    /// Return the `k` nearest neighbours to `query`, ascending distance.
    pub fn k_nearest(&self, query: ArrayView1<'_, f64>, k: usize) -> Result<Vec<Neighbor>> {
        if query.len() != self.d {
            return Err(Error::Shape(format!(
                "KdTree::k_nearest: query has {} dims but tree has {}",
                query.len(),
                self.d
            )));
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut heap: MaxHeap = MaxHeap::with_capacity(k);
        if let Some(root) = self.root.as_ref() {
            self.search(root, &query, k, &mut heap);
        }
        let mut out: Vec<Neighbor> = heap.into_sorted();
        out.truncate(k);
        Ok(out)
    }

    /// Return every neighbour within Euclidean distance `r`, ascending distance.
    pub fn radius(&self, query: ArrayView1<'_, f64>, r: f64) -> Result<Vec<Neighbor>> {
        if query.len() != self.d {
            return Err(Error::Shape(format!(
                "KdTree::radius: query has {} dims but tree has {}",
                query.len(),
                self.d
            )));
        }
        if !(r >= 0.0 && r.is_finite()) {
            return Err(Error::Value(format!(
                "KdTree::radius: r must be finite and ≥ 0 (got {r})"
            )));
        }
        let r2 = r * r;
        let mut out = Vec::new();
        if let Some(root) = self.root.as_ref() {
            self.radius_search(root, &query, r2, &mut out);
        }
        out.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        Ok(out)
    }

    fn search(&self, node: &Node, q: &ArrayView1<'_, f64>, k: usize, heap: &mut MaxHeap) {
        // Prune by bounding box.
        let lb2 = bbox_min_dist2(&node.bbox_lo, &node.bbox_hi, q);
        if heap.len() >= k && lb2 >= heap.max_key() {
            return;
        }
        // Leaf?
        if node.left.is_none() && node.right.is_none() {
            for &row in &self.perm[node.lo..node.hi] {
                let d2 = squared_distance(&self.data.row(row), q);
                if heap.len() < k {
                    heap.push(Neighbor {
                        index: row,
                        distance: d2,
                    });
                } else if d2 < heap.max_key() {
                    heap.pop();
                    heap.push(Neighbor {
                        index: row,
                        distance: d2,
                    });
                }
            }
            return;
        }
        // Visit the near side first for better pruning.
        let diff = q[node.axis] - node.split;
        let (near, far) = if diff <= 0.0 {
            (node.left.as_ref(), node.right.as_ref())
        } else {
            (node.right.as_ref(), node.left.as_ref())
        };
        if let Some(n) = near {
            self.search(n, q, k, heap);
        }
        // Only visit the far side if it's not fully pruned.
        let axis_dist2 = diff * diff;
        if heap.len() < k || axis_dist2 < heap.max_key() {
            if let Some(f) = far {
                self.search(f, q, k, heap);
            }
        }
    }

    fn radius_search(
        &self,
        node: &Node,
        q: &ArrayView1<'_, f64>,
        r2: f64,
        out: &mut Vec<Neighbor>,
    ) {
        let lb2 = bbox_min_dist2(&node.bbox_lo, &node.bbox_hi, q);
        if lb2 > r2 {
            return;
        }
        if node.left.is_none() && node.right.is_none() {
            for &row in &self.perm[node.lo..node.hi] {
                let d2 = squared_distance(&self.data.row(row), q);
                if d2 <= r2 {
                    out.push(Neighbor {
                        index: row,
                        distance: d2,
                    });
                }
            }
            return;
        }
        if let Some(l) = node.left.as_ref() {
            self.radius_search(l, q, r2, out);
        }
        if let Some(r) = node.right.as_ref() {
            self.radius_search(r, q, r2, out);
        }
    }
}

// ---------------------------------------------------------------------------
// Internal build
// ---------------------------------------------------------------------------

fn build_node(
    x: &ArrayView2<'_, f64>,
    perm: &mut [usize],
    lo: usize,
    hi: usize,
    d: usize,
    leaf_size: usize,
) -> Node {
    let (bbox_lo, bbox_hi) = bbox_of(x, &perm[lo..hi], d);
    if hi - lo <= leaf_size {
        return Node {
            axis: 0,
            split: 0.0,
            lo,
            hi,
            left: None,
            right: None,
            bbox_lo,
            bbox_hi,
        };
    }
    // Choose the axis with maximum spread.
    let mut axis = 0usize;
    let mut best_spread = -1.0_f64;
    for a in 0..d {
        let s = bbox_hi[a] - bbox_lo[a];
        if s > best_spread {
            best_spread = s;
            axis = a;
        }
    }
    // If everything is identical along every axis, fall back to a leaf.
    if best_spread <= 0.0 {
        return Node {
            axis: 0,
            split: 0.0,
            lo,
            hi,
            left: None,
            right: None,
            bbox_lo,
            bbox_hi,
        };
    }
    // Median-split on the chosen axis.
    let sub = &mut perm[lo..hi];
    let n = sub.len();
    let mid = n / 2;
    // Full sort for determinism (simpler than nth-element with a stable tie-break).
    sub.sort_by(|&a, &b| {
        let (va, vb) = (x[[a, axis]], x[[b, axis]]);
        va.partial_cmp(&vb).unwrap().then(a.cmp(&b))
    });
    let split = x[[sub[mid], axis]];
    let left = build_node(x, perm, lo, lo + mid, d, leaf_size);
    let right = build_node(x, perm, lo + mid, hi, d, leaf_size);
    Node {
        axis,
        split,
        lo,
        hi,
        left: Some(Box::new(left)),
        right: Some(Box::new(right)),
        bbox_lo,
        bbox_hi,
    }
}

fn bbox_of(x: &ArrayView2<'_, f64>, indices: &[usize], d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lo = vec![f64::INFINITY; d];
    let mut hi = vec![f64::NEG_INFINITY; d];
    for &i in indices {
        for a in 0..d {
            let v = x[[i, a]];
            if v < lo[a] {
                lo[a] = v;
            }
            if v > hi[a] {
                hi[a] = v;
            }
        }
    }
    (lo, hi)
}

fn bbox_min_dist2(lo: &[f64], hi: &[f64], q: &ArrayView1<'_, f64>) -> f64 {
    let mut s = 0.0_f64;
    for a in 0..lo.len() {
        let qa = q[a];
        if qa < lo[a] {
            let dd = lo[a] - qa;
            s += dd * dd;
        } else if qa > hi[a] {
            let dd = qa - hi[a];
            s += dd * dd;
        }
    }
    s
}

fn squared_distance(a: &ArrayView1<'_, f64>, b: &ArrayView1<'_, f64>) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

// ---------------------------------------------------------------------------
// Max-heap keyed by squared distance. Small and specialised — no crate dep.
// ---------------------------------------------------------------------------

struct MaxHeap {
    data: Vec<Neighbor>,
}

impl MaxHeap {
    fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
        }
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn max_key(&self) -> f64 {
        self.data[0].distance
    }

    fn push(&mut self, n: Neighbor) {
        self.data.push(n);
        let mut i = self.data.len() - 1;
        while i > 0 {
            let p = (i - 1) / 2;
            if self.data[p].distance < self.data[i].distance {
                self.data.swap(p, i);
                i = p;
            } else {
                break;
            }
        }
    }

    fn pop(&mut self) -> Neighbor {
        let last = self.data.pop().unwrap();
        if self.data.is_empty() {
            return last;
        }
        let root = std::mem::replace(&mut self.data[0], last);
        let n = self.data.len();
        let mut i = 0usize;
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut biggest = i;
            if l < n && self.data[l].distance > self.data[biggest].distance {
                biggest = l;
            }
            if r < n && self.data[r].distance > self.data[biggest].distance {
                biggest = r;
            }
            if biggest == i {
                break;
            }
            self.data.swap(i, biggest);
            i = biggest;
        }
        root
    }

    fn into_sorted(self) -> Vec<Neighbor> {
        let mut v = self.data;
        // Report Euclidean distances (sqrt of the squared distance we stored).
        for n in v.iter_mut() {
            n.distance = n.distance.sqrt();
        }
        v.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    #[test]
    fn k_nearest_matches_brute_force_1d() {
        let x = array![
            [1.0],
            [5.0],
            [2.0],
            [9.0],
            [3.0],
            [8.0],
            [4.0],
            [7.0],
            [6.0]
        ];
        let tree = KdTree::build(x.view(), 2).unwrap();
        let q = array![5.5];
        let k = 3;
        let mut expected: Vec<f64> = (0..x.nrows()).map(|i| (x[[i, 0]] - q[0]).abs()).collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let got = tree.k_nearest(q.view(), k).unwrap();
        // Comparing on distances only, since ties can be returned in
        // either index order (both are correct nearest-neighbour answers).
        for (g, &e) in got.iter().zip(expected.iter().take(k)) {
            assert_abs_diff_eq!(g.distance, e, epsilon = 1e-12);
        }
        // The reported indices really are among the sample rows and match
        // their reported distance.
        for g in &got {
            let expected_d = (x[[g.index, 0]] - q[0]).abs();
            assert_abs_diff_eq!(g.distance, expected_d, epsilon = 1e-12);
        }
    }

    #[test]
    fn radius_query_matches_brute_force_2d() {
        let x = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [-1.0, -1.0]
        ];
        let tree = KdTree::build(x.view(), 2).unwrap();
        let q = array![0.5, 0.5];
        let r = 1.0;
        let mut expected: Vec<usize> = (0..x.nrows())
            .filter(|&i| {
                let dx = x[[i, 0]] - q[0];
                let dy = x[[i, 1]] - q[1];
                (dx * dx + dy * dy).sqrt() <= r
            })
            .collect();
        expected.sort();
        let mut got: Vec<usize> = tree
            .radius(q.view(), r)
            .unwrap()
            .into_iter()
            .map(|n| n.index)
            .collect();
        got.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn deterministic_across_builds() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let a = KdTree::build(x.view(), 1).unwrap();
        let b = KdTree::build(x.view(), 1).unwrap();
        assert_eq!(a.perm, b.perm);
    }

    #[test]
    fn rejects_non_finite() {
        let x = array![[f64::NAN]];
        assert!(KdTree::build(x.view(), 1).is_err());
    }

    #[test]
    fn empty_k_zero_returns_empty() {
        let x = array![[1.0]];
        let tree = KdTree::build(x.view(), 1).unwrap();
        let out = tree.k_nearest(Array1::from(vec![0.0]).view(), 0).unwrap();
        assert!(out.is_empty());
    }
}

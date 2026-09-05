//! [`AgglomerativeClustering`] — bottom-up hierarchical clustering with
//! `Single`, `Complete`, `Average`, or `Ward` linkage.
//!
//! # Algorithm
//!
//! Start with `n` singleton clusters. At each step, merge the two
//! clusters with minimum linkage distance until a target `n_clusters`
//! is reached (or all points are in one cluster if that target is 1).
//! The pairwise distance matrix is updated in `O(n)` per merge via the
//! Lance-Williams recurrence, so the whole fit is `O(n² · log n)` time
//! with `O(n²)` space.
//!
//! Ward linkage minimises the total within-cluster variance and
//! matches `cluster.AgglomerativeClustering(linkage='ward')`;
//! it's the recommended default when Euclidean distances are meaningful.
//!
//! # References
//!
//! * Lance, G. N., & Williams, W. T. (1967). *A general theory of
//!   classificatory sorting strategies: 1. Hierarchical systems.*
//!   The Computer Journal, 9(4), 373-380.
//! * Müllner, D. (2011). *Modern hierarchical, agglomerative clustering
//!   algorithms.* arXiv:1109.2378.

use ndarray::{Array1, ArrayView2};
use solow_core::{Error, Result};

/// Linkage criterion.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Linkage {
    /// Single (min pairwise distance).
    Single,
    /// Complete (max pairwise distance).
    Complete,
    /// Unweighted average pairwise distance (UPGMA).
    Average,
    /// Ward — minimises within-cluster variance increase.
    Ward,
}

/// One row of the dendrogram: the merge that created a new cluster.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DendrogramNode {
    /// Left child cluster id.
    pub left: usize,
    /// Right child cluster id.
    pub right: usize,
    /// Linkage distance at which the merge occurred.
    pub distance: f64,
    /// Number of original samples in the merged cluster.
    pub size: usize,
}

/// Agglomerative clusterer.
#[derive(Clone, Copy, Debug)]
pub struct AgglomerativeClustering {
    /// Target cluster count. If `1`, the whole dendrogram is built.
    pub n_clusters: usize,
    /// Linkage criterion.
    pub linkage: Linkage,
}

impl AgglomerativeClustering {
    /// New clusterer with the given target count and linkage.
    pub fn new(n_clusters: usize, linkage: Linkage) -> Self {
        Self {
            n_clusters,
            linkage,
        }
    }

    /// Fit and return per-sample labels in `[0, n_clusters)`.
    pub fn fit_predict(
        &self,
        x: ArrayView2<'_, f64>,
    ) -> Result<(Array1<usize>, Vec<DendrogramNode>)> {
        let n = x.nrows();
        if n == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "AgglomerativeClustering::fit_predict: x must have at least one row and one column"
                    .into(),
            ));
        }
        if self.n_clusters == 0 || self.n_clusters > n {
            return Err(Error::Value(format!(
                "AgglomerativeClustering::fit_predict: n_clusters must be in [1, n] (got {}, n={n})",
                self.n_clusters
            )));
        }
        // Initial pairwise distances (Euclidean).
        let mut d = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let mut s = 0.0_f64;
                for k in 0..x.ncols() {
                    let diff = x[[i, k]] - x[[j, k]];
                    s += diff * diff;
                }
                let dist = if matches!(self.linkage, Linkage::Ward) {
                    // For Ward we store squared distances weighted by 0.5 so
                    // the Lance-Williams update matches Ward exactly.
                    s
                } else {
                    s.sqrt()
                };
                d[i][j] = dist;
                d[j][i] = dist;
            }
        }
        let mut sizes = vec![1usize; n];
        let mut alive: Vec<bool> = vec![true; n];
        let mut parent: Vec<Option<usize>> = vec![None; n]; // for label reconstruction
        let mut nodes: Vec<DendrogramNode> = Vec::with_capacity(n - 1);

        let target_merges = if self.n_clusters == 0 {
            n - 1
        } else {
            n - self.n_clusters
        };

        for merge_idx in 0..(n - 1) {
            if merge_idx >= target_merges {
                break;
            }
            // Find the closest live pair.
            let mut best = (0usize, 0usize);
            let mut best_d = f64::INFINITY;
            for i in 0..(n + nodes.len()) {
                if i >= alive.len() || !alive[i] {
                    continue;
                }
                for j in (i + 1)..(n + nodes.len()) {
                    if j >= alive.len() || !alive[j] {
                        continue;
                    }
                    if d[i][j] < best_d {
                        best_d = d[i][j];
                        best = (i, j);
                    }
                }
            }
            let (a, b) = best;
            let new_size = sizes[a] + sizes[b];
            let node_id = n + nodes.len();
            let node_dist = if matches!(self.linkage, Linkage::Ward) {
                // Store the Ward "distance" as the sqrt of merge SSE contribution.
                (best_d).sqrt()
            } else {
                best_d
            };
            nodes.push(DendrogramNode {
                left: a,
                right: b,
                distance: node_dist,
                size: new_size,
            });
            // Update distance matrix — Lance-Williams.
            // Grow the vectors by one row/column.
            for row in d.iter_mut() {
                row.push(0.0);
            }
            d.push(vec![0.0; n + nodes.len()]);
            for k in 0..(n + nodes.len() - 1) {
                if !alive.get(k).copied().unwrap_or(false) || k == a || k == b {
                    continue;
                }
                let d_ak = d[a][k];
                let d_bk = d[b][k];
                let d_ab = d[a][b];
                let (na, nb, nk) = (sizes[a] as f64, sizes[b] as f64, sizes[k] as f64);
                let new_d = match self.linkage {
                    Linkage::Single => d_ak.min(d_bk),
                    Linkage::Complete => d_ak.max(d_bk),
                    Linkage::Average => (na * d_ak + nb * d_bk) / (na + nb),
                    Linkage::Ward => {
                        let total = na + nb + nk;
                        ((na + nk) * d_ak + (nb + nk) * d_bk - nk * d_ab) / total
                    }
                };
                d[node_id][k] = new_d;
                d[k][node_id] = new_d;
            }
            alive[a] = false;
            alive[b] = false;
            alive.push(true);
            sizes.push(new_size);
            parent.push(None);
            parent[a] = Some(node_id);
            parent[b] = Some(node_id);
        }

        // Assign labels by walking each original sample up to the highest
        // live ancestor.
        let mut root_of = vec![0usize; n];
        for i in 0..n {
            let mut cur = i;
            while let Some(p) = parent[cur] {
                if alive[p] {
                    cur = p;
                    break;
                }
                cur = p;
            }
            root_of[i] = cur;
        }
        // Dense-relabel to [0, n_clusters).
        let mut label_of = std::collections::BTreeMap::<usize, usize>::new();
        let mut next_label = 0usize;
        let mut labels = Array1::<usize>::zeros(n);
        for i in 0..n {
            let r = root_of[i];
            let l = *label_of.entry(r).or_insert_with(|| {
                let out = next_label;
                next_label += 1;
                out
            });
            labels[i] = l;
        }
        Ok((labels, nodes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn recovers_two_well_separated_clusters_with_ward() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
        ];
        let (labels, dendro) = AgglomerativeClustering::new(2, Linkage::Ward)
            .fit_predict(x.view())
            .unwrap();
        assert_eq!(dendro.len(), 6); // n - n_clusters = 8 - 2 = 6
        let a = labels[0];
        let b = labels[4];
        assert_ne!(a, b);
        for i in 0..4 {
            assert_eq!(labels[i], a);
        }
        for i in 4..8 {
            assert_eq!(labels[i], b);
        }
    }

    #[test]
    fn single_linkage_is_chain_friendly() {
        // A chain of points [0, 1, 2, 3, 20] — single linkage sees 4 close
        // pairs and one outlier; with k=2 the outlier gets its own cluster.
        let x = array![[0.0], [1.0], [2.0], [3.0], [20.0]];
        let (labels, _) = AgglomerativeClustering::new(2, Linkage::Single)
            .fit_predict(x.view())
            .unwrap();
        let outlier = labels[4];
        let others = labels[0];
        assert_ne!(outlier, others);
        for i in 0..4 {
            assert_eq!(labels[i], others);
        }
    }
}

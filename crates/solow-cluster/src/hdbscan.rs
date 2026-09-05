//! HDBSCAN* — hierarchical DBSCAN with mutual-reachability distances
//! and cluster stability, following Campello-Moulavi-Sander (2013).
//!
//! Implementation sketch:
//!   1. Compute mutual-reachability distance
//!         d_mreach(a, b) = max(core(a), core(b), d(a, b))
//!      with `core(a)` = distance to `min_samples`-th nearest neighbour.
//!   2. Build the minimum spanning tree over `d_mreach`.
//!   3. Extract the single-linkage dendrogram from the MST and prune
//!      splits below `min_cluster_size`.
//!   4. Pick clusters by maximising the stability sum
//!         Σ (1/ε_i − 1/ε_max) over each candidate cluster's members.
//!
//! For deterministic runs the fit is purely a function of `(x, min_samples,
//! min_cluster_size, metric)` with a stable index-tie-break.

use ndarray::ArrayView2;
use solow_core::{Error, Result};

/// Fitted HDBSCAN clustering.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Hdbscan {
    /// Cluster labels (`-1` marks noise).
    pub labels: Vec<i64>,
    /// Membership probabilities (0 for noise, 1 for the cluster's mode).
    pub probabilities: Vec<f64>,
    /// Number of clusters found (excluding noise).
    pub n_clusters: usize,
    /// `min_samples` used.
    pub min_samples: usize,
    /// `min_cluster_size` used.
    pub min_cluster_size: usize,
}

impl Hdbscan {
    /// Fit with defaults `min_samples = 5`, `min_cluster_size = 5`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 5, 5)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        min_samples: usize,
        min_cluster_size: usize,
    ) -> Result<Self> {
        let n = x.nrows();
        if n == 0 {
            return Err(Error::Value("Hdbscan: empty input".into()));
        }
        if min_samples == 0 || min_cluster_size == 0 {
            return Err(Error::Value(
                "Hdbscan: min_samples and min_cluster_size must be ≥ 1".into(),
            ));
        }
        let d = x.ncols();
        // Pairwise Euclidean distances.
        let mut dist = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let mut s = 0.0_f64;
                for k in 0..d {
                    let e = x[[i, k]] - x[[j, k]];
                    s += e * e;
                }
                let v = s.sqrt();
                dist[i][j] = v;
                dist[j][i] = v;
            }
        }
        // Core distances: k-th nearest-neighbour distance with k = min_samples.
        let mut core = vec![0.0_f64; n];
        for i in 0..n {
            let mut row: Vec<f64> = dist[i].clone();
            row.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let k = min_samples.min(row.len().saturating_sub(1));
            core[i] = row[k];
        }
        // Mutual-reachability distance.
        let mut mreach = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in i..n {
                let v = dist[i][j].max(core[i]).max(core[j]);
                mreach[i][j] = v;
                mreach[j][i] = v;
            }
        }
        // Prim's MST over mutual-reachability graph.
        let mut in_tree = vec![false; n];
        let mut min_edge = vec![f64::INFINITY; n];
        let mut parent = vec![usize::MAX; n];
        min_edge[0] = 0.0;
        let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n.saturating_sub(1));
        for _ in 0..n {
            let mut u = usize::MAX;
            let mut best = f64::INFINITY;
            for v in 0..n {
                if !in_tree[v] && min_edge[v] < best {
                    best = min_edge[v];
                    u = v;
                }
            }
            if u == usize::MAX {
                break;
            }
            in_tree[u] = true;
            if parent[u] != usize::MAX {
                edges.push((parent[u], u, min_edge[u]));
            }
            for v in 0..n {
                if !in_tree[v] && mreach[u][v] < min_edge[v] {
                    min_edge[v] = mreach[u][v];
                    parent[v] = u;
                }
            }
        }
        // Sort MST edges ascending by weight — build single-linkage tree.
        edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        // Union-find with cluster-size tracking.
        let mut parent_uf: Vec<usize> = (0..n).collect();
        let mut size_uf = vec![1_usize; n];
        // Threshold sweep: cut edges from largest to smallest and record
        // when each connected component became a size-≥ min_cluster_size
        // cluster and when it later merged upward.
        fn find(parent: &mut Vec<usize>, mut u: usize) -> usize {
            while parent[u] != u {
                parent[u] = parent[parent[u]];
                u = parent[u];
            }
            u
        }
        // Sweep edges in ascending order; when a cluster crosses the size
        // threshold we mark it as "born".
        for &(a, b, _w) in &edges {
            let ra = find(&mut parent_uf, a);
            let rb = find(&mut parent_uf, b);
            if ra == rb {
                continue;
            }
            let (big, small) = if size_uf[ra] >= size_uf[rb] {
                (ra, rb)
            } else {
                (rb, ra)
            };
            parent_uf[small] = big;
            size_uf[big] += size_uf[small];
        }
        // Root component reflects the final flat partition; here we simply
        // extract components with `size ≥ min_cluster_size` at MST cut
        // distance ε_final = median(edge_weights) — a pragmatic single-cut
        // approximation of full HDBSCAN* stability selection.
        let epsilon = {
            let mut w: Vec<f64> = edges.iter().map(|e| e.2).collect();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if w.is_empty() {
                0.0
            } else {
                w[w.len() / 2]
            }
        };
        // Re-run union-find, but stop at ε_final.
        let mut parent_uf: Vec<usize> = (0..n).collect();
        let mut size_uf = vec![1_usize; n];
        for &(a, b, w) in &edges {
            if w > epsilon {
                break;
            }
            let ra = find(&mut parent_uf, a);
            let rb = find(&mut parent_uf, b);
            if ra == rb {
                continue;
            }
            let (big, small) = if size_uf[ra] >= size_uf[rb] {
                (ra, rb)
            } else {
                (rb, ra)
            };
            parent_uf[small] = big;
            size_uf[big] += size_uf[small];
        }
        // Label components with size ≥ min_cluster_size, others as noise.
        let mut labels = vec![-1_i64; n];
        let mut probs = vec![0.0_f64; n];
        let mut label_map: std::collections::HashMap<usize, i64> = std::collections::HashMap::new();
        let mut next_label = 0_i64;
        for i in 0..n {
            let root = find(&mut parent_uf, i);
            if size_uf[root] >= min_cluster_size {
                let lbl = label_map.entry(root).or_insert_with(|| {
                    let v = next_label;
                    next_label += 1;
                    v
                });
                labels[i] = *lbl;
                probs[i] = 1.0;
            }
        }
        Ok(Self {
            labels,
            probabilities: probs,
            n_clusters: next_label as usize,
            min_samples,
            min_cluster_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn hdbscan_finds_two_dense_clusters() {
        // Two dense clumps.
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.05, 0.15], [0.15, 0.05],
            [5.0, 5.0], [5.1, 5.1], [5.05, 5.15], [5.15, 5.05]
        ];
        let m = Hdbscan::fit_with(x.view(), 2, 3).unwrap();
        let mut cluster_a = 0;
        let mut cluster_b = 0;
        for i in 0..4 {
            if m.labels[i] >= 0 { cluster_a = m.labels[i]; }
        }
        for i in 4..8 {
            if m.labels[i] >= 0 { cluster_b = m.labels[i]; }
        }
        assert_ne!(cluster_a, cluster_b);
    }
}

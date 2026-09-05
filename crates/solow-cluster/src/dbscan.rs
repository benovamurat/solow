//! [`Dbscan`] — Density-Based Spatial Clustering of Applications with
//! Noise (Ester, Kriegel, Sander & Xu 1996).
//!
//! # Algorithm
//!
//! Given a distance ε and a minimum-samples threshold `m`, DBSCAN
//! assigns each point one of three roles:
//!
//! * **Core** — has at least `m` points (including itself) within
//!   distance ε.
//! * **Border** — is not core but lies within ε of some core point.
//! * **Noise** — everything else.
//!
//! Two core points reachable through a chain of core-point ε-jumps
//! belong to the same cluster; border points inherit the label of
//! whatever core point they were reached from.
//!
//! The great virtues of DBSCAN over KMeans are: (a) it does not need
//! `k` a priori, (b) it discovers non-convex clusters, and (c) it
//! isolates noise points instead of forcing them into a cluster.
//!
//! # Complexity
//!
//! Naive neighbour lookup is `O(n²)`. For low-dimensional data, pair
//! this crate with `solow-neighbors::KDTree` for `O(n · log n)`
//! neighbour queries.
//!
//! # Reference
//!
//! Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A
//! density-based algorithm for discovering clusters in large spatial
//! databases with noise.* KDD-96, 226-231.

use ndarray::{Array1, ArrayView2};
use solow_core::{Error, Result};

/// Role assigned to each sample by DBSCAN.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PointRole {
    /// Core point (has ≥ `min_samples` neighbours within ε).
    Core,
    /// Border point (in the ε-neighbourhood of some core point).
    Border,
    /// Noise point (no cluster assignment).
    Noise,
}

/// Output of [`Dbscan::fit`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct DbscanResult {
    /// Cluster label per sample. Noise points get `usize::MAX`.
    pub labels: Array1<usize>,
    /// Per-sample role.
    pub roles: Vec<PointRole>,
    /// Number of clusters found.
    pub n_clusters: usize,
    /// Number of noise points.
    pub n_noise: usize,
}

/// DBSCAN clusterer.
#[derive(Clone, Copy, Debug)]
pub struct Dbscan {
    /// Neighbourhood radius (ε).
    pub eps: f64,
    /// Minimum samples in an ε-ball (including the query point) for
    /// a point to be labelled *core*.
    pub min_samples: usize,
}

impl Dbscan {
    /// Build a DBSCAN clusterer.
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self { eps, min_samples }
    }

    /// Fit on `x` and return per-sample labels and roles.
    ///
    /// Uses Euclidean distance and naive `O(n²)` neighbour lookup —
    /// for large `n` in low dimensions, precompute neighbours with a
    /// [`solow-neighbors::KDTree`](https://docs.rs/solow-neighbors) and
    /// feed them into a caller-side loop.
    pub fn fit(&self, x: ArrayView2<'_, f64>) -> Result<DbscanResult> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "Dbscan::fit: x must have at least one row and one column".into(),
            ));
        }
        if !(self.eps > 0.0 && self.eps.is_finite()) {
            return Err(Error::Value(format!(
                "Dbscan::fit: eps must be finite and > 0 (got {})",
                self.eps
            )));
        }
        if self.min_samples < 1 {
            return Err(Error::Value("Dbscan::fit: min_samples must be ≥ 1".into()));
        }
        let n = x.nrows();
        // Precompute neighbours for every point.
        let neighbours: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                let mut nbrs = Vec::new();
                for j in 0..n {
                    if i == j {
                        nbrs.push(j);
                        continue;
                    }
                    let mut dd = 0.0_f64;
                    for k in 0..x.ncols() {
                        let diff = x[[i, k]] - x[[j, k]];
                        dd += diff * diff;
                    }
                    if dd.sqrt() <= self.eps {
                        nbrs.push(j);
                    }
                }
                nbrs
            })
            .collect();

        let mut labels = Array1::<usize>::from_elem(n, usize::MAX);
        let mut roles = vec![PointRole::Noise; n];
        let mut cluster_id = 0usize;

        for p in 0..n {
            if labels[p] != usize::MAX {
                continue;
            }
            if neighbours[p].len() < self.min_samples {
                // Might still become a border point later; leave as Noise for now.
                continue;
            }
            // Start a new cluster.
            let c = cluster_id;
            cluster_id += 1;
            labels[p] = c;
            roles[p] = PointRole::Core;
            // Seed set — BFS outward.
            let mut queue: Vec<usize> = neighbours[p].iter().copied().filter(|&i| i != p).collect();
            let mut head = 0usize;
            while head < queue.len() {
                let q = queue[head];
                head += 1;
                if labels[q] == usize::MAX {
                    labels[q] = c;
                    if neighbours[q].len() >= self.min_samples {
                        roles[q] = PointRole::Core;
                        for &nb in &neighbours[q] {
                            if !queue.contains(&nb) {
                                queue.push(nb);
                            }
                        }
                    } else if roles[q] == PointRole::Noise {
                        roles[q] = PointRole::Border;
                    }
                } else if roles[q] == PointRole::Noise {
                    // Already labelled by an earlier iteration; if it's noise,
                    // relabel as Border and give it the current cluster.
                    labels[q] = c;
                    roles[q] = PointRole::Border;
                }
            }
        }
        let n_noise = labels.iter().filter(|&&l| l == usize::MAX).count();
        Ok(DbscanResult {
            labels,
            roles,
            n_clusters: cluster_id,
            n_noise,
        })
    }

    /// Fit and return labels only.
    pub fn fit_predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<usize>> {
        Ok(self.fit(x)?.labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn finds_two_clusters_and_isolates_noise() {
        // Two dense blobs plus one lone outlier.
        let x = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
            [100.0, 100.0]
        ];
        let r = Dbscan::new(0.5, 3).fit(x.view()).unwrap();
        assert_eq!(r.n_clusters, 2);
        assert_eq!(r.n_noise, 1);
        assert_eq!(r.labels[8], usize::MAX);
        // Points 0..4 share one label; 4..8 share another.
        for i in 1..4 {
            assert_eq!(r.labels[i], r.labels[0]);
        }
        for i in 5..8 {
            assert_eq!(r.labels[i], r.labels[4]);
        }
    }

    #[test]
    fn rejects_bad_params() {
        let x = array![[1.0]];
        assert!(Dbscan::new(-1.0, 3).fit(x.view()).is_err());
        assert!(Dbscan::new(1.0, 0).fit(x.view()).is_err());
    }
}

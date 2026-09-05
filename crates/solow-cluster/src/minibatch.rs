//! MiniBatchKMeans (Sculley 2010) and BisectingKMeans (Steinbach-
//! Karypis-Kumar 2000) — two additional centroid-based clusterers.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::kmeans::{KMeans, KMeansInit};

/// MiniBatch K-Means — stochastic Lloyd's update on random subsets.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MiniBatchKMeans {
    /// Fitted centroids.
    pub centroids: Array2<f64>,
    /// Per-sample cluster label.
    pub labels: Vec<usize>,
    /// Iterations run.
    pub n_iter: usize,
    /// Cluster count.
    pub n_clusters: usize,
    /// Seed used.
    pub seed: u64,
}

impl MiniBatchKMeans {
    /// Fit with defaults `batch_size = 100`, `max_iter = 100`, `seed = 0`.
    pub fn fit(x: ArrayView2<'_, f64>, n_clusters: usize) -> Result<Self> {
        Self::fit_with(x, n_clusters, 100, 100, 0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_clusters: usize,
        batch_size: usize,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if n_clusters == 0 || n_clusters > n {
            return Err(Error::Value(format!(
                "MiniBatchKMeans: n_clusters must be in [1, {n}] (got {n_clusters})"
            )));
        }
        let batch = batch_size.min(n).max(1);
        // k-means++ init from a full-batch k-means with a small max_iter.
        let seed_km = KMeans::new(n_clusters, seed).init(KMeansInit::KMeansPlusPlus).max_iter(1).n_init(1);
        let init = seed_km.fit(x)?;
        let mut centroids = init.centroids.clone();
        let mut counts = vec![0_u64; n_clusters];
        let mut state = seed.wrapping_add(0xF00D_D00D_C0DE);
        let mut iters = 0_usize;
        for it in 0..max_iter {
            iters = it + 1;
            // Sample a mini-batch (with replacement).
            let mut indices = vec![0_usize; batch];
            for j in 0..batch {
                indices[j] = uniform_index(&mut state, n as u64);
            }
            for &i in &indices {
                // Assign to nearest centroid.
                let mut best = 0;
                let mut best_d = f64::INFINITY;
                for c in 0..n_clusters {
                    let mut s = 0.0_f64;
                    for k in 0..d {
                        let e = x[[i, k]] - centroids[[c, k]];
                        s += e * e;
                    }
                    if s < best_d {
                        best_d = s;
                        best = c;
                    }
                }
                counts[best] += 1;
                let eta = 1.0 / counts[best] as f64;
                for k in 0..d {
                    centroids[[best, k]] = (1.0 - eta) * centroids[[best, k]] + eta * x[[i, k]];
                }
            }
        }
        // Final assignment.
        let mut labels = vec![0_usize; n];
        for i in 0..n {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for c in 0..n_clusters {
                let mut s = 0.0_f64;
                for k in 0..d {
                    let e = x[[i, k]] - centroids[[c, k]];
                    s += e * e;
                }
                if s < best_d {
                    best_d = s;
                    best = c;
                }
            }
            labels[i] = best;
        }
        Ok(Self {
            centroids,
            labels,
            n_iter: iters,
            n_clusters,
            seed,
        })
    }
}

fn uniform_index(state: &mut u64, n: u64) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let max = u64::MAX - (u64::MAX % n);
    if *state < max {
        (*state % n) as usize
    } else {
        (state.wrapping_mul(3) % n) as usize
    }
}

/// Bisecting K-Means — hierarchical binary split of the largest
/// cluster until `n_clusters` leaves are reached.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BisectingKMeans {
    /// Fitted centroids.
    pub centroids: Array2<f64>,
    /// Per-sample cluster label.
    pub labels: Vec<usize>,
    /// Cluster count.
    pub n_clusters: usize,
}

impl BisectingKMeans {
    /// Fit with defaults `seed = 0`.
    pub fn fit(x: ArrayView2<'_, f64>, n_clusters: usize) -> Result<Self> {
        Self::fit_with(x, n_clusters, 0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_clusters: usize,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        if n_clusters == 0 || n_clusters > n {
            return Err(Error::Value("BisectingKMeans: n_clusters out of range".into()));
        }
        let mut clusters: Vec<Vec<usize>> = vec![(0..n).collect()];
        while clusters.len() < n_clusters {
            // Pick the largest cluster and 2-split it.
            let (idx, _) = clusters
                .iter()
                .enumerate()
                .max_by_key(|(_, c)| c.len())
                .unwrap();
            let members = clusters.remove(idx);
            if members.len() < 2 {
                clusters.push(members);
                break;
            }
            let sub = row_subset(x, &members);
            let km = KMeans::new(2, seed + clusters.len() as u64).n_init(3).fit(sub.view())?;
            let mut left: Vec<usize> = Vec::new();
            let mut right: Vec<usize> = Vec::new();
            for (r, &m) in members.iter().enumerate() {
                if km.labels[r] == 0 {
                    left.push(m);
                } else {
                    right.push(m);
                }
            }
            clusters.push(left);
            clusters.push(right);
        }
        let d = x.ncols();
        let mut centroids = Array2::<f64>::zeros((clusters.len(), d));
        let mut labels = vec![0_usize; n];
        for (c, cluster) in clusters.iter().enumerate() {
            for k in 0..d {
                let mut s = 0.0_f64;
                for &i in cluster {
                    s += x[[i, k]];
                }
                centroids[[c, k]] = if cluster.is_empty() {
                    0.0
                } else {
                    s / cluster.len() as f64
                };
            }
            for &i in cluster {
                labels[i] = c;
            }
        }
        Ok(Self {
            centroids,
            labels,
            n_clusters: clusters.len(),
        })
    }
}

fn row_subset(x: ArrayView2<'_, f64>, rows: &[usize]) -> Array2<f64> {
    let d = x.ncols();
    let mut out = Array2::<f64>::zeros((rows.len(), d));
    for (r, &i) in rows.iter().enumerate() {
        for j in 0..d {
            out[[r, j]] = x[[i, j]];
        }
    }
    out
}

// Prevent unused-import warning.
#[allow(dead_code)]
fn _touch(a: Array1<f64>) -> Array1<f64> {
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn mini_batch_kmeans_splits_two_dense_clumps() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.2]
        ];
        let m = MiniBatchKMeans::fit_with(x.view(), 2, 6, 30, 42).unwrap();
        assert_ne!(m.labels[0], m.labels[3]);
    }

    #[test]
    fn bisecting_kmeans_splits_two_dense_clumps() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.2]
        ];
        let m = BisectingKMeans::fit(x.view(), 2).unwrap();
        assert_ne!(m.labels[0], m.labels[3]);
    }
}

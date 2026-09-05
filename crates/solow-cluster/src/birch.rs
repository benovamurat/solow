//! BIRCH — Balanced Iterative Reducing and Clustering using Hierarchies
//! (Zhang-Ramakrishnan-Livny 1996).
//!
//! Online-incremental hierarchical clustering built around a
//! Clustering-Feature (CF) tree. In this compact implementation we keep
//! the leaf-node layer only — sufficient for the reference `Birch(n_clusters=None)`
//! semantics — and then optionally re-cluster the CF centroids using
//! agglomerative merging until `n_clusters` remains.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::agglomerative::{AgglomerativeClustering, Linkage};

/// Fitted BIRCH model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Birch {
    /// Sub-cluster centres (`n_sub × d`), each averaged from its CF.
    pub subcluster_centers: Array2<f64>,
    /// Sub-cluster label per input row.
    pub subcluster_labels: Vec<i64>,
    /// Optional final labels if `n_clusters` was set.
    pub labels: Vec<i64>,
    /// Threshold on the sub-cluster radius.
    pub threshold: f64,
    /// If `Some`, the number of hierarchical clusters returned in
    /// `labels`; if `None`, `labels == subcluster_labels`.
    pub n_clusters: Option<usize>,
}

impl Birch {
    /// Fit with `threshold = 0.5`, `n_clusters = None`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 0.5, None)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        threshold: f64,
        n_clusters: Option<usize>,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if n == 0 {
            return Err(Error::Value("Birch: empty input".into()));
        }
        if threshold <= 0.0 {
            return Err(Error::Value("Birch: threshold must be > 0".into()));
        }
        // Compact "CF" layer: (ls, ss, n) per subcluster.
        let mut ls: Vec<Vec<f64>> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        let mut sub_labels = vec![0_i64; n];
        for i in 0..n {
            let xi: Vec<f64> = (0..d).map(|k| x[[i, k]]).collect();
            let mut best = usize::MAX;
            let mut best_dist = f64::INFINITY;
            for (c, sums) in ls.iter().enumerate() {
                let mut s = 0.0_f64;
                let inv = 1.0 / counts[c] as f64;
                for k in 0..d {
                    let d_val = xi[k] - sums[k] * inv;
                    s += d_val * d_val;
                }
                let dist = s.sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            if best != usize::MAX && best_dist <= threshold {
                for k in 0..d {
                    ls[best][k] += xi[k];
                }
                counts[best] += 1;
                sub_labels[i] = best as i64;
            } else {
                sub_labels[i] = ls.len() as i64;
                ls.push(xi);
                counts.push(1);
            }
        }
        let n_sub = ls.len();
        let mut centers = Array2::<f64>::zeros((n_sub, d));
        for c in 0..n_sub {
            let inv = 1.0 / counts[c] as f64;
            for k in 0..d {
                centers[[c, k]] = ls[c][k] * inv;
            }
        }
        let final_labels: Vec<i64> = match n_clusters {
            None => sub_labels.clone(),
            Some(k) if k >= n_sub => sub_labels.clone(),
            Some(k) => {
                // Agglomerate sub-clusters down to k using Ward linkage.
                let agg = AgglomerativeClustering::new(k, Linkage::Ward);
                let (agg_labels, _dendro) = agg.fit_predict(centers.view())?;
                sub_labels
                    .iter()
                    .map(|&sl| agg_labels[sl as usize] as i64)
                    .collect()
            }
        };
        Ok(Self {
            subcluster_centers: centers,
            subcluster_labels: sub_labels,
            labels: final_labels,
            threshold,
            n_clusters,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn birch_forms_two_subclusters_for_two_clumps() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.05, 0.15],
            [5.0, 5.0], [5.1, 5.1], [5.05, 5.15]
        ];
        let b = Birch::fit_with(x.view(), 0.5, Some(2)).unwrap();
        assert_ne!(b.labels[0], b.labels[3]);
    }
}

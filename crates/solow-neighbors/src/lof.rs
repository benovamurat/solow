//! Local Outlier Factor (Breunig-Kriegel-Ng-Sander 2000).
//!
//! Computes for each point the ratio of its k-nearest-neighbours' local
//! reachability density to its own. Points with `lof ≫ 1` are anomalies.

use ndarray::ArrayView2;
use solow_core::{Error, Result};

/// Fitted LocalOutlierFactor.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LocalOutlierFactor {
    /// LOF score per input row (`~1` = inlier, `≫ 1` = outlier).
    pub lof: Vec<f64>,
    /// Inlier / outlier flag: `1` = inlier, `-1` = outlier.
    pub predictions: Vec<i64>,
    /// `k` used.
    pub n_neighbors: usize,
    /// Contamination threshold (fraction expected outliers).
    pub contamination: f64,
}

impl LocalOutlierFactor {
    /// Fit with the reference defaults `n_neighbors = 20`, `contamination = 0.1`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 20, 0.1)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_neighbors: usize,
        contamination: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        if n < 2 {
            return Err(Error::Value("LOF: need ≥ 2 samples".into()));
        }
        if n_neighbors == 0 || n_neighbors >= n {
            return Err(Error::Value(format!(
                "LOF: n_neighbors must be in [1, {}] (got {n_neighbors})",
                n - 1
            )));
        }
        let d = x.ncols();
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
        // k-distance and neighbour lists.
        let mut k_dist = vec![0.0_f64; n];
        let mut neighbours: Vec<Vec<usize>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut idx: Vec<(usize, f64)> =
                (0..n).filter(|&j| j != i).map(|j| (j, dist[i][j])).collect();
            idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let cutoff = idx[n_neighbors - 1].1;
            k_dist[i] = cutoff;
            let nb: Vec<usize> = idx
                .iter()
                .filter(|(_, d)| *d <= cutoff + 1e-12)
                .map(|(j, _)| *j)
                .collect();
            neighbours.push(nb);
        }
        // Reachability distance: max(k_dist(o), d(p, o)).
        // Local reachability density: 1 / mean_o reach(p, o).
        let mut lrd = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = 0.0_f64;
            for &j in &neighbours[i] {
                s += k_dist[j].max(dist[i][j]);
            }
            let mean_reach = (s / neighbours[i].len().max(1) as f64).max(1e-30);
            lrd[i] = 1.0 / mean_reach;
        }
        // LOF: mean over neighbours of lrd[j] / lrd[i].
        let mut lof = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = 0.0_f64;
            for &j in &neighbours[i] {
                s += lrd[j];
            }
            let mean = s / neighbours[i].len().max(1) as f64;
            lof[i] = mean / lrd[i].max(1e-30);
        }
        // Contamination threshold: mark the top-fraction as outliers.
        let mut sorted = lof.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let cutoff_rank = ((n as f64) * contamination).ceil() as usize;
        let threshold = if cutoff_rank == 0 || cutoff_rank > n {
            f64::INFINITY
        } else {
            sorted[cutoff_rank - 1]
        };
        let predictions: Vec<i64> = lof
            .iter()
            .map(|&v| if v >= threshold { -1 } else { 1 })
            .collect();
        Ok(Self {
            lof,
            predictions,
            n_neighbors,
            contamination,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn lof_flags_a_lone_outlier() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2], [0.15, 0.05],
            [0.05, 0.15], [50.0, 50.0]
        ];
        let m = LocalOutlierFactor::fit_with(x.view(), 3, 0.2).unwrap();
        assert_eq!(m.predictions[5], -1, "the point at (50, 50) should be flagged");
    }
}

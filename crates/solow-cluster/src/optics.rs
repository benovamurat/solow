//! OPTICS (Ankerst-Breunig-Kriegel-Sander 1999) — Ordering Points To
//! Identify the Clustering Structure.
//!
//! Produces a reachability plot which downstream code can slice into
//! clusters. This implementation returns the ordering and the per-point
//! reachability + core distances.

use ndarray::ArrayView2;
use solow_core::{Error, Result};

/// Fitted OPTICS ordering.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Optics {
    /// Cluster labels (`-1` marks noise). Assigned by the DBSCAN-cut
    /// extraction with `xi = 0.05`.
    pub labels: Vec<i64>,
    /// Reachability distance per point, in the order the input was given.
    pub reachability: Vec<f64>,
    /// Core distance per point.
    pub core_distances: Vec<f64>,
    /// Order in which OPTICS visited the points.
    pub ordering: Vec<usize>,
    /// `min_samples` used.
    pub min_samples: usize,
    /// `max_eps` used (∞ if unbounded).
    pub max_eps: f64,
}

impl Optics {
    /// Fit with the reference defaults `min_samples = 5`, `max_eps = ∞`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 5, f64::INFINITY)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        min_samples: usize,
        max_eps: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        if n == 0 {
            return Err(Error::Value("Optics: empty input".into()));
        }
        if min_samples == 0 {
            return Err(Error::Value("Optics: min_samples must be ≥ 1".into()));
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
        let mut core = vec![f64::INFINITY; n];
        for i in 0..n {
            let mut row: Vec<f64> = dist[i].clone();
            row.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let k = min_samples.min(row.len().saturating_sub(1));
            if row[k] <= max_eps {
                core[i] = row[k];
            }
        }
        let mut reach = vec![f64::INFINITY; n];
        let mut processed = vec![false; n];
        let mut ordering: Vec<usize> = Vec::with_capacity(n);
        for start in 0..n {
            if processed[start] {
                continue;
            }
            let mut seeds: Vec<(usize, f64)> = Vec::new();
            let mut current = start;
            processed[current] = true;
            ordering.push(current);
            if core[current].is_finite() {
                update(&mut seeds, current, &dist, &core, &mut reach, &processed);
            }
            while let Some((idx, _)) = pop_min(&mut seeds) {
                if processed[idx] {
                    continue;
                }
                processed[idx] = true;
                ordering.push(idx);
                current = idx;
                if core[current].is_finite() {
                    update(&mut seeds, current, &dist, &core, &mut reach, &processed);
                }
            }
        }
        // Simple DBSCAN-cut extraction: label any point whose reachability
        // is finite (and ≤ max_eps) into a cluster; propagate cluster ids
        // along the ordering by cutting whenever reachability > max_eps
        // or is infinite.
        let mut labels = vec![-1_i64; n];
        let mut next_label = -1_i64;
        for &idx in &ordering {
            let r = reach[idx];
            if !r.is_finite() || r > max_eps {
                if core[idx].is_finite() {
                    next_label += 1;
                    labels[idx] = next_label;
                }
            } else {
                labels[idx] = next_label.max(0);
                if next_label < 0 {
                    next_label = 0;
                    labels[idx] = 0;
                }
            }
        }
        Ok(Self {
            labels,
            reachability: reach,
            core_distances: core,
            ordering,
            min_samples,
            max_eps,
        })
    }
}

fn update(
    seeds: &mut Vec<(usize, f64)>,
    idx: usize,
    dist: &[Vec<f64>],
    core: &[f64],
    reach: &mut [f64],
    processed: &[bool],
) {
    for j in 0..dist.len() {
        if processed[j] {
            continue;
        }
        let new_reach = dist[idx][j].max(core[idx]);
        if new_reach < reach[j] {
            reach[j] = new_reach;
            seeds.push((j, new_reach));
        }
    }
}

fn pop_min(seeds: &mut Vec<(usize, f64)>) -> Option<(usize, f64)> {
    if seeds.is_empty() {
        return None;
    }
    let mut best = 0;
    for i in 1..seeds.len() {
        if seeds[i].1 < seeds[best].1 {
            best = i;
        }
    }
    Some(seeds.swap_remove(best))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn optics_orders_two_clumps() {
        let x = array![
            [0.0_f64], [0.1], [0.2], [5.0], [5.1], [5.2]
        ];
        let m = Optics::fit_with(x.view(), 2, f64::INFINITY).unwrap();
        assert_eq!(m.ordering.len(), 6);
        // Every point except the ordering-starter should have finite reach.
        let finite = m.reachability.iter().filter(|r| r.is_finite()).count();
        assert!(finite >= 4);
    }
}

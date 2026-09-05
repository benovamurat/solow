//! [`KBinsDiscretizer`] — bin each column of a feature matrix into a
//! small integer alphabet.
//!
//! Three binning strategies:
//!
//! * [`BinStrategy::Uniform`] — equal-width bins between the observed
//!   `min` and `max`.
//! * [`BinStrategy::Quantile`] — equal-frequency bins from R type-7
//!   quantiles (matches `numpy.quantile` and
//!   `preprocessing.KBinsDiscretizer(strategy='quantile')`).
//! * [`BinStrategy::KMeans`] — 1-D KMeans with `k-means++` initialisation
//!   and deterministic MMIX-LCG seeding, so the resulting bin edges are
//!   bit-for-bit reproducible across runs and platforms.
//!
//! # Output
//!
//! `transform` returns a `n × d` matrix of `usize` bin indices in
//! `[0, n_bins)` — the ordinal encoding the reference calls
//! `encode='ordinal'`. The one-hot encoding is trivially obtained by
//! piping the output through
//! [`crate::encoders::OneHotEncoder`] on the resulting numeric-
//! categorical matrix.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Binning strategy for [`KBinsDiscretizer`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinStrategy {
    /// Equal-width bins on `[min, max]`.
    Uniform,
    /// Equal-frequency bins (per-column quantiles).
    Quantile,
    /// 1-D KMeans with `k-means++` init.
    KMeans,
}

/// Fitted per-column bin edges.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct KBinsDiscretizer {
    /// Per-column bin edges. Column `j` has `edges[j]` of length
    /// `n_bins + 1`, ascending.
    pub edges: Vec<Vec<f64>>,
    /// Number of bins per column (same value for every column in this
    /// implementation, matching the reference default).
    pub n_bins: usize,
    /// The binning strategy used at fit time.
    pub strategy: BinStrategy,
}

impl KBinsDiscretizer {
    /// Fit with the given number of bins and strategy.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        n_bins: usize,
        strategy: BinStrategy,
        seed: u64,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "KBinsDiscretizer::fit: x must have at least one row and one column".into(),
            ));
        }
        if n_bins < 2 {
            return Err(Error::Value(format!(
                "KBinsDiscretizer::fit: n_bins must be ≥ 2 (got {n_bins})"
            )));
        }
        let d = x.ncols();
        let mut edges = Vec::with_capacity(d);
        for j in 0..d {
            let col: Vec<f64> = x.column(j).iter().copied().collect();
            let e = match strategy {
                BinStrategy::Uniform => uniform_edges(&col, n_bins)?,
                BinStrategy::Quantile => quantile_edges(&col, n_bins)?,
                BinStrategy::KMeans => kmeans_edges(&col, n_bins, seed.wrapping_add(j as u64))?,
            };
            edges.push(e);
        }
        Ok(Self {
            edges,
            n_bins,
            strategy,
        })
    }

    /// Assign each value to a bin index in `[0, n_bins)`.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<usize>> {
        if x.ncols() != self.edges.len() {
            return Err(Error::Shape(format!(
                "KBinsDiscretizer::transform: expected {} columns, got {}",
                self.edges.len(),
                x.ncols()
            )));
        }
        let mut out = Array2::<usize>::zeros((x.nrows(), x.ncols()));
        for j in 0..x.ncols() {
            let e = &self.edges[j];
            for i in 0..x.nrows() {
                let v = x[[i, j]];
                // Half-open bins `[e[k], e[k+1])`, last bin closed at both ends.
                let mut b = 0usize;
                for k in 0..(e.len() - 1) {
                    if v >= e[k] {
                        b = k;
                    }
                }
                if b >= self.n_bins {
                    b = self.n_bins - 1;
                }
                out[[i, j]] = b;
            }
        }
        Ok(out)
    }

    /// One-call fit + transform.
    pub fn fit_transform(
        x: ArrayView2<'_, f64>,
        n_bins: usize,
        strategy: BinStrategy,
        seed: u64,
    ) -> Result<(Self, Array2<usize>)> {
        let d = Self::fit(x, n_bins, strategy, seed)?;
        let t = d.transform(x)?;
        Ok((d, t))
    }
}

fn uniform_edges(col: &[f64], n_bins: usize) -> Result<Vec<f64>> {
    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in col {
        if !v.is_finite() {
            return Err(Error::Value("KBinsDiscretizer: non-finite value".into()));
        }
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    if mn == mx {
        // Degenerate column — assign every value to bin 0 by returning trivial edges.
        return Ok(vec![mn, mn + 1.0]);
    }
    let w = (mx - mn) / n_bins as f64;
    Ok((0..=n_bins).map(|k| mn + k as f64 * w).collect())
}

fn quantile_edges(col: &[f64], n_bins: usize) -> Result<Vec<f64>> {
    let mut sorted: Vec<f64> = col.to_vec();
    for &v in &sorted {
        if !v.is_finite() {
            return Err(Error::Value("KBinsDiscretizer: non-finite value".into()));
        }
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let mut edges = Vec::with_capacity(n_bins + 1);
    for k in 0..=n_bins {
        let q = k as f64 / n_bins as f64;
        let h = (n as f64 - 1.0) * q;
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = h - lo as f64;
        edges.push((1.0 - frac) * sorted[lo] + frac * sorted[hi]);
    }
    Ok(edges)
}

fn kmeans_edges(col: &[f64], n_bins: usize, seed: u64) -> Result<Vec<f64>> {
    // Simple 1-D KMeans with k-means++ init.
    for &v in col {
        if !v.is_finite() {
            return Err(Error::Value("KBinsDiscretizer: non-finite value".into()));
        }
    }
    let n = col.len();
    if n < n_bins {
        return Err(Error::Value(format!(
            "KBinsDiscretizer: KMeans strategy needs n ≥ n_bins (got n={n}, k={n_bins})"
        )));
    }
    let mut centers = kmeans_pp_init_1d(col, n_bins, seed);
    let max_iter = 100;
    for _ in 0..max_iter {
        // Assign
        let mut sums = vec![0.0_f64; n_bins];
        let mut counts = vec![0usize; n_bins];
        for &v in col {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (c, &cent) in centers.iter().enumerate() {
                let dd = (v - cent).powi(2);
                if dd < best_d {
                    best_d = dd;
                    best = c;
                }
            }
            sums[best] += v;
            counts[best] += 1;
        }
        let mut moved = 0.0_f64;
        for c in 0..n_bins {
            if counts[c] > 0 {
                let new_c = sums[c] / counts[c] as f64;
                moved += (new_c - centers[c]).abs();
                centers[c] = new_c;
            }
        }
        if moved < 1e-12 {
            break;
        }
    }
    centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Edges: midpoints between adjacent sorted centers, plus min/max as
    // the outer edges.
    let mut sorted = col.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut edges = Vec::with_capacity(n_bins + 1);
    edges.push(sorted[0]);
    for w in centers.windows(2) {
        edges.push(0.5 * (w[0] + w[1]));
    }
    edges.push(*sorted.last().unwrap());
    Ok(edges)
}

fn kmeans_pp_init_1d(col: &[f64], k: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_add(0xC0DE_C0DE_C0DE_C0DE);
    let mut centers: Vec<f64> = Vec::with_capacity(k);
    // First centre: pick uniformly at random.
    centers.push(col[uniform_index(&mut state, col.len() as u64)]);
    while centers.len() < k {
        // D²-proportional sampling.
        let d2: Vec<f64> = col
            .iter()
            .map(|&v| {
                centers
                    .iter()
                    .map(|&c| (v - c).powi(2))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let total: f64 = d2.iter().sum();
        let target = uniform_f64(&mut state) * total;
        let mut acc = 0.0_f64;
        let mut pick = 0usize;
        for (i, &d) in d2.iter().enumerate() {
            acc += d;
            if acc >= target {
                pick = i;
                break;
            }
        }
        centers.push(col[pick]);
    }
    centers
}

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn uniform_index(state: &mut u64, n: u64) -> usize {
    let max = u64::MAX - (u64::MAX % n);
    loop {
        let r = lcg_next(state);
        if r < max {
            return (r % n) as usize;
        }
    }
}

fn uniform_f64(state: &mut u64) -> f64 {
    // 53-bit uniform on [0, 1).
    (lcg_next(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn uniform_bins_partition_the_range() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let (d, t) = KBinsDiscretizer::fit_transform(x.view(), 4, BinStrategy::Uniform, 0).unwrap();
        assert_eq!(d.edges[0].len(), 5);
        // Edges are [0, 1, 2, 3, 4]; bins are the half-open intervals
        // `[0, 1)`, `[1, 2)`, `[2, 3)`, `[3, 4]`. Values 0.0..4.0 fall in
        // bins 0..3, with 4.0 clamped to the last bin (3).
        assert_eq!(t.column(0).to_vec(), vec![0, 1, 2, 3, 3]);
    }

    #[test]
    fn quantile_bins_are_equal_frequency() {
        // 8 samples, 4 bins → 2 per bin.
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let (_d, t) =
            KBinsDiscretizer::fit_transform(x.view(), 4, BinStrategy::Quantile, 0).unwrap();
        let mut counts = [0usize; 4];
        for &b in t.column(0).iter() {
            counts[b] += 1;
        }
        for &c in &counts {
            assert!(c >= 1 && c <= 3, "counts = {counts:?}");
        }
    }

    #[test]
    fn kmeans_is_deterministic_by_seed() {
        let x = array![[1.0], [2.0], [10.0], [11.0], [20.0], [21.0]];
        let a = KBinsDiscretizer::fit(x.view(), 3, BinStrategy::KMeans, 42).unwrap();
        let b = KBinsDiscretizer::fit(x.view(), 3, BinStrategy::KMeans, 42).unwrap();
        assert_eq!(a.edges, b.edges);
    }
}

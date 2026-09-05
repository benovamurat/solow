//! [`KMeans`] — Lloyd's algorithm with `k-means++` initialisation.
//!
//! # Algorithm
//!
//! Given a data matrix `X ∈ ℝ^{n × d}` and a number of clusters `k`,
//! KMeans minimises the within-cluster sum of squared distances
//! (inertia):
//!
//! ```text
//! J(C, {μ_c}) = Σ_i ‖x_i − μ_{c(i)}‖²
//! ```
//!
//! by alternating (Lloyd 1957):
//!
//! 1. **Assignment.** For each `xᵢ`, `c(i) ← argmin_c ‖xᵢ − μ_c‖²`.
//! 2. **Update.** For each `c`, `μ_c ← mean_{i : c(i) = c} xᵢ`.
//!
//! Iteration continues until the maximum centre shift falls below
//! `tolerance` or `max_iter` is reached. `fit` is re-run `n_init`
//! times with different seeds and the best inertia is kept — the
//! the reference default of 10 restarts is retained here.
//!
//! # Initialisation
//!
//! Two options:
//!
//! * [`KMeansInit::Random`] — pick `k` distinct rows uniformly at
//!   random. Cheap but variance-prone.
//! * [`KMeansInit::KMeansPlusPlus`] (default) — the Arthur-Vassilvitskii
//!   `k-means++` scheme: pick the first centre uniformly, then each
//!   subsequent centre with probability proportional to squared
//!   distance from the nearest existing centre. Guarantees a
//!   `Θ(log k)`-competitive initial inertia in expectation.
//!
//! # Determinism
//!
//! Random draws use the portable MMIX 64-bit LCG shared with
//! [`solow-cv`](https://docs.rs/solow-cv). A given `seed` produces
//! bit-identical fits across runs and platforms.
//!
//! # References
//!
//! * Lloyd, S. (1982). *Least Squares Quantization in PCM*.
//!   IEEE Transactions on Information Theory, 28(2), 129-137.
//! * Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The advantages
//!   of careful seeding*. SODA '07, 1027-1035.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Centroid-initialisation scheme.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KMeansInit {
    /// Uniform random draw of `k` distinct rows.
    Random,
    /// The Arthur-Vassilvitskii `k-means++` seeding (the recommended
    /// default). Guarantees `Θ(log k)`-competitive initial inertia.
    KMeansPlusPlus,
}

/// Fitted-KMeans output.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct KMeansResult {
    /// Final centroids, one per row.
    pub centroids: Array2<f64>,
    /// Per-sample cluster label in `[0, k)`.
    pub labels: Array1<usize>,
    /// Final inertia: `Σ_i ‖xᵢ − μ_{c(i)}‖²`.
    pub inertia: f64,
    /// Number of Lloyd iterations of the best restart.
    pub n_iter: usize,
    /// Whether the best restart converged within `max_iter`.
    pub converged: bool,
}

/// KMeans clusterer.
#[derive(Clone, Debug)]
pub struct KMeans {
    /// Number of clusters.
    pub k: usize,
    /// Initialisation scheme.
    pub init: KMeansInit,
    /// Number of restarts; the best inertia wins. Default 10.
    pub n_init: usize,
    /// Maximum Lloyd iterations per restart. Default 300.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum centroid shift. Default 1e-4.
    pub tol: f64,
    /// Seed for the PRNG.
    pub seed: u64,
}

impl KMeans {
    /// New KMeans with sensible defaults and the given `k` and `seed`.
    pub fn new(k: usize, seed: u64) -> Self {
        Self {
            k,
            init: KMeansInit::KMeansPlusPlus,
            n_init: 10,
            max_iter: 300,
            tol: 1e-4,
            seed,
        }
    }

    /// Override the number of restarts.
    pub fn n_init(mut self, n: usize) -> Self {
        self.n_init = n;
        self
    }

    /// Override the maximum Lloyd iterations.
    pub fn max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }

    /// Override the convergence tolerance.
    pub fn tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }

    /// Override the initialisation scheme.
    pub fn init(mut self, i: KMeansInit) -> Self {
        self.init = i;
        self
    }

    /// Fit onto `x` and return the best restart.
    pub fn fit(&self, x: ArrayView2<'_, f64>) -> Result<KMeansResult> {
        let (n, d) = (x.nrows(), x.ncols());
        if n == 0 || d == 0 {
            return Err(Error::Value(
                "KMeans::fit: x must have at least one row and one column".into(),
            ));
        }
        if self.k == 0 || self.k > n {
            return Err(Error::Value(format!(
                "KMeans::fit: k must be in [1, n] (got k={}, n={n})",
                self.k
            )));
        }
        let mut best: Option<KMeansResult> = None;
        for restart in 0..self.n_init.max(1) {
            let sub_seed = self
                .seed
                .wrapping_add((restart as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut centroids = initial_centroids(x, self.k, self.init, sub_seed)?;
            let mut labels = Array1::<usize>::zeros(n);
            let mut converged = false;
            let mut iter_count = 0usize;
            for it in 0..self.max_iter {
                iter_count = it + 1;
                // Assignment step.
                assign(x, centroids.view(), &mut labels);
                // Update step.
                let new_centroids = update(x, &labels, self.k, centroids.view())?;
                // Convergence check on max centre shift.
                let mut max_shift = 0.0_f64;
                for r in 0..self.k {
                    let mut s = 0.0_f64;
                    for j in 0..d {
                        let dd = new_centroids[[r, j]] - centroids[[r, j]];
                        s += dd * dd;
                    }
                    let s = s.sqrt();
                    if s > max_shift {
                        max_shift = s;
                    }
                }
                centroids = new_centroids;
                if max_shift <= self.tol {
                    converged = true;
                    break;
                }
            }
            // Final assignment (labels correspond to the last centroids).
            assign(x, centroids.view(), &mut labels);
            let inertia = compute_inertia(x, centroids.view(), &labels);
            let candidate = KMeansResult {
                centroids,
                labels,
                inertia,
                n_iter: iter_count,
                converged,
            };
            match &best {
                None => best = Some(candidate),
                Some(b) if candidate.inertia < b.inertia => best = Some(candidate),
                _ => {}
            }
        }
        Ok(best.unwrap())
    }

    /// Fit and return the labels only.
    pub fn fit_predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<usize>> {
        Ok(self.fit(x)?.labels)
    }
}

// ---------------------------------------------------------------------------
// Assignment / update / inertia
// ---------------------------------------------------------------------------

fn assign(x: ArrayView2<'_, f64>, centroids: ArrayView2<'_, f64>, labels: &mut Array1<usize>) {
    for i in 0..x.nrows() {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for c in 0..centroids.nrows() {
            let mut dd = 0.0_f64;
            for j in 0..x.ncols() {
                let diff = x[[i, j]] - centroids[[c, j]];
                dd += diff * diff;
            }
            if dd < best_d {
                best_d = dd;
                best = c;
            }
        }
        labels[i] = best;
    }
}

fn update(
    x: ArrayView2<'_, f64>,
    labels: &Array1<usize>,
    k: usize,
    prev: ArrayView2<'_, f64>,
) -> Result<Array2<f64>> {
    let d = x.ncols();
    let mut sums = Array2::<f64>::zeros((k, d));
    let mut counts = vec![0usize; k];
    for i in 0..x.nrows() {
        let c = labels[i];
        counts[c] += 1;
        for j in 0..d {
            sums[[c, j]] += x[[i, j]];
        }
    }
    let mut out = Array2::<f64>::zeros((k, d));
    for c in 0..k {
        if counts[c] == 0 {
            // Empty cluster — keep the previous centroid (matches the reference
            // "reassign the farthest sample" alternative is more complex;
            // for our use case, an empty cluster is a signal that k is too
            // large or n_init should be raised).
            for j in 0..d {
                out[[c, j]] = prev[[c, j]];
            }
        } else {
            for j in 0..d {
                out[[c, j]] = sums[[c, j]] / counts[c] as f64;
            }
        }
    }
    Ok(out)
}

fn compute_inertia(
    x: ArrayView2<'_, f64>,
    centroids: ArrayView2<'_, f64>,
    labels: &Array1<usize>,
) -> f64 {
    let mut total = 0.0_f64;
    for i in 0..x.nrows() {
        let c = labels[i];
        for j in 0..x.ncols() {
            let diff = x[[i, j]] - centroids[[c, j]];
            total += diff * diff;
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

fn initial_centroids(
    x: ArrayView2<'_, f64>,
    k: usize,
    init: KMeansInit,
    seed: u64,
) -> Result<Array2<f64>> {
    match init {
        KMeansInit::Random => random_init(x, k, seed),
        KMeansInit::KMeansPlusPlus => kpp_init(x, k, seed),
    }
}

fn random_init(x: ArrayView2<'_, f64>, k: usize, seed: u64) -> Result<Array2<f64>> {
    let n = x.nrows();
    let mut state = seed.wrapping_add(0xA0B1_C2D3_E4F5_0617);
    let mut chosen = std::collections::HashSet::<usize>::new();
    while chosen.len() < k {
        chosen.insert(uniform_index(&mut state, n as u64));
    }
    let mut centroids = Array2::<f64>::zeros((k, x.ncols()));
    for (r, &i) in chosen.iter().enumerate() {
        for j in 0..x.ncols() {
            centroids[[r, j]] = x[[i, j]];
        }
    }
    Ok(centroids)
}

fn kpp_init(x: ArrayView2<'_, f64>, k: usize, seed: u64) -> Result<Array2<f64>> {
    let (n, d) = (x.nrows(), x.ncols());
    let mut state = seed.wrapping_add(0xBEEF_D06E_1234_5678);
    let mut centroids = Array2::<f64>::zeros((k, d));
    // First centre — uniform random row.
    let first = uniform_index(&mut state, n as u64);
    for j in 0..d {
        centroids[[0, j]] = x[[first, j]];
    }
    // Nearest-so-far D² for each sample.
    let mut d2 = vec![f64::INFINITY; n];
    for i in 0..n {
        let mut dd = 0.0_f64;
        for j in 0..d {
            let diff = x[[i, j]] - centroids[[0, j]];
            dd += diff * diff;
        }
        d2[i] = dd;
    }
    for c in 1..k {
        let total: f64 = d2.iter().sum();
        if total <= 0.0 {
            // Duplicate points — fall back to a uniform pick.
            let pick = uniform_index(&mut state, n as u64);
            for j in 0..d {
                centroids[[c, j]] = x[[pick, j]];
            }
        } else {
            let target = uniform_f64(&mut state) * total;
            let mut acc = 0.0_f64;
            let mut pick = 0usize;
            for (i, &dd) in d2.iter().enumerate() {
                acc += dd;
                if acc >= target {
                    pick = i;
                    break;
                }
            }
            for j in 0..d {
                centroids[[c, j]] = x[[pick, j]];
            }
        }
        // Update nearest-so-far distances.
        for i in 0..n {
            let mut dd = 0.0_f64;
            for j in 0..d {
                let diff = x[[i, j]] - centroids[[c, j]];
                dd += diff * diff;
            }
            if dd < d2[i] {
                d2[i] = dd;
            }
        }
    }
    Ok(centroids)
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
    (lcg_next(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn recovers_two_well_separated_clusters() {
        // Two blobs at (0, 0) and (10, 10).
        let mut rows: Vec<[f64; 2]> = Vec::new();
        for i in 0..30 {
            rows.push([(i as f64 * 0.01).sin(), (i as f64 * 0.017).cos()]);
        }
        for i in 0..30 {
            rows.push([
                10.0 + (i as f64 * 0.03).sin(),
                10.0 + (i as f64 * 0.019).cos(),
            ]);
        }
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        let x = Array2::from_shape_vec((60, 2), flat).unwrap();
        let km = KMeans::new(2, 42);
        let r = km.fit(x.view()).unwrap();
        // Every sample in [0..30) should share a label; every one in [30..60) shares another.
        let a = r.labels[0];
        let b = r.labels[30];
        assert_ne!(a, b);
        for i in 0..30 {
            assert_eq!(r.labels[i], a);
        }
        for i in 30..60 {
            assert_eq!(r.labels[i], b);
        }
        assert!(r.inertia < 5.0);
    }

    #[test]
    fn is_deterministic_by_seed() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [5.0, 6.0],
            [5.2, 6.1],
            [4.9, 5.8]
        ];
        let a = KMeans::new(2, 7).fit(x.view()).unwrap();
        let b = KMeans::new(2, 7).fit(x.view()).unwrap();
        assert_eq!(a.labels, b.labels);
        for (u, v) in a.centroids.iter().zip(b.centroids.iter()) {
            assert_abs_diff_eq!(u, v, epsilon = 1e-12);
        }
    }

    #[test]
    fn rejects_impossible_k() {
        let x = array![[1.0], [2.0]];
        assert!(KMeans::new(5, 0).fit(x.view()).is_err());
    }
}

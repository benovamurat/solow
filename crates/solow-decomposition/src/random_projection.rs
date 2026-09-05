//! `GaussianRandomProjection` and `SparseRandomProjection` — the reference
//! Johnson-Lindenstrauss lemma-inspired dimensionality reducers.
//!
//! Both project `X ∈ ℝ^{n × d}` to `X · Rᵀ ∈ ℝ^{n × k}` where `R` is a
//! random `(k × d)` matrix. Distances between pairs of rows are preserved
//! up to a `(1 ± ε)` distortion with high probability when
//! `k = Θ(log(n)/ε²)`.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// GaussianRandomProjection — entries of `R` are `𝒩(0, 1/k)`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianRandomProjection {
    /// Projection matrix `R ∈ ℝ^{k × d}`.
    pub components: Array2<f64>,
    /// Kept rank.
    pub n_components: usize,
    /// Input dimension.
    pub n_features_in: usize,
    /// Seed used at fit.
    pub seed: u64,
}

impl GaussianRandomProjection {
    /// Fit with the reference `johnson_lindenstrauss_min_dim(n, eps = 0.1)`.
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize, seed: u64) -> Result<Self> {
        let d = x.ncols();
        if n_components == 0 {
            return Err(Error::Value("GaussianRandomProjection: n_components must be ≥ 1".into()));
        }
        let mut state = seed.wrapping_add(0xF00D_C0DE);
        let mut r = Array2::<f64>::zeros((n_components, d));
        let scale = (1.0 / n_components as f64).sqrt();
        for i in 0..n_components {
            for j in 0..d {
                r[[i, j]] = scale * standard_normal(&mut state);
            }
        }
        Ok(Self {
            components: r,
            n_components,
            n_features_in: d,
            seed,
        })
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features_in {
            return Err(Error::Shape("GaussianRandomProjection::transform: shape mismatch".into()));
        }
        let n = x.nrows();
        let k = self.n_components;
        let d = self.n_features_in;
        let mut out = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for c in 0..k {
                let mut s = 0.0_f64;
                for j in 0..d {
                    s += x[[i, j]] * self.components[[c, j]];
                }
                out[[i, c]] = s;
            }
        }
        Ok(out)
    }
}

/// SparseRandomProjection — Achlioptas (2003). Entries take one of
/// `{-√(s/k), 0, +√(s/k)}` with probabilities `{1/2s, 1 − 1/s, 1/2s}`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SparseRandomProjection {
    /// Projection matrix `R`.
    pub components: Array2<f64>,
    /// Kept rank.
    pub n_components: usize,
    /// Input dimension.
    pub n_features_in: usize,
    /// Density `1/s` (fraction of non-zero entries).
    pub density: f64,
    /// Seed used at fit.
    pub seed: u64,
}

impl SparseRandomProjection {
    /// Fit with `density = 1/sqrt(n_features)` (the reference default).
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize, seed: u64) -> Result<Self> {
        let d = x.ncols();
        let density = (1.0_f64 / (d as f64).sqrt()).max(1.0 / d as f64);
        Self::fit_with(x, n_components, density, seed)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_components: usize,
        density: f64,
        seed: u64,
    ) -> Result<Self> {
        let d = x.ncols();
        if n_components == 0 {
            return Err(Error::Value("SparseRandomProjection: n_components must be ≥ 1".into()));
        }
        if !(0.0..=1.0).contains(&density) || density == 0.0 {
            return Err(Error::Value(format!(
                "SparseRandomProjection: density must be in (0, 1] (got {density})"
            )));
        }
        let s = 1.0 / density;
        let scale = (s / n_components as f64).sqrt();
        let mut state = seed.wrapping_add(0xF00D_D00D);
        let mut r = Array2::<f64>::zeros((n_components, d));
        for i in 0..n_components {
            for j in 0..d {
                let u = uniform01(&mut state);
                r[[i, j]] = if u < 1.0 / (2.0 * s) {
                    -scale
                } else if u < 1.0 / s {
                    scale
                } else {
                    0.0
                };
            }
        }
        Ok(Self {
            components: r,
            n_components,
            n_features_in: d,
            density,
            seed,
        })
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features_in {
            return Err(Error::Shape("SparseRandomProjection::transform: shape mismatch".into()));
        }
        let n = x.nrows();
        let k = self.n_components;
        let d = self.n_features_in;
        let mut out = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for c in 0..k {
                let mut s = 0.0_f64;
                for j in 0..d {
                    s += x[[i, j]] * self.components[[c, j]];
                }
                out[[i, c]] = s;
            }
        }
        Ok(out)
    }
}

/// Return the minimum `n_components` such that JL preserves distances
/// up to `1 ± eps` for `n_samples` points.
pub fn johnson_lindenstrauss_min_dim(n_samples: usize, eps: f64) -> usize {
    if eps <= 0.0 || eps >= 1.0 {
        return 0;
    }
    let denom = eps * eps / 2.0 - eps * eps * eps / 3.0;
    ((4.0 * (n_samples as f64).ln() / denom).ceil() as usize).max(1)
}

fn standard_normal(state: &mut u64) -> f64 {
    let u1 = uniform01(state).max(1e-12);
    let u2 = uniform01(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn uniform01(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let r = *state >> 11;
    (r as f64) * f64::from_bits(0x3CA0_0000_0000_0000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn gaussian_random_projection_is_deterministic_at_a_seed() {
        let x = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let a = GaussianRandomProjection::fit(x.view(), 5, 42).unwrap();
        let b = GaussianRandomProjection::fit(x.view(), 5, 42).unwrap();
        for i in 0..5 {
            for j in 0..3 {
                assert_eq!(a.components[[i, j]], b.components[[i, j]]);
            }
        }
    }

    #[test]
    fn sparse_random_projection_produces_a_projection_with_the_expected_shape() {
        let x = array![
            [1.0_f64, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]
        ];
        let m = SparseRandomProjection::fit(x.view(), 3, 7).unwrap();
        let z = m.transform(x.view()).unwrap();
        assert_eq!(z.shape(), &[3, 3]);
    }

    #[test]
    fn jl_min_dim_grows_logarithmically_with_n_samples() {
        let a = johnson_lindenstrauss_min_dim(100, 0.1);
        let b = johnson_lindenstrauss_min_dim(10_000, 0.1);
        // Log-scaling → b/a should be roughly log(100).
        assert!(b > a);
        assert!(b < 10 * a);
    }
}

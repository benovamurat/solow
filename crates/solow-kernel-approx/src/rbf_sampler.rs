//! Random Fourier Features for the RBF kernel (Rahimi-Recht 2007).
//!
//! The RBF kernel `k(x, y) = exp(−γ ‖x − y‖²)` has spectral density
//! `p(ω) = 𝒩(0, 2γ I)`. Sampling `ω_i ∼ p` and `b_i ∼ 𝒰[0, 2π]` gives
//! the explicit feature map
//!
//! ```text
//!     z(x) = sqrt(2/D) · cos(Wᵀx + b) ∈ ℝᴰ
//! ```
//!
//! whose inner product is an unbiased estimator of `k`.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::rng::Lcg;

/// Fitted RBFSampler.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct RBFSampler {
    /// Random-frequency matrix `W ∈ ℝ^{p × D}`.
    pub random_weights: Array2<f64>,
    /// Random offsets `b ∈ ℝᴰ`.
    pub random_offset: Array1<f64>,
    /// Kernel width `γ`.
    pub gamma: f64,
    /// Feature dimension.
    pub n_components: usize,
    /// Input dimension.
    pub n_features: usize,
    /// Seed used at fit.
    pub seed: u64,
}

impl RBFSampler {
    /// Fit with `gamma = 1/n_features`, `n_components = 100`, `seed = 0`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        let g = 1.0 / (x.ncols() as f64).max(1.0);
        Self::fit_with(x, g, 100, 0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        gamma: f64,
        n_components: usize,
        seed: u64,
    ) -> Result<Self> {
        if x.ncols() == 0 {
            return Err(Error::Value("RBFSampler::fit_with: x has 0 columns".into()));
        }
        if !gamma.is_finite() || gamma <= 0.0 {
            return Err(Error::Value(format!(
                "RBFSampler::fit_with: gamma must be > 0 (got {gamma})"
            )));
        }
        if n_components == 0 {
            return Err(Error::Value(
                "RBFSampler::fit_with: n_components must be ≥ 1".into(),
            ));
        }
        let p = x.ncols();
        let sigma = (2.0 * gamma).sqrt();
        let mut rng = Lcg::new(seed);
        let mut w = Array2::<f64>::zeros((p, n_components));
        for j in 0..n_components {
            for i in 0..p {
                w[[i, j]] = sigma * rng.standard_normal();
            }
        }
        let mut b = Array1::<f64>::zeros(n_components);
        for j in 0..n_components {
            b[j] = 2.0 * std::f64::consts::PI * rng.uniform01();
        }
        Ok(Self {
            random_weights: w,
            random_offset: b,
            gamma,
            n_components,
            n_features: p,
            seed,
        })
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(Error::Shape(format!(
                "RBFSampler::transform: expected {} cols, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let d = self.n_components;
        let scale = (2.0 / d as f64).sqrt();
        let mut out = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                let mut s = self.random_offset[j];
                for k in 0..self.n_features {
                    s += x[[i, k]] * self.random_weights[[k, j]];
                }
                out[[i, j]] = scale * s.cos();
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn rbf_sampler_deterministic_and_correct_shape() {
        let x = array![[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
        let m1 = RBFSampler::fit_with(x.view(), 0.5, 8, 42).unwrap();
        let m2 = RBFSampler::fit_with(x.view(), 0.5, 8, 42).unwrap();
        let t1 = m1.transform(x.view()).unwrap();
        let t2 = m2.transform(x.view()).unwrap();
        assert_eq!(t1.shape(), &[3, 8]);
        for i in 0..3 {
            for j in 0..8 {
                assert_eq!(t1[[i, j]], t2[[i, j]]);
            }
        }
    }

    #[test]
    fn rbf_sampler_approx_the_kernel_at_high_dimension() {
        // Two points at unit distance. k(x, y) = exp(-γ · 1) with γ = 0.5 → 0.6065.
        let x = array![[0.0, 0.0], [1.0, 0.0]];
        let m = RBFSampler::fit_with(x.view(), 0.5, 4096, 123).unwrap();
        let z = m.transform(x.view()).unwrap();
        let mut est = 0.0_f64;
        for j in 0..4096 {
            est += z[[0, j]] * z[[1, j]];
        }
        // MC error at D = 4096 should easily hit within 0.05 of the true value.
        let true_k = (-0.5_f64).exp();
        assert!(
            (est - true_k).abs() < 0.05,
            "estimate = {est}, true = {true_k}"
        );
    }
}

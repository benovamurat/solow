//! Random skewed-χ² feature map for strictly positive inputs.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::rng::Lcg;

/// SkewedChi2Sampler.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SkewedChi2Sampler {
    /// Random-frequency matrix `W ∈ ℝ^{p × D}` drawn from a sech-π
    /// density in log-space.
    pub random_weights: Array2<f64>,
    /// Uniform offsets `b ∈ [0, 2π)ᴰ`.
    pub random_offset: Array1<f64>,
    /// Positive skewness parameter `c` (added inside the log).
    pub skewedness: f64,
    /// Number of output components.
    pub n_components: usize,
    /// Input features count.
    pub n_features: usize,
    /// Seed used at fit.
    pub seed: u64,
}

impl SkewedChi2Sampler {
    /// Fit with the reference defaults `skewedness = 1.0`, `n_components = 100`, `seed = 0`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 1.0, 100, 0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        skewedness: f64,
        n_components: usize,
        seed: u64,
    ) -> Result<Self> {
        if skewedness <= 0.0 || !skewedness.is_finite() {
            return Err(Error::Value(
                "SkewedChi2Sampler: skewedness must be > 0".into(),
            ));
        }
        if n_components == 0 {
            return Err(Error::Value(
                "SkewedChi2Sampler: n_components must be ≥ 1".into(),
            ));
        }
        let p = x.ncols();
        let mut rng = Lcg::new(seed);
        // Draw ω_i from f(ω) = sech(π ω) / 2 via the inverse-CDF
        //     u ∼ 𝒰(0, 1)  →  ω = (1/π)·ln(tan((π/2)·u))
        let mut w = Array2::<f64>::zeros((p, n_components));
        for j in 0..n_components {
            for i in 0..p {
                let u = rng.uniform01().clamp(1e-10, 1.0 - 1e-10);
                w[[i, j]] = (std::f64::consts::FRAC_1_PI)
                    * (std::f64::consts::FRAC_PI_2 * u).tan().ln();
            }
        }
        let mut b = Array1::<f64>::zeros(n_components);
        for j in 0..n_components {
            b[j] = 2.0 * std::f64::consts::PI * rng.uniform01();
        }
        Ok(Self {
            random_weights: w,
            random_offset: b,
            skewedness,
            n_components,
            n_features: p,
            seed,
        })
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(Error::Shape("SkewedChi2Sampler::transform: shape mismatch".into()));
        }
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                if x[[i, j]] + self.skewedness <= 0.0 {
                    return Err(Error::Value(
                        "SkewedChi2Sampler: x + skewedness must be > 0".into(),
                    ));
                }
            }
        }
        let n = x.nrows();
        let d = self.n_components;
        let p = self.n_features;
        let scale = (2.0 / d as f64).sqrt();
        let mut out = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                let mut acc = self.random_offset[j];
                for k in 0..p {
                    let l = (x[[i, k]] + self.skewedness).ln();
                    acc += l * self.random_weights[[k, j]];
                }
                out[[i, j]] = scale * acc.cos();
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
    fn skewed_chi2_is_deterministic_at_a_seed() {
        let x = array![[0.5, 1.0], [1.5, 2.0]];
        let m1 = SkewedChi2Sampler::fit_with(x.view(), 1.0, 8, 5).unwrap();
        let m2 = SkewedChi2Sampler::fit_with(x.view(), 1.0, 8, 5).unwrap();
        let t1 = m1.transform(x.view()).unwrap();
        let t2 = m2.transform(x.view()).unwrap();
        for i in 0..2 {
            for j in 0..8 {
                assert_eq!(t1[[i, j]], t2[[i, j]]);
            }
        }
    }
}

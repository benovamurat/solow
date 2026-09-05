//! PolynomialCountSketch — TensorSketch feature map (Pham-Pagh 2013).
//!
//! Approximates the polynomial kernel `k(x, y) = (γ·x·y + coef0)^degree`
//! via `degree` independent Count-Sketch projections combined with
//! FFT-free element-wise multiplication.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::rng::Lcg;

/// PolynomialCountSketch.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PolynomialCountSketch {
    /// Hash-index table `h ∈ ℕ^{degree × p}`.
    pub hash_indices: Vec<Vec<usize>>,
    /// Sign table `s ∈ {−1, +1}^{degree × p}`.
    pub signs: Vec<Vec<f64>>,
    /// Number of output components.
    pub n_components: usize,
    /// Input features.
    pub n_features: usize,
    /// Kernel degree.
    pub degree: usize,
    /// Kernel scale `γ`.
    pub gamma: f64,
    /// Kernel bias `coef0` (folded into features by prepending a constant `1`).
    pub coef0: f64,
    /// Seed used at fit.
    pub seed: u64,
}

impl PolynomialCountSketch {
    /// Fit with defaults `gamma = 1`, `degree = 2`, `coef0 = 0`, `100` components.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 1.0, 2, 0.0, 100, 0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        gamma: f64,
        degree: usize,
        coef0: f64,
        n_components: usize,
        seed: u64,
    ) -> Result<Self> {
        if degree == 0 {
            return Err(Error::Value(
                "PolynomialCountSketch: degree must be ≥ 1".into(),
            ));
        }
        if n_components == 0 {
            return Err(Error::Value(
                "PolynomialCountSketch: n_components must be ≥ 1".into(),
            ));
        }
        if gamma <= 0.0 {
            return Err(Error::Value(
                "PolynomialCountSketch: gamma must be > 0".into(),
            ));
        }
        // Effective input dimension includes the +1 constant if coef0 != 0.
        let p = x.ncols() + if coef0 != 0.0 { 1 } else { 0 };
        let mut rng = Lcg::new(seed);
        let mut hash = vec![vec![0_usize; p]; degree];
        let mut sgn = vec![vec![0.0_f64; p]; degree];
        for d in 0..degree {
            for i in 0..p {
                hash[d][i] = rng.uniform_index(n_components);
                sgn[d][i] = rng.rademacher();
            }
        }
        Ok(Self {
            hash_indices: hash,
            signs: sgn,
            n_components,
            n_features: x.ncols(),
            degree,
            gamma,
            coef0,
            seed,
        })
    }

    /// Transform via TensorSketch.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(Error::Shape(
                "PolynomialCountSketch::transform: shape mismatch".into(),
            ));
        }
        let n = x.nrows();
        let d = self.n_components;
        let deg = self.degree;
        let p_eff = self.n_features + if self.coef0 != 0.0 { 1 } else { 0 };
        let g = self.gamma.sqrt();
        let mut out = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            // Sketch each of `degree` independent Count-Sketches.
            let mut sketches: Vec<Vec<f64>> = vec![vec![0.0_f64; d]; deg];
            for kd in 0..deg {
                for j in 0..p_eff {
                    let xij = if j < self.n_features {
                        g * x[[i, j]]
                    } else {
                        self.coef0.abs().sqrt()
                    };
                    let bin = self.hash_indices[kd][j];
                    sketches[kd][bin] += self.signs[kd][j] * xij;
                }
            }
            // Combine by pointwise multiplication.
            for b in 0..d {
                let mut prod = sketches[0][b];
                for kd in 1..deg {
                    prod *= sketches[kd][b];
                }
                out[[i, b]] = prod;
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
    fn polynomial_sketch_deterministic_at_a_seed() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let m1 = PolynomialCountSketch::fit_with(x.view(), 1.0, 2, 0.0, 32, 1).unwrap();
        let m2 = PolynomialCountSketch::fit_with(x.view(), 1.0, 2, 0.0, 32, 1).unwrap();
        let a = m1.transform(x.view()).unwrap();
        let b = m2.transform(x.view()).unwrap();
        for i in 0..3 {
            for j in 0..32 {
                assert_eq!(a[[i, j]], b[[i, j]]);
            }
        }
    }
}

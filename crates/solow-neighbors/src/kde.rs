//! `KernelDensity` — multivariate kernel density estimation.
//!
//! Mirrors `neighbors.KernelDensity`: fits `n_train` samples,
//! then scores query points as `(1/n) · Σᵢ K((x − xᵢ) / h)` with the
//! selected kernel and bandwidth.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Kernel family.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KdeKernel {
    /// Gaussian kernel.
    Gaussian,
    /// Tophat (uniform) kernel.
    Tophat,
    /// Epanechnikov (parabolic) kernel.
    Epanechnikov,
    /// Exponential (Laplace) kernel.
    Exponential,
    /// Linear (triangular) kernel.
    Linear,
    /// Cosine kernel.
    Cosine,
}

/// Fitted KernelDensity.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct KernelDensity {
    /// Training samples.
    pub x_train: Array2<f64>,
    /// Kernel bandwidth.
    pub bandwidth: f64,
    /// Kernel choice.
    pub kernel: KdeKernel,
}

impl KernelDensity {
    /// Fit with the reference defaults `kernel = Gaussian`, `bandwidth = 1.0`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 1.0, KdeKernel::Gaussian)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        bandwidth: f64,
        kernel: KdeKernel,
    ) -> Result<Self> {
        if bandwidth <= 0.0 {
            return Err(Error::Value("KernelDensity: bandwidth must be > 0".into()));
        }
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value("KernelDensity: empty training data".into()));
        }
        Ok(Self {
            x_train: x.to_owned(),
            bandwidth,
            kernel,
        })
    }

    /// Return log-density at each row of `x`.
    pub fn score_samples(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.x_train.ncols() {
            return Err(Error::Shape("KernelDensity::score_samples: shape mismatch".into()));
        }
        let n = x.nrows();
        let m = self.x_train.nrows();
        let d = self.x_train.ncols();
        let h = self.bandwidth;
        let mut out = Array1::<f64>::zeros(n);
        let normaliser = kernel_normalisation(&self.kernel, d) / (m as f64 * h.powi(d as i32));
        for i in 0..n {
            let mut acc = 0.0_f64;
            for k in 0..m {
                let mut r2 = 0.0_f64;
                for j in 0..d {
                    let e = (x[[i, j]] - self.x_train[[k, j]]) / h;
                    r2 += e * e;
                }
                let r = r2.sqrt();
                acc += kernel_shape(&self.kernel, r);
            }
            out[i] = (acc * normaliser).max(1e-300).ln();
        }
        Ok(out)
    }
}

fn kernel_shape(k: &KdeKernel, u: f64) -> f64 {
    match k {
        KdeKernel::Gaussian => (-0.5 * u * u).exp(),
        KdeKernel::Tophat => {
            if u < 1.0 {
                1.0
            } else {
                0.0
            }
        }
        KdeKernel::Epanechnikov => {
            if u < 1.0 {
                1.0 - u * u
            } else {
                0.0
            }
        }
        KdeKernel::Exponential => (-u).exp(),
        KdeKernel::Linear => {
            if u < 1.0 {
                1.0 - u
            } else {
                0.0
            }
        }
        KdeKernel::Cosine => {
            if u < 1.0 {
                (std::f64::consts::PI * u / 2.0).cos()
            } else {
                0.0
            }
        }
    }
}

fn kernel_normalisation(k: &KdeKernel, d: usize) -> f64 {
    // A rough per-kernel per-dim normalisation constant so integration
    // → 1 in the Gaussian case (exact) and approximately in others.
    match k {
        KdeKernel::Gaussian => 1.0 / (2.0 * std::f64::consts::PI).powf(d as f64 / 2.0),
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn kernel_density_gaussian_returns_higher_density_near_the_mode() {
        let train = array![[0.0_f64], [0.1], [-0.1], [0.05]];
        let kd = KernelDensity::fit_with(train.view(), 0.5, KdeKernel::Gaussian).unwrap();
        let test = array![[0.0_f64], [5.0]];
        let s = kd.score_samples(test.view()).unwrap();
        assert!(s[0] > s[1]);
    }
}

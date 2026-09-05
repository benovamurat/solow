//! # solow-kernel-approx
//!
//! Kernel-approximation feature maps: the `kernel_approximation`
//! family. These transform a dataset into an explicit feature space where
//! the inner product approximates the kernel, so downstream linear
//! learners (SGD, Ridge, LinearSVC) behave as kernelised methods without
//! ever materialising the `n × n` kernel matrix.
//!
//! * [`RBFSampler`] — Rahimi-Recht (2007) random Fourier features for
//!   the RBF kernel `k(x, y) = exp(−γ ‖x − y‖²)`.
//! * [`Nystroem`] — Williams-Seeger (2001) landmark sampling for
//!   arbitrary positive-definite kernels.
//! * [`AdditiveChi2Sampler`] — Vedaldi-Zisserman (2011) exact additive-
//!   χ² feature map.
//! * [`SkewedChi2Sampler`] — random skewed-χ² sampler for
//!   histogram-shaped positive features.
//! * [`PolynomialCountSketch`] — Kar-Karnick (2012) TensorSketch for the
//!   polynomial kernel.
//!
//! # References
//!
//! * Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale
//!   Kernel Machines.* NIPS 20.
//! * Williams, C. K. I., & Seeger, M. (2001). *Using the Nyström Method
//!   to Speed Up Kernel Machines.* NIPS 13.
//! * Vedaldi, A., & Zisserman, A. (2011). *Efficient Additive Kernels
//!   via Explicit Feature Maps.* IEEE T-PAMI 34(3), 480-492.
//! * Kar, P., & Karnick, H. (2012). *Random Feature Maps for Dot Product
//!   Kernels.* AISTATS.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod additive_chi2;
pub mod nystroem;
pub mod polynomial_sketch;
pub mod rbf_sampler;
pub mod skewed_chi2;

pub use additive_chi2::AdditiveChi2Sampler;
pub use nystroem::Nystroem;
pub use polynomial_sketch::PolynomialCountSketch;
pub use rbf_sampler::RBFSampler;
pub use skewed_chi2::SkewedChi2Sampler;

/// Commonly-used imports.
pub mod prelude {
    pub use crate::additive_chi2::AdditiveChi2Sampler;
    pub use crate::nystroem::Nystroem;
    pub use crate::polynomial_sketch::PolynomialCountSketch;
    pub use crate::rbf_sampler::RBFSampler;
    pub use crate::skewed_chi2::SkewedChi2Sampler;
}

pub(crate) mod rng {
    //! Portable deterministic PRNG shared across samplers.

    pub struct Lcg {
        state: u64,
    }

    impl Lcg {
        pub fn new(seed: u64) -> Self {
            Self {
                state: seed.wrapping_add(0xCAFE_BABE_F00D_D00D),
            }
        }

        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.state
        }

        pub fn uniform01(&mut self) -> f64 {
            // 53-bit uniform in [0, 1). 2^-53 in IEEE 754 = 0x3CA0…
            let r = self.next() >> 11;
            (r as f64) * f64::from_bits(0x3CA0_0000_0000_0000)
        }

        pub fn standard_normal(&mut self) -> f64 {
            // Box-Muller with rejection guard.
            loop {
                let u1 = self.uniform01();
                if u1 > 1e-300 {
                    let u2 = self.uniform01();
                    let r = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * std::f64::consts::PI * u2;
                    return r * theta.cos();
                }
            }
        }

        pub fn uniform_index(&mut self, n: usize) -> usize {
            let n = n as u64;
            let max = u64::MAX - (u64::MAX % n);
            loop {
                let r = self.next();
                if r < max {
                    return (r % n) as usize;
                }
            }
        }

        pub fn rademacher(&mut self) -> f64 {
            if self.next() & 1 == 0 {
                -1.0
            } else {
                1.0
            }
        }
    }
}

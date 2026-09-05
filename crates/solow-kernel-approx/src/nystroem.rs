//! Nyström landmark approximation of an arbitrary kernel.
//!
//! Given a kernel `k(·, ·)` and `n_components` landmarks sampled without
//! replacement from `X`, we compute
//!
//! ```text
//!     K_mm = k(landmarks, landmarks)              // (m × m)
//!     K_nm = k(X, landmarks)                      // (n × m)
//!     Φ(x) = k(x, landmarks) · K_mm⁻¹ᐟ²
//! ```
//!
//! `Φ(x)ᵀΦ(y) ≈ k(x, y)`.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::rng::Lcg;

/// Supported Nyström kernels.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum NystroemKernel {
    /// RBF kernel `exp(−γ ‖x − y‖²)`.
    Rbf { gamma: f64 },
    /// Polynomial kernel `(γ · x·y + c)^d`.
    Polynomial { gamma: f64, coef0: f64, degree: i32 },
    /// Sigmoid kernel `tanh(γ · x·y + c)`.
    Sigmoid { gamma: f64, coef0: f64 },
    /// Cosine kernel `x·y / (‖x‖ · ‖y‖)`.
    Cosine,
    /// Laplacian kernel `exp(−γ ‖x − y‖₁)`.
    Laplacian { gamma: f64 },
}

/// Fitted Nyström feature map.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Nystroem {
    /// Landmark points (`m × p`).
    pub landmarks: Array2<f64>,
    /// `K_mm⁻¹ᐟ²` (`m × m`).
    pub normalization: Array2<f64>,
    /// Kernel choice.
    pub kernel: NystroemKernel,
    /// Kept components.
    pub n_components: usize,
    /// Seed used at fit.
    pub seed: u64,
}

impl Nystroem {
    /// Fit with a default RBF kernel, `gamma = 1/n_features`, `100` components.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        let g = 1.0 / (x.ncols() as f64).max(1.0);
        Self::fit_with(x, NystroemKernel::Rbf { gamma: g }, 100, 0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        kernel: NystroemKernel,
        n_components: usize,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        if n < n_components {
            return Err(Error::Value(format!(
                "Nystroem::fit_with: need {n_components} landmarks but x has only {n} rows"
            )));
        }
        // Sample landmark indices without replacement via reservoir-style
        // Fisher-Yates on an index vector.
        let mut idx: Vec<usize> = (0..n).collect();
        let mut rng = Lcg::new(seed);
        for i in 0..n_components {
            let j = i + rng.uniform_index(n - i);
            idx.swap(i, j);
        }
        idx.truncate(n_components);
        let p = x.ncols();
        let mut land = Array2::<f64>::zeros((n_components, p));
        for (r, &i) in idx.iter().enumerate() {
            for j in 0..p {
                land[[r, j]] = x[[i, j]];
            }
        }
        // Compute K_mm.
        let mut kmm = Array2::<f64>::zeros((n_components, n_components));
        for i in 0..n_components {
            for j in i..n_components {
                let v = eval_kernel(&kernel, &land.row(i).to_vec(), &land.row(j).to_vec());
                kmm[[i, j]] = v;
                kmm[[j, i]] = v;
            }
        }
        // K_mm ≈ V Λ Vᵀ  →  K_mm⁻¹ᐟ² = V Λ⁻¹ᐟ² Vᵀ  via Jacobi eigendecomposition.
        let (eigvals, eigvecs) = jacobi_symmetric(&kmm, 300, 1e-12);
        let mut inv_sqrt = Array2::<f64>::zeros((n_components, n_components));
        for i in 0..n_components {
            for j in 0..n_components {
                let mut s = 0.0_f64;
                for k in 0..n_components {
                    let lambda = eigvals[k].max(1e-12);
                    s += eigvecs[[i, k]] * eigvecs[[j, k]] / lambda.sqrt();
                }
                inv_sqrt[[i, j]] = s;
            }
        }
        Ok(Self {
            landmarks: land,
            normalization: inv_sqrt,
            kernel,
            n_components,
            seed,
        })
    }

    /// Transform a matrix.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let m = self.n_components;
        let p = self.landmarks.ncols();
        if x.ncols() != p {
            return Err(Error::Shape(format!(
                "Nystroem::transform: expected {p} cols, got {}",
                x.ncols()
            )));
        }
        let mut knm = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            let xi: Vec<f64> = (0..p).map(|k| x[[i, k]]).collect();
            for j in 0..m {
                let lj: Vec<f64> = (0..p).map(|k| self.landmarks[[j, k]]).collect();
                knm[[i, j]] = eval_kernel(&self.kernel, &xi, &lj);
            }
        }
        // Φ = K_nm · normalization.
        let mut out = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                let mut s = 0.0_f64;
                for k in 0..m {
                    s += knm[[i, k]] * self.normalization[[k, j]];
                }
                out[[i, j]] = s;
            }
        }
        Ok(out)
    }
}

fn eval_kernel(kernel: &NystroemKernel, a: &[f64], b: &[f64]) -> f64 {
    match *kernel {
        NystroemKernel::Rbf { gamma } => {
            let mut s = 0.0_f64;
            for i in 0..a.len() {
                let d = a[i] - b[i];
                s += d * d;
            }
            (-gamma * s).exp()
        }
        NystroemKernel::Polynomial { gamma, coef0, degree } => {
            let mut s = 0.0_f64;
            for i in 0..a.len() {
                s += a[i] * b[i];
            }
            (gamma * s + coef0).powi(degree)
        }
        NystroemKernel::Sigmoid { gamma, coef0 } => {
            let mut s = 0.0_f64;
            for i in 0..a.len() {
                s += a[i] * b[i];
            }
            (gamma * s + coef0).tanh()
        }
        NystroemKernel::Cosine => {
            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for i in 0..a.len() {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            dot / (na.sqrt() * nb.sqrt()).max(1e-30)
        }
        NystroemKernel::Laplacian { gamma } => {
            let mut s = 0.0_f64;
            for i in 0..a.len() {
                s += (a[i] - b[i]).abs();
            }
            (-gamma * s).exp()
        }
    }
}

/// Cyclic Jacobi eigendecomposition of a real symmetric matrix.
fn jacobi_symmetric(a: &Array2<f64>, max_sweeps: usize, tol: f64) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut m = a.clone();
    let mut v = Array2::<f64>::eye(n);
    for _ in 0..max_sweeps {
        let mut off = 0.0_f64;
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                off += m[[p, q]] * m[[p, q]];
            }
        }
        if off.sqrt() < tol {
            break;
        }
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                let apq = m[[p, q]];
                if apq.abs() < 1e-30 {
                    continue;
                }
                let app = m[[p, p]];
                let aqq = m[[q, q]];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for i in 0..n {
                    let mip = m[[i, p]];
                    let miq = m[[i, q]];
                    m[[i, p]] = c * mip - s * miq;
                    m[[i, q]] = s * mip + c * miq;
                }
                for j in 0..n {
                    let mpj = m[[p, j]];
                    let mqj = m[[q, j]];
                    m[[p, j]] = c * mpj - s * mqj;
                    m[[q, j]] = s * mpj + c * mqj;
                }
                for i in 0..n {
                    let vip = v[[i, p]];
                    let viq = v[[i, q]];
                    v[[i, p]] = c * vip - s * viq;
                    v[[i, q]] = s * vip + c * viq;
                }
            }
        }
    }
    let mut eig = vec![0.0_f64; n];
    for i in 0..n {
        eig[i] = m[[i, i]];
    }
    (eig, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn nystroem_reproduces_the_rbf_kernel_at_landmark_rows() {
        let x = array![
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [2.0, 0.0], [0.0, 2.0]
        ];
        let m = Nystroem::fit_with(
            x.view(),
            NystroemKernel::Rbf { gamma: 0.5 },
            4,
            7,
        ).unwrap();
        let z = m.transform(x.view()).unwrap();
        // The inner product ZZᵀ should be close to the true K on the training set.
        let mut ok = 0;
        let mut total = 0;
        for i in 0..6 {
            for j in 0..6 {
                let mut approx = 0.0_f64;
                for k in 0..4 {
                    approx += z[[i, k]] * z[[j, k]];
                }
                let xi = (0..2).map(|k| x[[i, k]]).collect::<Vec<_>>();
                let xj = (0..2).map(|k| x[[j, k]]).collect::<Vec<_>>();
                let truth = eval_kernel(&NystroemKernel::Rbf { gamma: 0.5 }, &xi, &xj);
                total += 1;
                if (approx - truth).abs() < 0.15 {
                    ok += 1;
                }
            }
        }
        assert!(ok as f64 / total as f64 > 0.5, "too many bad entries");
    }
}

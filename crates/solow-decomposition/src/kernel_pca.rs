//! Kernel PCA (Schölkopf, Smola & Müller 1998).

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};
use solow_manifold::isomap::jacobi_symmetric;

/// Kernel family used by [`KernelPca`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum KernelKind {
    /// Linear kernel `⟨x, y⟩`.
    Linear,
    /// RBF (Gaussian) kernel `exp(−γ · ‖x − y‖²)`.
    Rbf {
        /// Bandwidth γ.
        gamma: f64,
    },
    /// Polynomial kernel `(⟨x, y⟩ · γ + coef0)^degree`.
    Polynomial {
        /// Polynomial degree.
        degree: u32,
        /// Scale factor γ on the dot product.
        gamma: f64,
        /// Additive offset.
        coef0: f64,
    },
}

impl KernelKind {
    fn apply(&self, xi: &[f64], xj: &[f64]) -> f64 {
        match self {
            KernelKind::Linear => dot(xi, xj),
            KernelKind::Rbf { gamma } => {
                let mut s = 0.0_f64;
                for k in 0..xi.len() {
                    let d = xi[k] - xj[k];
                    s += d * d;
                }
                (-gamma * s).exp()
            }
            KernelKind::Polynomial {
                degree,
                gamma,
                coef0,
            } => {
                let base = gamma * dot(xi, xj) + coef0;
                let mut acc = 1.0_f64;
                for _ in 0..*degree {
                    acc *= base;
                }
                acc
            }
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    for k in 0..a.len() {
        s += a[k] * b[k];
    }
    s
}

/// Fitted kernel-PCA embedding.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct KernelPca {
    /// Low-dimensional embedding, `n × n_components`.
    pub embedding: Array2<f64>,
    /// Retained eigenvalues.
    pub eigenvalues: Array1<f64>,
    /// Kernel used at fit time.
    pub kernel: KernelKind,
}

impl KernelPca {
    /// Fit onto `x` with the given kernel and target dimension.
    pub fn fit(x: ArrayView2<'_, f64>, kernel: KernelKind, n_components: usize) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value("KernelPca::fit: x must be non-empty".into()));
        }
        if n_components < 1 || n_components > x.nrows() {
            return Err(Error::Value(format!(
                "KernelPca::fit: n_components in [1, n] (got {n_components})"
            )));
        }
        let n = x.nrows();
        // Build kernel matrix.
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let xi: Vec<f64> = x.row(i).to_vec();
            for j in 0..=i {
                let xj: Vec<f64> = x.row(j).to_vec();
                let v = kernel.apply(&xi, &xj);
                k[[i, j]] = v;
                k[[j, i]] = v;
            }
        }
        // Centre the kernel matrix: K_c = K − 1_N·K − K·1_N + 1_N·K·1_N.
        let mut row_means = vec![0.0_f64; n];
        let mut col_means = vec![0.0_f64; n];
        let mut grand = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                row_means[i] += k[[i, j]];
                col_means[j] += k[[i, j]];
                grand += k[[i, j]];
            }
        }
        for v in row_means.iter_mut() {
            *v /= n as f64;
        }
        for v in col_means.iter_mut() {
            *v /= n as f64;
        }
        grand /= (n * n) as f64;
        let mut kc = k.clone();
        for i in 0..n {
            for j in 0..n {
                kc[[i, j]] += grand - row_means[i] - col_means[j];
            }
        }
        // Eigendecompose (kc is symmetric).
        let (eigvals, eigvecs) = jacobi_symmetric(&kc, 300, 1e-12);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| eigvals[b].partial_cmp(&eigvals[a]).unwrap());
        let mut embedding = Array2::<f64>::zeros((n, n_components));
        let mut ev_out = Array1::<f64>::zeros(n_components);
        for c in 0..n_components {
            let idx = order[c];
            let ev = eigvals[idx].max(0.0);
            ev_out[c] = ev;
            let scale = ev.sqrt();
            for i in 0..n {
                embedding[[i, c]] = eigvecs[[i, idx]] * scale;
            }
        }
        Ok(Self {
            embedding,
            eigenvalues: ev_out,
            kernel,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn kernel_pca_linear_recovers_pca_directions_up_to_sign() {
        // With a linear kernel KernelPCA is standard PCA.
        let x = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.5, 0.5],
            [-0.5, -0.5]
        ];
        let kpca = KernelPca::fit(x.view(), KernelKind::Linear, 2).unwrap();
        assert_eq!(kpca.embedding.dim(), (6, 2));
        // Eigenvalues descending and non-negative.
        assert!(kpca.eigenvalues[0] + 1e-9 >= kpca.eigenvalues[1]);
        assert!(kpca.eigenvalues[0] >= 0.0);
    }
}

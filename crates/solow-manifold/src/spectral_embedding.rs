//! SpectralEmbedding — Belkin-Niyogi (2003) Laplacian eigenmaps.
//!
//! Builds an RBF-affinity graph, computes the normalised graph
//! Laplacian, and takes its bottom-`k+1` eigenvectors (skipping the
//! trivial first one).

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted SpectralEmbedding.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SpectralEmbedding {
    /// Low-dim embedding `(n × k)`.
    pub embedding: Array2<f64>,
    /// Kept rank.
    pub n_components: usize,
    /// Affinity kernel γ.
    pub gamma: f64,
}

impl SpectralEmbedding {
    /// Fit with `gamma = 1.0`.
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize) -> Result<Self> {
        Self::fit_with(x, n_components, 1.0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_components: usize,
        gamma: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if n_components == 0 || n_components + 1 > n {
            return Err(Error::Value(format!(
                "SpectralEmbedding: n_components must be in [1, {}] (got {n_components})",
                n - 1
            )));
        }
        let mut w = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0_f64;
                for k in 0..d {
                    let e = x[[i, k]] - x[[j, k]];
                    s += e * e;
                }
                w[[i, j]] = (-gamma * s).exp();
            }
        }
        // Normalised Laplacian L = I − D⁻¹ᐟ² W D⁻¹ᐟ².
        let mut deg = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                deg[i] += w[[i, j]];
            }
        }
        let mut lap = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let di = deg[i].sqrt().max(1e-30);
                let dj = deg[j].sqrt().max(1e-30);
                lap[[i, j]] = -w[[i, j]] / (di * dj) + if i == j { 1.0 } else { 0.0 };
            }
        }
        let (eig, vecs) = jacobi_symmetric(&lap, 400, 1e-12);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &c| eig[a].partial_cmp(&eig[c]).unwrap());
        // Skip the first (trivial) eigenvector.
        let mut embed = Array2::<f64>::zeros((n, n_components));
        for (k, &orig) in idx.iter().skip(1).take(n_components).enumerate() {
            for i in 0..n {
                embed[[i, k]] = vecs[[i, orig]];
            }
        }
        Ok(Self {
            embedding: embed,
            n_components,
            gamma,
        })
    }
}

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
                let theta = (m[[q, q]] - m[[p, p]]) / (2.0 * apq);
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

// Prevent unused-import warning.
#[allow(dead_code)]
fn _touch(a: Array1<f64>) -> Array1<f64> {
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn spectral_embedding_gives_the_requested_dimension() {
        let x = array![
            [0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
        ];
        let m = SpectralEmbedding::fit(x.view(), 2).unwrap();
        assert_eq!(m.embedding.shape(), &[4, 2]);
    }
}

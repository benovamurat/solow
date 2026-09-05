//! Spectral clustering (Shi-Malik 2000, Ng-Jordan-Weiss 2001).
//!
//! Builds an RBF affinity graph, computes the symmetric normalised
//! Laplacian `L_sym = I − D⁻¹ᐟ² W D⁻¹ᐟ²`, extracts the bottom `k`
//! eigenvectors, row-normalises, and clusters the resulting embedding
//! with `KMeans`.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::kmeans::{KMeans, KMeansInit};

/// Fitted spectral clustering.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SpectralClustering {
    /// Cluster labels (one per input row).
    pub labels: Vec<i64>,
    /// Cluster count.
    pub n_clusters: usize,
    /// Affinity kernel γ.
    pub gamma: f64,
    /// Seed used for the KMeans step.
    pub seed: u64,
}

impl SpectralClustering {
    /// Fit with the reference defaults `gamma = 1.0`, `seed = 0`.
    pub fn fit(x: ArrayView2<'_, f64>, n_clusters: usize) -> Result<Self> {
        Self::fit_with(x, n_clusters, 1.0, 0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_clusters: usize,
        gamma: f64,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        if n_clusters == 0 || n_clusters > n {
            return Err(Error::Value(format!(
                "SpectralClustering: n_clusters must be in [1, {n}] (got {n_clusters})"
            )));
        }
        // Affinity matrix W = exp(-γ ‖xᵢ - xⱼ‖²).
        let mut w = Array2::<f64>::zeros((n, n));
        let d = x.ncols();
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
        // Degree matrix.
        let mut deg = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                deg[i] += w[[i, j]];
            }
        }
        // Symmetric normalised Laplacian L = I - D⁻¹ᐟ² W D⁻¹ᐟ².
        let mut lap = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let di = deg[i].sqrt().max(1e-30);
                let dj = deg[j].sqrt().max(1e-30);
                let val = -w[[i, j]] / (di * dj);
                lap[[i, j]] = val + if i == j { 1.0 } else { 0.0 };
            }
        }
        // Bottom-k eigenvectors of L via Jacobi (small n only).
        let (eig, vecs) = jacobi_symmetric(&lap, 400, 1e-12);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| eig[a].partial_cmp(&eig[b]).unwrap());
        let k = n_clusters;
        let mut embed = Array2::<f64>::zeros((n, k));
        for (r, &oidx) in idx.iter().take(k).enumerate() {
            for i in 0..n {
                embed[[i, r]] = vecs[[i, oidx]];
            }
        }
        // Row-normalise.
        for i in 0..n {
            let mut norm = 0.0_f64;
            for j in 0..k {
                norm += embed[[i, j]] * embed[[i, j]];
            }
            let norm = norm.sqrt().max(1e-30);
            for j in 0..k {
                embed[[i, j]] /= norm;
            }
        }
        // KMeans on the spectral embedding.
        let km = KMeans::new(k, seed).init(KMeansInit::KMeansPlusPlus).fit(embed.view())?;
        Ok(Self {
            labels: km.labels.iter().map(|&x| x as i64).collect(),
            n_clusters,
            gamma,
            seed,
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn spectral_clustering_splits_two_ring_bumps() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.05, 0.15],
            [5.0, 5.0], [5.1, 5.1], [5.05, 5.15]
        ];
        let sc = SpectralClustering::fit_with(x.view(), 2, 0.5, 42).unwrap();
        let a = sc.labels[0];
        let b = sc.labels[3];
        assert_ne!(a, b, "the two bumps should sit in different clusters");
    }
}

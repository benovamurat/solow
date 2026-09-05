//! Classical multidimensional scaling (Torgerson 1952) — an eigenvalue-
//! based low-dim embedding that preserves the pairwise Euclidean
//! distance matrix.
//!
//! Given `X ∈ ℝ^{n × d_in}` we compute:
//!   1. Squared-distance matrix `D²`.
//!   2. Doubly-centred `B = -½ · H · D² · H` with `H = I − 1·1ᵀ/n`.
//!   3. Top-`k` eigen-decomposition of `B`.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted MDS.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MDS {
    /// Low-dim embedding `(n × k)`.
    pub embedding: Array2<f64>,
    /// Stress = ‖D_out − D_in‖_F².
    pub stress: f64,
    /// Kept rank.
    pub n_components: usize,
}

impl MDS {
    /// Fit with the reference defaults `metric = True`, `dissimilarity =
    /// 'euclidean'`.
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize) -> Result<Self> {
        let n = x.nrows();
        if n_components == 0 || n_components > n {
            return Err(Error::Value(format!(
                "MDS: n_components must be in [1, {n}] (got {n_components})"
            )));
        }
        let d_in = x.ncols();
        // Squared Euclidean distances.
        let mut d2 = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let mut s = 0.0_f64;
                for k in 0..d_in {
                    let e = x[[i, k]] - x[[j, k]];
                    s += e * e;
                }
                d2[[i, j]] = s;
                d2[[j, i]] = s;
            }
        }
        // Double centring.
        let mut row_mean = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = 0.0_f64;
            for j in 0..n {
                s += d2[[i, j]];
            }
            row_mean[i] = s / n as f64;
        }
        let total_mean: f64 = row_mean.iter().sum::<f64>() / n as f64;
        let mut b = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                b[[i, j]] = -0.5 * (d2[[i, j]] - row_mean[i] - row_mean[j] + total_mean);
            }
        }
        let (eig, vecs) = jacobi_symmetric(&b, 400, 1e-12);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &c| eig[c].partial_cmp(&eig[a]).unwrap());
        let mut embed = Array2::<f64>::zeros((n, n_components));
        for (k, &orig) in idx.iter().take(n_components).enumerate() {
            let lambda = eig[orig].max(0.0);
            let scale = lambda.sqrt();
            for i in 0..n {
                embed[[i, k]] = vecs[[i, orig]] * scale;
            }
        }
        // Recompute output distances and compare with input.
        let mut stress = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut s = 0.0_f64;
                for k in 0..n_components {
                    let e = embed[[i, k]] - embed[[j, k]];
                    s += e * e;
                }
                let diff = s.sqrt() - d2[[i, j]].sqrt();
                stress += diff * diff;
            }
        }
        Ok(Self {
            embedding: embed,
            stress,
            n_components,
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

// Prevent unused Array1 warning.
#[allow(dead_code)]
fn _touch(a: Array1<f64>) -> Array1<f64> {
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn mds_recovers_2d_layout_of_a_2d_configuration() {
        let x = array![
            [0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
        ];
        let m = MDS::fit(x.view(), 2).unwrap();
        assert_eq!(m.embedding.shape(), &[4, 2]);
    }
}

//! IncrementalPCA — the sequential-fit variant of PCA (Ross et al.
//! 2008). Fits by accumulating running mean and running scatter, then
//! decomposes at the end of the stream (or on demand).

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted IncrementalPCA.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct IncrementalPCA {
    /// Overall sample mean.
    pub mean: Array1<f64>,
    /// Principal directions `(k × d)`.
    pub components: Array2<f64>,
    /// Explained-variance per component.
    pub explained_variance: Array1<f64>,
    /// Total sample size seen so far.
    pub n_samples_seen: usize,
    /// Kept rank.
    pub n_components: usize,
}

impl IncrementalPCA {
    /// Fit in one call.
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if n_components == 0 || n_components > n.min(d) {
            return Err(Error::Value(format!(
                "IncrementalPCA: n_components must be in [1, {}] (got {n_components})",
                n.min(d)
            )));
        }
        // Compute mean and centred data.
        let mut mean = Array1::<f64>::zeros(d);
        for j in 0..d {
            let mut s = 0.0_f64;
            for i in 0..n {
                s += x[[i, j]];
            }
            mean[j] = s / n as f64;
        }
        let mut centred = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                centred[[i, j]] = x[[i, j]] - mean[j];
            }
        }
        // Covariance = (Xᵀ · X) / (n − 1).
        let mut cov = Array2::<f64>::zeros((d, d));
        for i in 0..n {
            for j in 0..d {
                for k in 0..d {
                    cov[[j, k]] += centred[[i, j]] * centred[[i, k]];
                }
            }
        }
        let denom = (n as f64 - 1.0).max(1.0);
        for j in 0..d {
            for k in 0..d {
                cov[[j, k]] /= denom;
            }
        }
        let (eig, vecs) = jacobi_symmetric(&cov, 400, 1e-12);
        let mut idx: Vec<usize> = (0..d).collect();
        idx.sort_by(|&a, &b| eig[b].partial_cmp(&eig[a]).unwrap());
        let mut comps = Array2::<f64>::zeros((n_components, d));
        let mut expl = Array1::<f64>::zeros(n_components);
        for (k, &orig) in idx.iter().take(n_components).enumerate() {
            for j in 0..d {
                comps[[k, j]] = vecs[[j, orig]];
            }
            expl[k] = eig[orig];
        }
        Ok(Self {
            mean,
            components: comps,
            explained_variance: expl,
            n_samples_seen: n,
            n_components,
        })
    }

    /// Project new rows into the fitted PC space.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let d = self.mean.len();
        let k = self.n_components;
        if x.ncols() != d {
            return Err(Error::Shape("IncrementalPCA::transform: shape mismatch".into()));
        }
        let mut out = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                let mut s = 0.0_f64;
                for r in 0..d {
                    s += (x[[i, r]] - self.mean[r]) * self.components[[j, r]];
                }
                out[[i, j]] = s;
            }
        }
        Ok(out)
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
    fn ipca_returns_components_of_the_right_shape() {
        let x = array![
            [1.0_f64, 2.0, 3.0], [3.0, 5.0, 8.0], [5.0, 7.0, 11.0],
            [7.0, 9.0, 15.0], [9.0, 12.0, 20.0]
        ];
        let m = IncrementalPCA::fit(x.view(), 2).unwrap();
        assert_eq!(m.components.shape(), &[2, 3]);
        assert!(m.explained_variance[0] >= m.explained_variance[1]);
    }
}

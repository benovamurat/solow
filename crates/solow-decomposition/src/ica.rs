//! FastICA (Hyvärinen 1999) with symmetric decorrelation.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};
use solow_manifold::isomap::jacobi_symmetric;

/// Non-Gaussianity contrast function.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IcaFun {
    /// `G(u) = (1/α) log cosh(α u)` (default, α = 1).
    LogCosh,
    /// `G(u) = -exp(-u²/2)`.
    Exp,
}

impl IcaFun {
    fn g_and_gprime(&self, u: f64) -> (f64, f64) {
        match self {
            IcaFun::LogCosh => {
                let t = u.tanh();
                (t, 1.0 - t * t)
            }
            IcaFun::Exp => {
                let e = (-0.5 * u * u).exp();
                (u * e, (1.0 - u * u) * e)
            }
        }
    }
}

/// Fitted FastICA decomposition.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct FastIca {
    /// Un-mixing matrix `W` of shape `(n_components, d)` (applied to
    /// the centered data).
    pub components: Array2<f64>,
    /// Whitening matrix `K` of shape `(n_components, d)`.
    pub whitening: Array2<f64>,
    /// Data mean subtracted before whitening.
    pub mean: Array1<f64>,
    /// Number of components.
    pub n_components: usize,
    /// Number of iterations run.
    pub n_iter: usize,
}

impl FastIca {
    /// Fit with defaults (`fun = LogCosh`, `max_iter = 200`, `tol = 1e-4`).
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize, seed: u64) -> Result<Self> {
        Self::fit_with(x, n_components, IcaFun::LogCosh, 200, 1e-4, seed)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_components: usize,
        fun: IcaFun,
        max_iter: usize,
        tol: f64,
        seed: u64,
    ) -> Result<Self> {
        if x.nrows() < 2 || x.ncols() == 0 {
            return Err(Error::Value(
                "FastIca::fit_with: need ≥ 2 samples and ≥ 1 feature".into(),
            ));
        }
        if n_components == 0 || n_components > x.ncols() {
            return Err(Error::Value(format!(
                "FastIca::fit_with: n_components in [1, d] (got {n_components})"
            )));
        }
        let (n, d) = (x.nrows(), x.ncols());
        // Centre.
        let mut mean = Array1::<f64>::zeros(d);
        for j in 0..d {
            mean[j] = x.column(j).iter().sum::<f64>() / n as f64;
        }
        let mut xc = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                xc[[i, j]] = x[[i, j]] - mean[j];
            }
        }
        // Whitening: eigendecompose XᵀX/n.
        let mut cov = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0_f64;
                for k in 0..n {
                    s += xc[[k, i]] * xc[[k, j]];
                }
                cov[[i, j]] = s / n as f64;
            }
        }
        let (eigvals, eigvecs) = jacobi_symmetric(&cov, 300, 1e-12);
        let mut order: Vec<usize> = (0..d).collect();
        order.sort_by(|&a, &b| eigvals[b].partial_cmp(&eigvals[a]).unwrap());
        // Build whitening K of shape (n_components, d): rows = eigvecs[:, i] / sqrt(λ_i).
        let mut k = Array2::<f64>::zeros((n_components, d));
        for c in 0..n_components {
            let idx = order[c];
            let s = eigvals[idx].max(1e-12).sqrt();
            for j in 0..d {
                k[[c, j]] = eigvecs[[j, idx]] / s;
            }
        }
        // Whitened data X1 = xc · Kᵀ  (shape n × n_components).
        let mut x1 = Array2::<f64>::zeros((n, n_components));
        for i in 0..n {
            for c in 0..n_components {
                let mut s = 0.0_f64;
                for j in 0..d {
                    s += xc[[i, j]] * k[[c, j]];
                }
                x1[[i, c]] = s;
            }
        }
        // Initialise W (n_components × n_components) with a deterministic random matrix.
        let mut state = seed.wrapping_add(0xABCD_EF01_2345_6789);
        let mut w = Array2::<f64>::zeros((n_components, n_components));
        for i in 0..n_components {
            for j in 0..n_components {
                w[[i, j]] = uniform_symmetric(&mut state, 1.0);
            }
        }
        symmetric_decorrelate(&mut w);
        // FastICA symmetric loop.
        let mut n_iter_used = 0usize;
        for it in 0..max_iter {
            n_iter_used = it + 1;
            let mut new_w = Array2::<f64>::zeros(w.dim());
            for c in 0..n_components {
                let (mut mean_g, mut mean_gp) = (0.0_f64, 0.0_f64);
                let mut acc = Array1::<f64>::zeros(n_components);
                for i in 0..n {
                    let mut u = 0.0_f64;
                    for kk in 0..n_components {
                        u += w[[c, kk]] * x1[[i, kk]];
                    }
                    let (g, gp) = fun.g_and_gprime(u);
                    mean_g += g;
                    mean_gp += gp;
                    for kk in 0..n_components {
                        acc[kk] += x1[[i, kk]] * g;
                    }
                }
                mean_g /= n as f64;
                mean_gp /= n as f64;
                for kk in 0..n_components {
                    new_w[[c, kk]] = acc[kk] / n as f64 - mean_gp * w[[c, kk]];
                }
                let _ = mean_g;
            }
            symmetric_decorrelate(&mut new_w);
            // Convergence.
            let mut delta = 0.0_f64;
            for c in 0..n_components {
                let mut s = 0.0_f64;
                for kk in 0..n_components {
                    s += new_w[[c, kk]] * w[[c, kk]];
                }
                delta = delta.max((s.abs() - 1.0).abs());
            }
            w = new_w;
            if delta < tol {
                break;
            }
        }
        Ok(Self {
            components: w,
            whitening: k,
            mean,
            n_components,
            n_iter: n_iter_used,
        })
    }

    /// Project `x` into the independent-component space.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let (n, d) = (x.nrows(), x.ncols());
        // (x - mean) · Kᵀ · Wᵀ
        let mut out = Array2::<f64>::zeros((n, self.n_components));
        for i in 0..n {
            // Centre.
            let mut centered = vec![0.0_f64; d];
            for j in 0..d {
                centered[j] = x[[i, j]] - self.mean[j];
            }
            // Whiten.
            let mut w1 = vec![0.0_f64; self.n_components];
            for c in 0..self.n_components {
                let mut s = 0.0_f64;
                for j in 0..d {
                    s += centered[j] * self.whitening[[c, j]];
                }
                w1[c] = s;
            }
            // Un-mix.
            for c in 0..self.n_components {
                let mut s = 0.0_f64;
                for kk in 0..self.n_components {
                    s += self.components[[c, kk]] * w1[kk];
                }
                out[[i, c]] = s;
            }
        }
        out
    }
}

fn symmetric_decorrelate(w: &mut Array2<f64>) {
    let n = w.nrows();
    // A = W · Wᵀ
    let mut a = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += w[[i, k]] * w[[j, k]];
            }
            a[[i, j]] = s;
        }
    }
    let (eigvals, eigvecs) = jacobi_symmetric(&a, 300, 1e-12);
    // A^{-1/2} = V · diag(1/√λ) · Vᵀ.
    let mut d_half = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let inv = if eigvals[i] > 1e-12 {
            1.0 / eigvals[i].sqrt()
        } else {
            0.0
        };
        d_half[[i, i]] = inv;
    }
    // vt = eigvecs.T · W, then result = eigvecs · d_half · vt
    // Efficient: (eigvecs · d_half · eigvecsᵀ) · W
    let mut tmp = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += eigvecs[[i, k]] * d_half[[k, k]] * eigvecs[[j, k]];
            }
            tmp[[i, j]] = s;
        }
    }
    // W_new = tmp · W.
    let mut new_w = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += tmp[[i, k]] * w[[k, j]];
            }
            new_w[[i, j]] = s;
        }
    }
    *w = new_w;
}

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn uniform_f64(state: &mut u64) -> f64 {
    (lcg_next(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

fn uniform_symmetric(state: &mut u64, scale: f64) -> f64 {
    (uniform_f64(state) - 0.5) * 2.0 * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fast_ica_runs_on_small_input() {
        // Mixed two sources; ICA should return finite output.
        let x = array![
            [1.0, 0.5],
            [0.2, 0.9],
            [-1.0, -0.4],
            [0.3, -0.6],
            [-0.7, 0.1],
            [0.6, -0.9]
        ];
        let ica = FastIca::fit_with(x.view(), 2, IcaFun::LogCosh, 100, 1e-4, 7).unwrap();
        let s = ica.transform(x.view());
        assert_eq!(s.dim(), (6, 2));
        for v in s.iter() {
            assert!(v.is_finite());
        }
    }
}

//! MiniBatchNMF — online non-negative matrix factorisation via
//! multiplicative-update mini-batches (Févotte-Idier 2011).

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted MiniBatchNMF.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MiniBatchNmf {
    /// Basis matrix `W` (`n × n_components`) at fit time.
    pub w: Array2<f64>,
    /// Loadings `H` (`n_components × d`).
    pub h: Array2<f64>,
    /// Reconstruction error (Frobenius).
    pub reconstruction_error: f64,
    /// Kept rank.
    pub n_components: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Iterations run.
    pub n_iter: usize,
}

impl MiniBatchNmf {
    /// Fit with the reference defaults `batch_size = 1024`, `max_iter = 100`, `tol = 1e-4`.
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize) -> Result<Self> {
        Self::fit_with(x, n_components, 1024, 100, 1e-4, 0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_components: usize,
        batch_size: usize,
        max_iter: usize,
        tol: f64,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if n_components == 0 || n_components > n.min(d) {
            return Err(Error::Value("MiniBatchNMF: n_components out of range".into()));
        }
        if x.iter().any(|&v| v < 0.0) {
            return Err(Error::Value("MiniBatchNMF: inputs must be non-negative".into()));
        }
        // Initialise W, H uniformly random in [0, 1) with a small offset.
        let mut state = seed.wrapping_add(0xF00D_C0DE);
        let mut w = Array2::<f64>::zeros((n, n_components));
        for i in 0..n {
            for k in 0..n_components {
                w[[i, k]] = 0.5 + 0.5 * uniform01(&mut state);
            }
        }
        let mut h = Array2::<f64>::zeros((n_components, d));
        for k in 0..n_components {
            for j in 0..d {
                h[[k, j]] = 0.5 + 0.5 * uniform01(&mut state);
            }
        }
        let batch = batch_size.min(n).max(1);
        let mut iters = 0_usize;
        let mut prev_err = f64::INFINITY;
        for it in 0..max_iter {
            iters = it + 1;
            let mut start = 0_usize;
            while start < n {
                let end = (start + batch).min(n);
                // Multiplicative update — H first, then W.
                // H = H * (Wᵀ · X_batch) / (Wᵀ · W · H) — computed per column j.
                let bs = end - start;
                // Wt_batch = W[start..end, :].T (n_components × bs).
                // For H update we use only rows in [start, end):
                //   H = H .* (W_batchᵀ X_batch) ./ (W_batchᵀ W_batch H)
                let mut wt_x = Array2::<f64>::zeros((n_components, d));
                let mut wt_w = Array2::<f64>::zeros((n_components, n_components));
                for k in 0..n_components {
                    for j in 0..d {
                        let mut s = 0.0_f64;
                        for r in start..end {
                            s += w[[r, k]] * x[[r, j]];
                        }
                        wt_x[[k, j]] = s;
                    }
                    for k2 in 0..n_components {
                        let mut s = 0.0_f64;
                        for r in start..end {
                            s += w[[r, k]] * w[[r, k2]];
                        }
                        wt_w[[k, k2]] = s;
                    }
                }
                let wt_w_h = matmul_nn(&wt_w, &h);
                for k in 0..n_components {
                    for j in 0..d {
                        let denom = wt_w_h[[k, j]].max(1e-30);
                        h[[k, j]] *= wt_x[[k, j]] / denom;
                    }
                }
                // W_batch update: W = W .* (X Hᵀ) ./ (W H Hᵀ)
                let mut xh_t = Array2::<f64>::zeros((bs, n_components));
                let mut hh_t = Array2::<f64>::zeros((n_components, n_components));
                for k in 0..n_components {
                    for k2 in 0..n_components {
                        let mut s = 0.0_f64;
                        for j in 0..d {
                            s += h[[k, j]] * h[[k2, j]];
                        }
                        hh_t[[k, k2]] = s;
                    }
                }
                for rr in 0..bs {
                    for k in 0..n_components {
                        let mut s = 0.0_f64;
                        for j in 0..d {
                            s += x[[start + rr, j]] * h[[k, j]];
                        }
                        xh_t[[rr, k]] = s;
                    }
                }
                let mut w_hh_t = Array2::<f64>::zeros((bs, n_components));
                for rr in 0..bs {
                    for k in 0..n_components {
                        let mut s = 0.0_f64;
                        for k2 in 0..n_components {
                            s += w[[start + rr, k2]] * hh_t[[k2, k]];
                        }
                        w_hh_t[[rr, k]] = s;
                    }
                }
                for rr in 0..bs {
                    for k in 0..n_components {
                        let denom = w_hh_t[[rr, k]].max(1e-30);
                        w[[start + rr, k]] *= xh_t[[rr, k]] / denom;
                    }
                }
                start = end;
            }
            // Reconstruction error.
            let recon = matmul_nn(&w, &h);
            let mut err = 0.0_f64;
            for i in 0..n {
                for j in 0..d {
                    let e = x[[i, j]] - recon[[i, j]];
                    err += e * e;
                }
            }
            err = err.sqrt();
            if (prev_err - err).abs() < tol * prev_err.max(1e-30) {
                prev_err = err;
                break;
            }
            prev_err = err;
        }
        Ok(Self {
            w,
            h,
            reconstruction_error: prev_err,
            n_components,
            batch_size: batch,
            n_iter: iters,
        })
    }
}

fn matmul_nn(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let n = b.ncols();
    let inner = a.ncols();
    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f64;
            for r in 0..inner {
                s += a[[i, r]] * b[[r, j]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

fn uniform01(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let r = *state >> 11;
    (r as f64) * f64::from_bits(0x3CA0_0000_0000_0000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn minibatch_nmf_reduces_reconstruction_error() {
        let x = array![
            [1.0_f64, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0],
            [1.0, 0.0, 1.0], [0.0, 2.0, 2.0]
        ];
        let m = MiniBatchNmf::fit_with(x.view(), 2, 3, 100, 1e-4, 42).unwrap();
        assert_eq!(m.w.shape(), &[5, 2]);
        assert_eq!(m.h.shape(), &[2, 3]);
        assert!(m.reconstruction_error.is_finite());
    }
}

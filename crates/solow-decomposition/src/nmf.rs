//! Non-negative Matrix Factorisation with multiplicative updates
//! (Lee & Seung 2001) under the Frobenius objective.
//!
//! Approximates `X ≈ W · H` with `W ∈ ℝ⁺^{n × k}` and `H ∈ ℝ⁺^{k × d}`.
//! Multiplicative updates preserve non-negativity by construction.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn uniform_f64(state: &mut u64) -> f64 {
    (lcg_next(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Fitted NMF factors.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Nmf {
    /// `W` — the sample-side factor `(n × n_components)`.
    pub w: Array2<f64>,
    /// `H` — the component-side factor `(n_components × d)`.
    pub h: Array2<f64>,
    /// Final Frobenius reconstruction error `‖X − W·H‖_F`.
    pub reconstruction_err: f64,
    /// Number of iterations run.
    pub n_iter: usize,
}

impl Nmf {
    /// Fit with defaults (`max_iter = 200`, `tol = 1e-4`).
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize, seed: u64) -> Result<Self> {
        Self::fit_with(x, n_components, 200, 1e-4, seed)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_components: usize,
        max_iter: usize,
        tol: f64,
        seed: u64,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value("Nmf::fit_with: x must be non-empty".into()));
        }
        for &v in x.iter() {
            if v < 0.0 || !v.is_finite() {
                return Err(Error::Value(
                    "Nmf::fit_with: X must be non-negative and finite".into(),
                ));
            }
        }
        if n_components == 0 || n_components > x.nrows().min(x.ncols()) {
            return Err(Error::Value(format!(
                "Nmf::fit_with: n_components must be in [1, min(n, d)] (got {n_components})"
            )));
        }
        let (n, d) = (x.nrows(), x.ncols());
        let k = n_components;
        // Random non-negative init.
        let mut state = seed.wrapping_add(0x1122_3344_5566_7788);
        let mut w = Array2::<f64>::zeros((n, k));
        let mut h = Array2::<f64>::zeros((k, d));
        for i in 0..n {
            for j in 0..k {
                w[[i, j]] = uniform_f64(&mut state);
            }
        }
        for i in 0..k {
            for j in 0..d {
                h[[i, j]] = uniform_f64(&mut state);
            }
        }
        let mut prev_err = f64::INFINITY;
        let mut n_iter_used = 0usize;
        for it in 0..max_iter {
            n_iter_used = it + 1;
            // Update H: H ← H * (Wᵀ X) / (Wᵀ W H).
            let wt_x = matmul(&transpose(&w), &array_view_to_array(x));
            let wt_w = matmul(&transpose(&w), &w);
            let wt_w_h = matmul(&wt_w, &h);
            for i in 0..k {
                for j in 0..d {
                    let denom = wt_w_h[[i, j]] + 1e-12;
                    h[[i, j]] *= wt_x[[i, j]] / denom;
                }
            }
            // Update W: W ← W * (X Hᵀ) / (W H Hᵀ).
            let x_ht = matmul(&array_view_to_array(x), &transpose(&h));
            let h_ht = matmul(&h, &transpose(&h));
            let w_h_ht = matmul(&w, &h_ht);
            for i in 0..n {
                for j in 0..k {
                    let denom = w_h_ht[[i, j]] + 1e-12;
                    w[[i, j]] *= x_ht[[i, j]] / denom;
                }
            }
            // Frobenius error.
            let reconstr = matmul(&w, &h);
            let mut err = 0.0_f64;
            for i in 0..n {
                for j in 0..d {
                    let dd = x[[i, j]] - reconstr[[i, j]];
                    err += dd * dd;
                }
            }
            err = err.sqrt();
            if (prev_err - err).abs() < tol {
                return Ok(Self {
                    w,
                    h,
                    reconstruction_err: err,
                    n_iter: n_iter_used,
                });
            }
            prev_err = err;
        }
        // Final error.
        let reconstr = matmul(&w, &h);
        let mut err = 0.0_f64;
        for i in 0..n {
            for j in 0..d {
                let dd = x[[i, j]] - reconstr[[i, j]];
                err += dd * dd;
            }
        }
        err = err.sqrt();
        Ok(Self {
            w,
            h,
            reconstruction_err: err,
            n_iter: n_iter_used,
        })
    }
}

fn transpose(m: &Array2<f64>) -> Array2<f64> {
    let (r, c) = m.dim();
    let mut out = Array2::<f64>::zeros((c, r));
    for i in 0..r {
        for j in 0..c {
            out[[j, i]] = m[[i, j]];
        }
    }
    out
}

fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (r, mid) = a.dim();
    let (_, c) = b.dim();
    let mut out = Array2::<f64>::zeros((r, c));
    for i in 0..r {
        for k in 0..mid {
            let aik = a[[i, k]];
            if aik == 0.0 {
                continue;
            }
            for j in 0..c {
                out[[i, j]] += aik * b[[k, j]];
            }
        }
    }
    out
}

fn array_view_to_array(x: ArrayView2<'_, f64>) -> Array2<f64> {
    x.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn nmf_reduces_reconstruction_error() {
        // A structured non-negative matrix — NMF with k=2 should reach low error.
        let x = array![
            [5.0, 4.0, 0.0, 0.0],
            [4.0, 5.0, 0.0, 1.0],
            [0.0, 1.0, 5.0, 4.0],
            [0.0, 0.0, 4.0, 5.0],
        ];
        let nmf = Nmf::fit(x.view(), 2, 42).unwrap();
        // Reconstruction should account for most of X's Frobenius norm.
        let total: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            nmf.reconstruction_err < 0.5 * total,
            "err = {}",
            nmf.reconstruction_err
        );
    }
}

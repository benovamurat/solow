//! Canonical-mode PLS. Same NIPALS core as PLSRegression, but deflates
//! `Y` by its own `u`-scores instead of the X-scores.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::pls_regression::{
    center_scale, dot_vv, invert, matmul_nn, matmul_tn, nipals, to_scaled,
};

/// Fitted PLSCanonical model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PLSCanonical {
    /// X-mean at fit time.
    pub x_mean: Array1<f64>,
    /// X-std at fit time (or ones).
    pub x_std: Array1<f64>,
    /// Y-mean at fit time.
    pub y_mean: Array1<f64>,
    /// Y-std at fit time.
    pub y_std: Array1<f64>,
    /// X-weights (`p × k`).
    pub x_weights: Array2<f64>,
    /// Y-weights (`q × k`).
    pub y_weights: Array2<f64>,
    /// X-loadings (`p × k`).
    pub x_loadings: Array2<f64>,
    /// Y-loadings (`q × k`).
    pub y_loadings: Array2<f64>,
    /// X-rotations `W* = W(PᵀW)⁻¹`.
    pub x_rotations: Array2<f64>,
    /// Y-rotations `C* = C(QᵀC)⁻¹`.
    pub y_rotations: Array2<f64>,
    /// Number of latent components.
    pub n_components: usize,
    /// Whether the fit scaled columns to unit std.
    pub scale: bool,
}

impl PLSCanonical {
    /// Fit with defaults: `scale = true`, `max_iter = 500`, `tol = 1e-6`.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        n_components: usize,
    ) -> Result<Self> {
        Self::fit_with(x, y, n_components, true, 500, 1e-6)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        n_components: usize,
        scale: bool,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        if x.nrows() != y.nrows() {
            return Err(Error::Shape("PLSCanonical: row counts differ".into()));
        }
        let n = x.nrows();
        let p = x.ncols();
        let q = y.ncols();
        if n_components == 0 || n_components > p.min(q).min(n - 1) {
            return Err(Error::Value("PLSCanonical: n_components out of range".into()));
        }
        let (x_mean, x_std) = center_scale(x, scale);
        let (y_mean, y_std) = center_scale(y, scale);
        let mut xk = to_scaled(x, &x_mean, &x_std);
        let mut yk = to_scaled(y, &y_mean, &y_std);
        let mut w_mat = Array2::<f64>::zeros((p, n_components));
        let mut c_mat = Array2::<f64>::zeros((q, n_components));
        let mut p_mat = Array2::<f64>::zeros((p, n_components));
        let mut q_mat = Array2::<f64>::zeros((q, n_components));
        for k in 0..n_components {
            let (w, c, t, u, _iters) = nipals(&xk, &yk, max_iter, tol);
            let t_dot_t = dot_vv(&t, &t).max(1e-30);
            let u_dot_u = dot_vv(&u, &u).max(1e-30);
            let mut p_load = vec![0.0_f64; p];
            for j in 0..p {
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += xk[[i, j]] * t[i];
                }
                p_load[j] = s / t_dot_t;
            }
            let mut q_load = vec![0.0_f64; q];
            for j in 0..q {
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += yk[[i, j]] * u[i];
                }
                q_load[j] = s / u_dot_u;
            }
            // Canonical deflation: Y by its own u-scores.
            for i in 0..n {
                for j in 0..p {
                    xk[[i, j]] -= t[i] * p_load[j];
                }
                for j in 0..q {
                    yk[[i, j]] -= u[i] * q_load[j];
                }
            }
            for j in 0..p {
                w_mat[[j, k]] = w[j];
                p_mat[[j, k]] = p_load[j];
            }
            for j in 0..q {
                c_mat[[j, k]] = c[j];
                q_mat[[j, k]] = q_load[j];
            }
        }
        let ptw = matmul_tn(&p_mat, &w_mat);
        let ptw_inv = invert(&ptw)?;
        let x_rotations = matmul_nn(&w_mat, &ptw_inv);
        let qtc = matmul_tn(&q_mat, &c_mat);
        let y_rotations = match invert(&qtc) {
            Ok(inv) => matmul_nn(&c_mat, &inv),
            Err(_) => Array2::<f64>::zeros((q, n_components)),
        };
        Ok(Self {
            x_mean,
            x_std,
            y_mean,
            y_std,
            x_weights: w_mat,
            y_weights: c_mat,
            x_loadings: p_mat,
            y_loadings: q_mat,
            x_rotations,
            y_rotations,
            n_components,
            scale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn pls_canonical_returns_the_requested_rank() {
        let x = array![
            [1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 1.0, 2.0],
            [4.0, 2.0, 3.0], [5.0, 3.0, 4.0]
        ];
        let y = array![
            [1.0, 2.0], [2.0, 4.0], [3.0, 1.0], [4.0, 2.0], [5.0, 3.0]
        ];
        let m = PLSCanonical::fit(x.view(), y.view(), 2).unwrap();
        assert_eq!(m.x_weights.ncols(), 2);
        assert_eq!(m.y_weights.ncols(), 2);
    }
}

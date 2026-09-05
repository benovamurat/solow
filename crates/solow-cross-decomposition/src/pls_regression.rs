//! Non-linear-iterative partial-least-squares (NIPALS) PLSRegression.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted PLS regression model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PLSRegression {
    /// X-mean at fit time.
    pub x_mean: Array1<f64>,
    /// X-scale at fit time (all ones when `scale = false`).
    pub x_std: Array1<f64>,
    /// Y-mean at fit time.
    pub y_mean: Array1<f64>,
    /// Y-scale at fit time.
    pub y_std: Array1<f64>,
    /// Latent-space weights on `X` (`p × k`).
    pub x_weights: Array2<f64>,
    /// Latent-space weights on `Y` (`q × k`).
    pub y_weights: Array2<f64>,
    /// X-loadings (`p × k`).
    pub x_loadings: Array2<f64>,
    /// Y-loadings (`q × k`).
    pub y_loadings: Array2<f64>,
    /// Rotations `W* = W(PᵀW)⁻¹`.
    pub x_rotations: Array2<f64>,
    /// Response rotations (mirror-image of the X rotations).
    pub y_rotations: Array2<f64>,
    /// Coefficient matrix mapping (scaled) X to (scaled) Y (`p × q`).
    pub coef: Array2<f64>,
    /// Number of latent components actually kept.
    pub n_components: usize,
    /// Number of NIPALS sweeps required (max across components).
    pub max_iter_used: usize,
    /// Whether the fit scaled columns to unit std.
    pub scale: bool,
}

impl PLSRegression {
    /// Fit with `n_components`, `scale = true`, `max_iter = 500`, `tol = 1e-6`.
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
            return Err(Error::Shape(format!(
                "PLSRegression: x has {} rows, y has {}",
                x.nrows(),
                y.nrows()
            )));
        }
        if x.nrows() < 2 || x.ncols() == 0 || y.ncols() == 0 {
            return Err(Error::Value(
                "PLSRegression: need ≥ 2 samples and ≥ 1 feature per block".into(),
            ));
        }
        let n = x.nrows();
        let p = x.ncols();
        let q = y.ncols();
        let max_k = p.min(n - 1);
        if n_components == 0 || n_components > max_k {
            return Err(Error::Value(format!(
                "PLSRegression: n_components={n_components} out of valid range [1, {max_k}]"
            )));
        }
        let (x_mean, x_std) = center_scale(x, scale);
        let (y_mean, y_std) = center_scale(y, scale);
        let mut xk = to_scaled(x, &x_mean, &x_std);
        let mut yk = to_scaled(y, &y_mean, &y_std);
        let mut w_mat = Array2::<f64>::zeros((p, n_components));
        let mut c_mat = Array2::<f64>::zeros((q, n_components));
        let mut t_mat = Array2::<f64>::zeros((n, n_components));
        let mut u_mat = Array2::<f64>::zeros((n, n_components));
        let mut p_mat = Array2::<f64>::zeros((p, n_components));
        let mut q_mat = Array2::<f64>::zeros((q, n_components));
        let mut max_iter_used = 0_usize;
        for k in 0..n_components {
            let (w, c, t, u, iters) = nipals(&xk, &yk, max_iter, tol);
            if iters > max_iter_used {
                max_iter_used = iters;
            }
            // Loadings.
            let t_dot_t = dot_vv(&t, &t).max(1e-30);
            let mut p_load = vec![0.0_f64; p];
            for j in 0..p {
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += xk[[i, j]] * t[i];
                }
                p_load[j] = s / t_dot_t;
            }
            let u_dot_u = dot_vv(&u, &u).max(1e-30);
            let mut q_load = vec![0.0_f64; q];
            for j in 0..q {
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += yk[[i, j]] * u[i];
                }
                q_load[j] = s / u_dot_u;
            }
            // Deflate — regression mode uses `t·pᵀ` for X and `t·(uᵀy)/(uᵀu)` for Y
            // via `t·cᵀ` after normalising; the classic PLS-regression choice
            // is `t·(1/tᵀt · tᵀy)` which is `t·qᵀ` with `q = Yᵀt/tᵀt` — that is
            // what the reference does with `deflation_mode='regression'`.
            let mut q_reg = vec![0.0_f64; q];
            for j in 0..q {
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += yk[[i, j]] * t[i];
                }
                q_reg[j] = s / t_dot_t;
            }
            for i in 0..n {
                for j in 0..p {
                    xk[[i, j]] -= t[i] * p_load[j];
                }
                for j in 0..q {
                    yk[[i, j]] -= t[i] * q_reg[j];
                }
            }
            // Stash.
            for j in 0..p {
                w_mat[[j, k]] = w[j];
                p_mat[[j, k]] = p_load[j];
            }
            for j in 0..q {
                c_mat[[j, k]] = c[j];
                q_mat[[j, k]] = q_reg[j];
            }
            for i in 0..n {
                t_mat[[i, k]] = t[i];
                u_mat[[i, k]] = u[i];
            }
        }
        // Rotations: W* = W (PᵀW)⁻¹
        let ptw = matmul_tn(&p_mat, &w_mat); // (k × k)
        let ptw_inv = invert(&ptw)?;
        let x_rotations = matmul_nn(&w_mat, &ptw_inv);
        let y_rotations = {
            // the reference returns the analogous C(QᵀC)⁻¹ in canonical mode; in
            // regression it uses W* on X and (nothing meaningful) on Y — we
            // still expose the analytic C(QᵀC)⁻¹ for downstream inspection.
            let qtc = matmul_tn(&q_mat, &c_mat);
            match invert(&qtc) {
                Ok(inv) => matmul_nn(&c_mat, &inv),
                Err(_) => Array2::<f64>::zeros((q, n_components)),
            }
        };
        // Regression coefficients on the *scaled* variables.
        let coef = matmul_nt(&x_rotations, &q_mat);
        // For downstream `predict` we also need the un-scaling info; kept in
        // the struct fields directly (x_mean/x_std/y_mean/y_std).
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
            coef,
            n_components,
            max_iter_used,
            scale,
        })
    }

    /// Predict on new predictors (original, un-scaled space).
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let p = self.x_mean.len();
        let q = self.y_mean.len();
        if x.ncols() != p {
            return Err(Error::Shape(format!(
                "PLSRegression::predict: expected {p} cols, got {}",
                x.ncols()
            )));
        }
        let n = x.nrows();
        let mut out = Array2::<f64>::zeros((n, q));
        for i in 0..n {
            for j in 0..q {
                let mut acc = 0.0_f64;
                for kk in 0..p {
                    let xc = (x[[i, kk]] - self.x_mean[kk]) / self.x_std[kk];
                    acc += xc * self.coef[[kk, j]];
                }
                out[[i, j]] = acc * self.y_std[j] + self.y_mean[j];
            }
        }
        Ok(out)
    }
}

pub(crate) fn center_scale(m: ArrayView2<'_, f64>, scale: bool) -> (Array1<f64>, Array1<f64>) {
    let n = m.nrows() as f64;
    let d = m.ncols();
    let mut mean = Array1::<f64>::zeros(d);
    for j in 0..d {
        let mut s = 0.0_f64;
        for i in 0..m.nrows() {
            s += m[[i, j]];
        }
        mean[j] = s / n;
    }
    let mut std = Array1::<f64>::from_elem(d, 1.0);
    if scale {
        for j in 0..d {
            let mut ss = 0.0_f64;
            for i in 0..m.nrows() {
                let e = m[[i, j]] - mean[j];
                ss += e * e;
            }
            let v = (ss / n).max(1e-30).sqrt();
            std[j] = if v > 0.0 { v } else { 1.0 };
        }
    }
    (mean, std)
}

pub(crate) fn to_scaled(m: ArrayView2<'_, f64>, mean: &Array1<f64>, std: &Array1<f64>) -> Array2<f64> {
    let n = m.nrows();
    let d = m.ncols();
    let mut out = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            out[[i, j]] = (m[[i, j]] - mean[j]) / std[j];
        }
    }
    out
}

pub(crate) fn nipals(
    xk: &Array2<f64>,
    yk: &Array2<f64>,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    let n = xk.nrows();
    let p = xk.ncols();
    let q = yk.ncols();
    // u₀ ← first column of Y, or first column of X if Y is constant zero.
    let mut u = vec![0.0_f64; n];
    for i in 0..n {
        u[i] = yk[[i, 0]];
    }
    if dot_vv(&u, &u) < 1e-30 {
        for i in 0..n {
            u[i] = xk[[i, 0]];
        }
    }
    let mut w = vec![0.0_f64; p];
    let mut t = vec![0.0_f64; n];
    let mut c = vec![0.0_f64; q];
    let mut prev = vec![f64::INFINITY; p];
    let mut iters = 0_usize;
    for it in 0..max_iter {
        iters = it + 1;
        // w = Xᵀu, normalise
        for j in 0..p {
            let mut s = 0.0_f64;
            for i in 0..n {
                s += xk[[i, j]] * u[i];
            }
            w[j] = s;
        }
        let wn = dot_vv(&w, &w).sqrt().max(1e-30);
        for j in 0..p {
            w[j] /= wn;
        }
        // t = Xw
        for i in 0..n {
            let mut s = 0.0_f64;
            for j in 0..p {
                s += xk[[i, j]] * w[j];
            }
            t[i] = s;
        }
        // c = Yᵀt, normalise
        for j in 0..q {
            let mut s = 0.0_f64;
            for i in 0..n {
                s += yk[[i, j]] * t[i];
            }
            c[j] = s;
        }
        let cn = dot_vv(&c, &c).sqrt().max(1e-30);
        for j in 0..q {
            c[j] /= cn;
        }
        // u = Yc
        for i in 0..n {
            let mut s = 0.0_f64;
            for j in 0..q {
                s += yk[[i, j]] * c[j];
            }
            u[i] = s;
        }
        // Convergence: change in w.
        let mut delta = 0.0_f64;
        for j in 0..p {
            delta += (w[j] - prev[j]).powi(2);
        }
        if delta.sqrt() < tol {
            break;
        }
        prev.copy_from_slice(&w);
    }
    (w, c, t, u, iters)
}

pub(crate) fn dot_vv(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut s = 0.0_f64;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

pub(crate) fn matmul_tn(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let k = a.ncols();
    let l = b.ncols();
    let inner = a.nrows();
    let mut out = Array2::<f64>::zeros((k, l));
    for i in 0..k {
        for j in 0..l {
            let mut s = 0.0_f64;
            for r in 0..inner {
                s += a[[r, i]] * b[[r, j]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

pub(crate) fn matmul_nn(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
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

pub(crate) fn matmul_nt(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let n = b.nrows();
    let inner = a.ncols();
    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f64;
            for r in 0..inner {
                s += a[[i, r]] * b[[j, r]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

pub(crate) fn invert(m: &Array2<f64>) -> Result<Array2<f64>> {
    let n = m.nrows();
    if n == 0 || n != m.ncols() {
        return Err(Error::Shape("invert: expected a square matrix".into()));
    }
    let mut a = vec![vec![0.0_f64; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = m[[i, j]];
        }
        a[i][n + i] = 1.0;
    }
    for i in 0..n {
        let mut piv = i;
        let mut best = a[i][i].abs();
        for r in (i + 1)..n {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                piv = r;
            }
        }
        if best < 1e-30 {
            return Err(Error::Value("invert: singular matrix".into()));
        }
        if piv != i {
            a.swap(i, piv);
        }
        let d = a[i][i];
        for c in 0..(2 * n) {
            a[i][c] /= d;
        }
        for r in 0..n {
            if r == i {
                continue;
            }
            let f = a[r][i];
            if f == 0.0 {
                continue;
            }
            for c in 0..(2 * n) {
                a[r][c] -= f * a[i][c];
            }
        }
    }
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = a[i][n + j];
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn pls_predicts_close_to_y_on_a_linear_signal() {
        // y = 2·x₁ + 3·x₂
        let x = array![
            [1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 2.0],
            [5.0, 4.0], [6.0, 5.0], [7.0, 4.0], [8.0, 6.0]
        ];
        let y_data: Vec<f64> = (0..8)
            .flat_map(|i| {
                let row = x.row(i);
                vec![2.0 * row[0] + 3.0 * row[1]]
            })
            .collect();
        let y = Array2::from_shape_vec((8, 1), y_data).unwrap();
        let m = PLSRegression::fit(x.view(), y.view(), 2).unwrap();
        let pred = m.predict(x.view()).unwrap();
        for i in 0..8 {
            assert!(
                (pred[[i, 0]] - y[[i, 0]]).abs() < 1e-8,
                "row {i}: pred={} y={}",
                pred[[i, 0]],
                y[[i, 0]]
            );
        }
    }
}

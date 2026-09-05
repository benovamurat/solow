//! Canonical-Correlation Analysis (Hotelling 1936). The correlation-
//! maximising dual of PLS.
//!
//! We solve the generalised eigenproblem
//!
//! ```text
//!     Σ_xy Σ_yy⁻¹ Σ_yx  wₓ = λ  Σ_xx  wₓ
//!     Σ_yx Σ_xx⁻¹ Σ_xy  wᵧ = λ  Σ_yy  wᵧ
//! ```
//!
//! by reducing to a symmetric eigenproblem after the Cholesky whitening
//! `Σ_xx = LₓLₓᵀ`, `Σ_yy = LᵧLᵧᵀ`, then computing the SVD of
//! `M = Lₓ⁻¹ Σ_xy Lᵧ⁻ᵀ`. The canonical correlations are the singular
//! values of `M`, and the loading vectors come from back-substituting the
//! left / right singular vectors.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::pls_regression::{center_scale, matmul_nn, matmul_tn, to_scaled};

/// Fitted CCA model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct CCA {
    /// X-mean at fit time.
    pub x_mean: Array1<f64>,
    /// X-std at fit time (or ones).
    pub x_std: Array1<f64>,
    /// Y-mean at fit time.
    pub y_mean: Array1<f64>,
    /// Y-std at fit time.
    pub y_std: Array1<f64>,
    /// X-side canonical directions (`p × k`).
    pub x_rotations: Array2<f64>,
    /// Y-side canonical directions (`q × k`).
    pub y_rotations: Array2<f64>,
    /// Canonical correlations (length `k`).
    pub correlations: Array1<f64>,
    /// Kept rank.
    pub n_components: usize,
}

impl CCA {
    /// Fit with `scale = true` and a small ridge on `Σ_xx`, `Σ_yy` for
    /// numerical stability.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        n_components: usize,
    ) -> Result<Self> {
        Self::fit_with(x, y, n_components, true, 1e-8)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        n_components: usize,
        scale: bool,
        ridge: f64,
    ) -> Result<Self> {
        if x.nrows() != y.nrows() {
            return Err(Error::Shape("CCA: row counts differ".into()));
        }
        let n = x.nrows();
        let p = x.ncols();
        let q = y.ncols();
        if n_components == 0 || n_components > p.min(q).min(n - 1) {
            return Err(Error::Value("CCA: n_components out of range".into()));
        }
        let (x_mean, x_std) = center_scale(x, scale);
        let (y_mean, y_std) = center_scale(y, scale);
        let xs = to_scaled(x, &x_mean, &x_std);
        let ys = to_scaled(y, &y_mean, &y_std);
        let sxx = gram(&xs, ridge);
        let syy = gram(&ys, ridge);
        let sxy = matmul_tn(&xs, &ys); // (p × q)
        let lx = cholesky(&sxx)?;
        let ly = cholesky(&syy)?;
        // M = Lₓ⁻¹ Σ_xy Lᵧ⁻ᵀ  →  solve Lₓ X = Σ_xy   then  Lᵧ Yᵀ = Xᵀ.
        let m1 = solve_lower(&lx, &sxy)?; // (p × q)
        // Solve Lᵧᵀ · Zᵀ = m1ᵀ  →  Z = solve_lower_transposed on rows.
        let m2t = solve_lower(&ly, &m1.t().to_owned())?; // (q × p)
        let m = m2t.t().to_owned(); // (p × q)
        let (u, s, v) = super::pls_svd::svd_helper(&m, 300, 1e-12);
        // Backtransform:
        //     wₓ ← Lₓ⁻ᵀ · uₖ   (solve Lₓᵀ x = u)
        //     wᵧ ← Lᵧ⁻ᵀ · vₖ
        let k = n_components.min(u.ncols());
        let mut x_rot = Array2::<f64>::zeros((p, k));
        let mut y_rot = Array2::<f64>::zeros((q, k));
        for j in 0..k {
            let mut col_u = vec![0.0_f64; p];
            let mut col_v = vec![0.0_f64; q];
            for i in 0..p {
                col_u[i] = u[[i, j]];
            }
            for i in 0..q {
                col_v[i] = v[[i, j]];
            }
            let wx = solve_upper_from_lower(&lx, &col_u)?;
            let wy = solve_upper_from_lower(&ly, &col_v)?;
            for i in 0..p {
                x_rot[[i, j]] = wx[i];
            }
            for i in 0..q {
                y_rot[[i, j]] = wy[i];
            }
        }
        let mut corr = Array1::<f64>::zeros(k);
        for j in 0..k {
            corr[j] = (s[j] / (n as f64).max(1.0)).min(1.0);
        }
        Ok(Self {
            x_mean,
            x_std,
            y_mean,
            y_std,
            x_rotations: x_rot,
            y_rotations: y_rot,
            correlations: corr,
            n_components: k,
        })
    }
}

fn gram(m: &Array2<f64>, ridge: f64) -> Array2<f64> {
    let g = matmul_tn(m, m);
    let mut out = g;
    let d = out.nrows();
    for i in 0..d {
        out[[i, i]] += ridge;
    }
    out
}

/// Cholesky decomposition L such that L·Lᵀ = A, A symmetric PD.
fn cholesky(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0_f64;
            for k in 0..j {
                s += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let v = a[[i, i]] - s;
                if v <= 0.0 {
                    return Err(Error::Value(
                        "cholesky: matrix not positive definite (raise ridge)".into(),
                    ));
                }
                l[[i, j]] = v.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - s) / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// Solve L·X = B for X where L is lower-triangular; B is `(n × k)`.
fn solve_lower(l: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let n = l.nrows();
    let k = b.ncols();
    if b.nrows() != n {
        return Err(Error::Shape("solve_lower: row-count mismatch".into()));
    }
    let mut x = Array2::<f64>::zeros((n, k));
    for c in 0..k {
        for i in 0..n {
            let mut s = b[[i, c]];
            for r in 0..i {
                s -= l[[i, r]] * x[[r, c]];
            }
            x[[i, c]] = s / l[[i, i]];
        }
    }
    Ok(x)
}

/// Solve Lᵀ·x = b for a single-column b, where L is lower-triangular.
fn solve_upper_from_lower(l: &Array2<f64>, b: &[f64]) -> Result<Vec<f64>> {
    let n = l.nrows();
    if b.len() != n {
        return Err(Error::Shape("solve_upper_from_lower: length mismatch".into()));
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for r in (i + 1)..n {
            s -= l[[r, i]] * x[r];
        }
        x[i] = s / l[[i, i]];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cca_gives_descending_correlations() {
        let x = array![
            [1.0, 2.0, 3.0], [2.0, 3.0, 5.0], [3.0, 5.0, 8.0],
            [4.0, 7.0, 11.0], [5.0, 9.0, 14.0]
        ];
        let y = array![
            [2.0, 1.0], [3.0, 2.0], [5.0, 3.0], [7.0, 4.0], [9.0, 5.0]
        ];
        let m = CCA::fit(x.view(), y.view(), 2).unwrap();
        assert_eq!(m.correlations.len(), 2);
        assert!(m.correlations[0] >= m.correlations[1]);
    }
}

// Anti-dead-code helper: touch matmul_nn so the shared function stays
// callable from downstream tests when they need it.
#[allow(dead_code)]
fn _touch(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    matmul_nn(a, b)
}

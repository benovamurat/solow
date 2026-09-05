//! LARS / LassoLars / OrthogonalMatchingPursuit — the least-angle
//! regression path family (Efron-Hastie-Johnstone-Tibshirani 2004) plus
//! the correlation-greedy OMP (Pati-Rezaifar-Krishnaprasad 1993).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// Fitted Lars.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Lars {
    /// Coefficients at the final step (`d`).
    pub coef: Array1<f64>,
    /// Optional per-column coefficient path (`n_iter × d`).
    pub coef_path: Vec<Array1<f64>>,
    /// Column indices in the order they entered the active set.
    pub active_order: Vec<usize>,
    /// Column mean subtracted at fit (returned in `intercept`).
    pub intercept: f64,
    /// Steps taken.
    pub n_iter: usize,
}

impl Lars {
    /// Fit up to `n_nonzero_coefs` nonzero components (`d` by default via
    /// `fit`).
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<Self> {
        let d = x.ncols();
        Self::fit_with(x, y, d.min(x.nrows() - 1).max(1))
    }

    /// Fit with a caller-supplied non-zero-coefficient cap.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        n_nonzero_coefs: usize,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if y.len() != n {
            return Err(Error::Shape("Lars: y/x row mismatch".into()));
        }
        if n_nonzero_coefs == 0 {
            return Err(Error::Value("Lars: n_nonzero_coefs must be ≥ 1".into()));
        }
        // Centre y and columns of X.
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let mut yc = Array1::<f64>::zeros(n);
        for i in 0..n {
            yc[i] = y[i] - y_mean;
        }
        let mut xc = Array2::<f64>::zeros((n, d));
        let mut x_mean = Array1::<f64>::zeros(d);
        for j in 0..d {
            let m: f64 = (0..n).map(|i| x[[i, j]]).sum::<f64>() / n as f64;
            x_mean[j] = m;
            for i in 0..n {
                xc[[i, j]] = x[[i, j]] - m;
            }
        }
        // Residual r = y − Xβ.
        let mut beta = Array1::<f64>::zeros(d);
        let mut resid = yc.clone();
        let mut active: Vec<usize> = Vec::new();
        let mut path: Vec<Array1<f64>> = vec![beta.clone()];
        let mut iters = 0_usize;
        while active.len() < n_nonzero_coefs && active.len() < d {
            iters += 1;
            // Correlations c = Xᵀ r.
            let mut cor = Array1::<f64>::zeros(d);
            for j in 0..d {
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += xc[[i, j]] * resid[i];
                }
                cor[j] = s;
            }
            // Pick the feature with largest |cor| that isn't already active.
            let mut best = usize::MAX;
            let mut best_v = -1.0_f64;
            for j in 0..d {
                if active.contains(&j) {
                    continue;
                }
                if cor[j].abs() > best_v {
                    best_v = cor[j].abs();
                    best = j;
                }
            }
            if best == usize::MAX || best_v < 1e-12 {
                break;
            }
            active.push(best);
            // Equiangular direction: w = (XₐᵀXₐ)⁻¹·sgn(c_active); u = Xₐ·w.
            let a = active.len();
            let mut xa = Array2::<f64>::zeros((n, a));
            let mut signs = Array1::<f64>::zeros(a);
            for (k, &j) in active.iter().enumerate() {
                signs[k] = cor[j].signum();
                for i in 0..n {
                    xa[[i, k]] = xc[[i, j]] * signs[k];
                }
            }
            let mut xtx = Array2::<f64>::zeros((a, a));
            for i in 0..a {
                for j in 0..a {
                    let mut s = 0.0_f64;
                    for r in 0..n {
                        s += xa[[r, i]] * xa[[r, j]];
                    }
                    xtx[[i, j]] = s;
                }
            }
            let inv = match invert(&xtx) {
                Ok(m) => m,
                Err(_) => break,
            };
            let ones = Array1::<f64>::from_elem(a, 1.0);
            let ainv1 = matvec(&inv, &ones);
            let mut inner = 0.0_f64;
            for v in ainv1.iter() {
                inner += v;
            }
            // A_A = 1 / sqrt(1ᵀ G⁻¹ 1); reciprocal is used throughout.
            let aa = 1.0 / inner.abs().sqrt().max(1e-30);
            let mut w = Array1::<f64>::zeros(a);
            for k in 0..a {
                w[k] = aa * ainv1[k];
            }
            let mut u = Array1::<f64>::zeros(n);
            for i in 0..n {
                let mut s = 0.0_f64;
                for k in 0..a {
                    s += xa[[i, k]] * w[k];
                }
                u[i] = s;
            }
            // Compute the step size γ.
            let corr_max = best_v;
            let mut gamma = corr_max / aa;
            for j in 0..d {
                if active.contains(&j) {
                    continue;
                }
                let mut aj = 0.0_f64;
                for i in 0..n {
                    aj += xc[[i, j]] * u[i];
                }
                let g1 = (corr_max - cor[j]) / (aa - aj).max(1e-30);
                let g2 = (corr_max + cor[j]) / (aa + aj).max(1e-30);
                if g1 > 1e-12 && g1 < gamma {
                    gamma = g1;
                }
                if g2 > 1e-12 && g2 < gamma {
                    gamma = g2;
                }
            }
            // Update.
            for (k, &j) in active.iter().enumerate() {
                beta[j] += gamma * signs[k] * w[k];
            }
            for i in 0..n {
                resid[i] -= gamma * u[i];
            }
            path.push(beta.clone());
        }
        let intercept = y_mean - {
            let mut s = 0.0_f64;
            for j in 0..d {
                s += x_mean[j] * beta[j];
            }
            s
        };
        Ok(Self {
            coef: beta,
            coef_path: path,
            active_order: active,
            intercept,
            n_iter: iters,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        if d != self.coef.len() {
            return Err(Error::Shape("Lars::predict: shape mismatch".into()));
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = self.intercept;
            for j in 0..d {
                s += x[[i, j]] * self.coef[j];
            }
            out[i] = s;
        }
        Ok(out)
    }
}

/// LassoLars — L1-regularised path of Lars (Efron-Hastie-Johnstone-
/// Tibshirani 2004, §7). For simplicity we return the Lars solution
/// with a caller-specified L1 penalty enforced via early-stopping when
/// the maximum correlation drops below `alpha`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LassoLars {
    /// Coefficients.
    pub coef: Array1<f64>,
    /// Intercept.
    pub intercept: f64,
    /// L1 penalty used.
    pub alpha: f64,
    /// Steps taken.
    pub n_iter: usize,
}

impl LassoLars {
    /// Fit with the reference defaults `alpha = 1.0`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<Self> {
        Self::fit_with(x, y, 1.0)
    }

    /// Full-configuration fit.
    pub fn fit_with(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>, alpha: f64) -> Result<Self> {
        let n = x.nrows();
        if alpha < 0.0 {
            return Err(Error::Value("LassoLars: alpha must be ≥ 0".into()));
        }
        // Full Lars path, then post-truncate by soft-thresholding.
        let lars = Lars::fit(x, y)?;
        let mut coef = lars.coef.clone();
        for c in coef.iter_mut() {
            *c = if c.abs() < alpha {
                0.0
            } else if *c > 0.0 {
                *c - alpha
            } else {
                *c + alpha
            };
        }
        // Recompute intercept given the truncated coefficients.
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let d = x.ncols();
        let mut x_mean = vec![0.0_f64; d];
        for j in 0..d {
            let m: f64 = (0..n).map(|i| x[[i, j]]).sum::<f64>() / n as f64;
            x_mean[j] = m;
        }
        let mut intercept = y_mean;
        for j in 0..d {
            intercept -= x_mean[j] * coef[j];
        }
        Ok(Self {
            coef,
            intercept,
            alpha,
            n_iter: lars.n_iter,
        })
    }
}

/// OrthogonalMatchingPursuit — greedy sparse regression that adds one
/// atom per iteration until either `n_nonzero_coefs` are selected or
/// the residual norm drops below `tol`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct OrthogonalMatchingPursuit {
    /// Coefficients.
    pub coef: Array1<f64>,
    /// Intercept.
    pub intercept: f64,
    /// Ordered active atoms.
    pub active: Vec<usize>,
    /// Iterations taken.
    pub n_iter: usize,
}

impl OrthogonalMatchingPursuit {
    /// Fit.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        n_nonzero_coefs: usize,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if y.len() != n {
            return Err(Error::Shape("OMP: y/x row mismatch".into()));
        }
        if n_nonzero_coefs == 0 {
            return Err(Error::Value("OMP: n_nonzero_coefs must be ≥ 1".into()));
        }
        // Centre.
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let mut resid = Array1::<f64>::zeros(n);
        for i in 0..n {
            resid[i] = y[i] - y_mean;
        }
        let mut xc = Array2::<f64>::zeros((n, d));
        let mut x_mean = vec![0.0_f64; d];
        for j in 0..d {
            let m: f64 = (0..n).map(|i| x[[i, j]]).sum::<f64>() / n as f64;
            x_mean[j] = m;
            for i in 0..n {
                xc[[i, j]] = x[[i, j]] - m;
            }
        }
        let mut active: Vec<usize> = Vec::new();
        let mut coef = Array1::<f64>::zeros(d);
        let mut iters = 0_usize;
        for _ in 0..n_nonzero_coefs.min(d) {
            iters += 1;
            let mut best = usize::MAX;
            let mut best_v = -1.0_f64;
            for j in 0..d {
                if active.contains(&j) {
                    continue;
                }
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += xc[[i, j]] * resid[i];
                }
                if s.abs() > best_v {
                    best_v = s.abs();
                    best = j;
                }
            }
            if best == usize::MAX || best_v < 1e-12 {
                break;
            }
            active.push(best);
            // Solve LS on the active columns.
            let a = active.len();
            let mut xa = Array2::<f64>::zeros((n, a));
            for (k, &j) in active.iter().enumerate() {
                for i in 0..n {
                    xa[[i, k]] = xc[[i, j]];
                }
            }
            let mut xtx = Array2::<f64>::zeros((a, a));
            let mut xty = Array1::<f64>::zeros(a);
            for i in 0..a {
                for j in 0..a {
                    let mut s = 0.0_f64;
                    for r in 0..n {
                        s += xa[[r, i]] * xa[[r, j]];
                    }
                    xtx[[i, j]] = s;
                }
                let mut s = 0.0_f64;
                for r in 0..n {
                    s += xa[[r, i]] * (y[r] - y_mean);
                }
                xty[i] = s;
            }
            let inv = match invert(&xtx) {
                Ok(m) => m,
                Err(_) => break,
            };
            let sol = matvec(&inv, &xty);
            for (k, &j) in active.iter().enumerate() {
                coef[j] = sol[k];
            }
            // Update residual.
            for i in 0..n {
                let mut yhat = y_mean;
                for j in 0..d {
                    yhat += xc[[i, j]] * coef[j];
                }
                resid[i] = y[i] - yhat;
            }
        }
        let mut intercept = y_mean;
        for j in 0..d {
            intercept -= x_mean[j] * coef[j];
        }
        Ok(Self {
            coef,
            intercept,
            active,
            n_iter: iters,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        if d != self.coef.len() {
            return Err(Error::Shape("OrthogonalMatchingPursuit::predict: shape mismatch".into()));
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = self.intercept;
            for j in 0..d {
                s += x[[i, j]] * self.coef[j];
            }
            out[i] = s;
        }
        Ok(out)
    }
}

fn matvec(m: &Array2<f64>, v: &Array1<f64>) -> Array1<f64> {
    let n = m.nrows();
    let p = m.ncols();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..p {
            s += m[[i, j]] * v[j];
        }
        out[i] = s;
    }
    out
}

fn invert(m: &Array2<f64>) -> Result<Array2<f64>> {
    let n = m.nrows();
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
            return Err(Error::Value("lars::invert: singular matrix".into()));
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
    fn lars_recovers_a_simple_linear_signal() {
        let x = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0_f64, 4.0, 6.0, 8.0, 10.0];
        let m = Lars::fit(x.view(), y.view()).unwrap();
        let p = m.predict(x.view()).unwrap();
        for i in 0..5 {
            assert!((p[i] - y[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn omp_selects_the_correct_active_set() {
        // y = 3·x₀; x₁, x₂ are noise.
        let x = array![
            [1.0_f64, 0.7, -0.2], [2.0, -1.3, 0.4], [3.0, 0.5, -0.1],
            [4.0, 1.9, 2.4], [5.0, -0.6, 1.1]
        ];
        let y = array![3.0_f64, 6.0, 9.0, 12.0, 15.0];
        let m = OrthogonalMatchingPursuit::fit(x.view(), y.view(), 1).unwrap();
        assert_eq!(m.active, vec![0]);
    }
}

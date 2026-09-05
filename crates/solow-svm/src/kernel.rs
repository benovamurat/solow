//! Kernel SVM — a Sequential-Minimal-Optimisation-lite fit for binary
//! classification (Svc) and ε-insensitive regression (Svr) with
//! `Linear`, `Rbf`, `Polynomial`, or `Sigmoid` kernels.
//!
//! The solver is a simplified two-variable smo-style coordinate update
//! chosen to be small, deterministic, and easy to reason about. It is
//! quadratic in `n` per epoch, so this implementation targets small-to-
//! medium datasets (thousands of rows) — appropriate for the reference parity
//! on the reference-fixture suite.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// Kernel family.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum KernelKind {
    Linear,
    Rbf { gamma: f64 },
    Polynomial { gamma: f64, coef0: f64, degree: i32 },
    Sigmoid { gamma: f64, coef0: f64 },
}

/// Fitted binary Svc.
#[derive(Clone, Debug)]
pub struct Svc {
    /// Support-vector rows (`n_sv × d`).
    pub support_vectors: Array2<f64>,
    /// Dual coefficients `α · y` (`n_sv`).
    pub dual_coef: Array1<f64>,
    /// Intercept.
    pub intercept: f64,
    /// Kernel used.
    pub kernel: KernelKind,
    /// Class labels seen at fit, stored so predict returns {c0, c1}.
    pub classes: (i64, i64),
}

impl Svc {
    /// Fit with defaults `C = 1.0`, `max_iter = 200`, `tol = 1e-3`.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: &[i64],
        kernel: KernelKind,
    ) -> Result<Self> {
        Self::fit_with(x, y, kernel, 1.0, 200, 1e-3)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: &[i64],
        kernel: KernelKind,
        c: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("Svc::fit_with: y/x length mismatch".into()));
        }
        let mut labels: Vec<i64> = y.to_vec();
        labels.sort();
        labels.dedup();
        if labels.len() != 2 {
            return Err(Error::Value("Svc::fit_with: need exactly 2 classes".into()));
        }
        let (c0, c1) = (labels[0], labels[1]);
        let ys: Vec<f64> = y.iter().map(|&yi| if yi == c1 { 1.0 } else { -1.0 }).collect();
        let (alpha, b) = smo_binary(&x, &ys, &kernel, c, max_iter, tol)?;
        let mut sv_rows: Vec<usize> = Vec::new();
        let mut coefs: Vec<f64> = Vec::new();
        for i in 0..n {
            if alpha[i].abs() > 1e-6 {
                sv_rows.push(i);
                coefs.push(alpha[i] * ys[i]);
            }
        }
        let mut svs = Array2::<f64>::zeros((sv_rows.len(), x.ncols()));
        for (r, &i) in sv_rows.iter().enumerate() {
            for j in 0..x.ncols() {
                svs[[r, j]] = x[[i, j]];
            }
        }
        Ok(Self {
            support_vectors: svs,
            dual_coef: Array1::from_vec(coefs),
            intercept: b,
            kernel,
            classes: (c0, c1),
        })
    }

    /// Decision-function value per row.
    pub fn decision_function(&self, x: ArrayView2<'_, f64>) -> Array1<f64> {
        let n = x.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = self.intercept;
            let xi = x.row(i).to_owned();
            for r in 0..self.support_vectors.nrows() {
                let sv = self.support_vectors.row(r).to_owned();
                s += self.dual_coef[r] * kernel_value(&self.kernel, xi.view(), sv.view());
            }
            out[i] = s;
        }
        out
    }

    /// Predict labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Array1<i64> {
        self.decision_function(x)
            .map(|z| if *z >= 0.0 { self.classes.1 } else { self.classes.0 })
    }
}

/// Fitted ε-insensitive kernel regressor.
#[derive(Clone, Debug)]
pub struct Svr {
    /// Support-vector rows.
    pub support_vectors: Array2<f64>,
    /// Dual coefficients (`α − α*`).
    pub dual_coef: Array1<f64>,
    /// Intercept.
    pub intercept: f64,
    /// Kernel.
    pub kernel: KernelKind,
    /// ε used.
    pub epsilon: f64,
}

impl Svr {
    /// Fit with defaults `C = 1.0`, `epsilon = 0.1`, `max_iter = 200`, `tol = 1e-3`.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: &[f64],
        kernel: KernelKind,
    ) -> Result<Self> {
        Self::fit_with(x, y, kernel, 1.0, 0.1, 200, 1e-3)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: &[f64],
        kernel: KernelKind,
        c: f64,
        epsilon: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        if x.nrows() != y.len() {
            return Err(Error::Shape("Svr::fit_with: y/x length mismatch".into()));
        }
        let (coefs, b) = smo_regression(&x, y, &kernel, c, epsilon, max_iter, tol)?;
        let mut sv_rows: Vec<usize> = Vec::new();
        let mut kept: Vec<f64> = Vec::new();
        for i in 0..y.len() {
            if coefs[i].abs() > 1e-6 {
                sv_rows.push(i);
                kept.push(coefs[i]);
            }
        }
        let mut svs = Array2::<f64>::zeros((sv_rows.len(), x.ncols()));
        for (r, &i) in sv_rows.iter().enumerate() {
            for j in 0..x.ncols() {
                svs[[r, j]] = x[[i, j]];
            }
        }
        Ok(Self {
            support_vectors: svs,
            dual_coef: Array1::from_vec(kept),
            intercept: b,
            kernel,
            epsilon,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Array1<f64> {
        let n = x.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = self.intercept;
            let xi = x.row(i).to_owned();
            for r in 0..self.support_vectors.nrows() {
                let sv = self.support_vectors.row(r).to_owned();
                s += self.dual_coef[r] * kernel_value(&self.kernel, xi.view(), sv.view());
            }
            out[i] = s;
        }
        out
    }
}

pub(crate) fn kernel_value(
    kernel: &KernelKind,
    a: ArrayView1<'_, f64>,
    b: ArrayView1<'_, f64>,
) -> f64 {
    match *kernel {
        KernelKind::Linear => {
            let mut s = 0.0_f64;
            for i in 0..a.len() {
                s += a[i] * b[i];
            }
            s
        }
        KernelKind::Rbf { gamma } => {
            let mut s = 0.0_f64;
            for i in 0..a.len() {
                let d = a[i] - b[i];
                s += d * d;
            }
            (-gamma * s).exp()
        }
        KernelKind::Polynomial { gamma, coef0, degree } => {
            let mut s = 0.0_f64;
            for i in 0..a.len() {
                s += a[i] * b[i];
            }
            (gamma * s + coef0).powi(degree)
        }
        KernelKind::Sigmoid { gamma, coef0 } => {
            let mut s = 0.0_f64;
            for i in 0..a.len() {
                s += a[i] * b[i];
            }
            (gamma * s + coef0).tanh()
        }
    }
}

pub(crate) fn smo_binary(
    x: &ArrayView2<'_, f64>,
    y: &[f64],
    kernel: &KernelKind,
    c: f64,
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, f64)> {
    let n = x.nrows();
    if n < 2 {
        return Err(Error::Value("smo_binary: need at least 2 samples".into()));
    }
    let mut alpha = vec![0.0_f64; n];
    let mut b = 0.0_f64;
    // Pre-cache row views for the kernel calls.
    let rows: Vec<_> = (0..n).map(|i| x.row(i).to_owned()).collect();
    for _pass in 0..max_iter {
        let mut num_changed = 0_usize;
        for i in 0..n {
            let ei = decision(&alpha, y, kernel, &rows, i, b) - y[i];
            if (y[i] * ei < -tol && alpha[i] < c) || (y[i] * ei > tol && alpha[i] > 0.0) {
                // Pick j deterministically ≠ i.
                let j = (i + 1) % n;
                let ej = decision(&alpha, y, kernel, &rows, j, b) - y[j];
                let a_i_old = alpha[i];
                let a_j_old = alpha[j];
                let (l, h) = if y[i] != y[j] {
                    ((alpha[j] - alpha[i]).max(0.0), (c + alpha[j] - alpha[i]).min(c))
                } else {
                    ((alpha[i] + alpha[j] - c).max(0.0), (alpha[i] + alpha[j]).min(c))
                };
                if (h - l).abs() < 1e-8 {
                    continue;
                }
                let k_ii = kernel_value(kernel, rows[i].view(), rows[i].view());
                let k_jj = kernel_value(kernel, rows[j].view(), rows[j].view());
                let k_ij = kernel_value(kernel, rows[i].view(), rows[j].view());
                let eta = 2.0 * k_ij - k_ii - k_jj;
                if eta >= 0.0 {
                    continue;
                }
                alpha[j] = (a_j_old - y[j] * (ei - ej) / eta).clamp(l, h);
                if (alpha[j] - a_j_old).abs() < 1e-6 {
                    continue;
                }
                alpha[i] = a_i_old + y[i] * y[j] * (a_j_old - alpha[j]);
                let b1 = b - ei
                    - y[i] * (alpha[i] - a_i_old) * k_ii
                    - y[j] * (alpha[j] - a_j_old) * k_ij;
                let b2 = b - ej
                    - y[i] * (alpha[i] - a_i_old) * k_ij
                    - y[j] * (alpha[j] - a_j_old) * k_jj;
                b = if 0.0 < alpha[i] && alpha[i] < c {
                    b1
                } else if 0.0 < alpha[j] && alpha[j] < c {
                    b2
                } else {
                    0.5 * (b1 + b2)
                };
                num_changed += 1;
            }
        }
        if num_changed == 0 {
            break;
        }
    }
    Ok((alpha, b))
}

fn decision(
    alpha: &[f64],
    y: &[f64],
    kernel: &KernelKind,
    rows: &[Array1<f64>],
    i: usize,
    b: f64,
) -> f64 {
    let mut s = b;
    for k in 0..alpha.len() {
        if alpha[k].abs() < 1e-12 {
            continue;
        }
        s += alpha[k] * y[k] * kernel_value(kernel, rows[k].view(), rows[i].view());
    }
    s
}

pub(crate) fn smo_regression(
    x: &ArrayView2<'_, f64>,
    y: &[f64],
    kernel: &KernelKind,
    c: f64,
    epsilon: f64,
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, f64)> {
    let n = x.nrows();
    if n < 2 {
        return Err(Error::Value("smo_regression: need ≥ 2 samples".into()));
    }
    // Standard SVR SMO folds α, α* into a signed coefficient `η_i = αᵢ − αᵢ*`
    // and iterates on `η`. Here we use a simple coordinate descent on `η`.
    let rows: Vec<_> = (0..n).map(|i| x.row(i).to_owned()).collect();
    let mut eta = vec![0.0_f64; n];
    let mut b = 0.0_f64;
    for _pass in 0..max_iter {
        let mut num_changed = 0_usize;
        for i in 0..n {
            let f_i = predict_reg(&eta, kernel, &rows, i, b);
            let err = f_i - y[i];
            let update = -err / kernel_value(kernel, rows[i].view(), rows[i].view()).max(1e-12);
            let mut new = eta[i] + update;
            // Apply epsilon-insensitive shrinkage and box constraint.
            if err.abs() < epsilon {
                new = 0.0;
            }
            new = new.clamp(-c, c);
            if (new - eta[i]).abs() > tol {
                num_changed += 1;
            }
            eta[i] = new;
        }
        // Update bias to centre residuals.
        let mut mean_resid = 0.0_f64;
        for i in 0..n {
            mean_resid += y[i] - predict_reg(&eta, kernel, &rows, i, 0.0);
        }
        b = mean_resid / n as f64;
        if num_changed == 0 {
            break;
        }
    }
    Ok((eta, b))
}

fn predict_reg(
    eta: &[f64],
    kernel: &KernelKind,
    rows: &[Array1<f64>],
    i: usize,
    b: f64,
) -> f64 {
    let mut s = b;
    for k in 0..eta.len() {
        if eta[k].abs() < 1e-12 {
            continue;
        }
        s += eta[k] * kernel_value(kernel, rows[k].view(), rows[i].view());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn kernel_svc_rbf_learns_a_ring_bump_dataset() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.2]
        ];
        let y = vec![-1_i64, -1, -1, 1, 1, 1];
        let m = Svc::fit(x.view(), &y, KernelKind::Rbf { gamma: 0.1 }).unwrap();
        let p = m.predict(x.view());
        for i in 0..3 { assert_eq!(p[i], -1); }
        for i in 3..6 { assert_eq!(p[i], 1); }
    }
}

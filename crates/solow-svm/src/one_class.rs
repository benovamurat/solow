//! One-class SVM (Schölkopf-Platt-Shawe-Taylor-Smola-Williamson 2001)
//! — density-support estimation via a hyperplane separating the data
//! from the origin in feature space.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::kernel::{kernel_value, KernelKind};

/// Fitted OneClassSVM.
#[derive(Clone, Debug)]
pub struct OneClassSvm {
    /// Support-vector rows (`n_sv × d`).
    pub support_vectors: Array2<f64>,
    /// Dual coefficients (`n_sv`).
    pub dual_coef: Array1<f64>,
    /// Bias `ρ` (subtracted from decision function).
    pub rho: f64,
    /// Kernel.
    pub kernel: KernelKind,
    /// ν used.
    pub nu: f64,
}

impl OneClassSvm {
    /// Fit with `nu = 0.5`.
    pub fn fit(x: ArrayView2<'_, f64>, kernel: KernelKind) -> Result<Self> {
        Self::fit_with(x, kernel, 0.5, 300, 1e-3)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        kernel: KernelKind,
        nu: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        if n < 2 {
            return Err(Error::Value("OneClassSvm: need ≥ 2 samples".into()));
        }
        if !(0.0..=1.0).contains(&nu) || nu == 0.0 {
            return Err(Error::Value("OneClassSvm: nu must be in (0, 1]".into()));
        }
        let ub = 1.0 / (nu * n as f64);
        let mut alpha = vec![1.0 / n as f64; n];
        // Renormalise so Σ αᵢ = 1.
        let s: f64 = alpha.iter().sum::<f64>();
        for a in alpha.iter_mut() {
            *a /= s;
        }
        let rows: Vec<_> = (0..n).map(|i| x.row(i).to_owned()).collect();
        // Coordinate descent updates paired with a projection back onto
        // {α : Σα = 1, 0 ≤ αᵢ ≤ ub}.
        for _iter in 0..max_iter {
            let mut max_change = 0.0_f64;
            for i in 0..n {
                let mut grad_i = 0.0_f64;
                for j in 0..n {
                    grad_i += alpha[j] * kernel_value(&kernel, rows[i].view(), rows[j].view());
                }
                // Pair with the next index in round-robin fashion.
                let j = (i + 1) % n;
                let mut grad_j = 0.0_f64;
                for k in 0..n {
                    grad_j += alpha[k] * kernel_value(&kernel, rows[j].view(), rows[k].view());
                }
                let k_ii = kernel_value(&kernel, rows[i].view(), rows[i].view());
                let k_jj = kernel_value(&kernel, rows[j].view(), rows[j].view());
                let k_ij = kernel_value(&kernel, rows[i].view(), rows[j].view());
                let eta = k_ii + k_jj - 2.0 * k_ij;
                if eta < 1e-12 {
                    continue;
                }
                let sum = alpha[i] + alpha[j];
                let a_j_new = (alpha[j] + (grad_i - grad_j) / eta).clamp(
                    (sum - ub).max(0.0),
                    sum.min(ub),
                );
                let a_i_new = sum - a_j_new;
                let change = (a_i_new - alpha[i]).abs() + (a_j_new - alpha[j]).abs();
                if change > max_change {
                    max_change = change;
                }
                alpha[i] = a_i_new;
                alpha[j] = a_j_new;
            }
            if max_change < tol {
                break;
            }
        }
        // Compute ρ from a free support vector (0 < αᵢ < ub).
        let mut rho = 0.0_f64;
        let mut n_free = 0_usize;
        for i in 0..n {
            if alpha[i] > 1e-8 && alpha[i] < ub - 1e-8 {
                let mut s = 0.0_f64;
                for k in 0..n {
                    s += alpha[k] * kernel_value(&kernel, rows[k].view(), rows[i].view());
                }
                rho += s;
                n_free += 1;
            }
        }
        if n_free > 0 {
            rho /= n_free as f64;
        }
        let mut sv_rows: Vec<usize> = Vec::new();
        let mut kept: Vec<f64> = Vec::new();
        for i in 0..n {
            if alpha[i].abs() > 1e-8 {
                sv_rows.push(i);
                kept.push(alpha[i]);
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
            rho,
            kernel,
            nu,
        })
    }

    /// Decision-function value per row. Positive = in-support, negative = outlier.
    pub fn decision_function(&self, x: ArrayView2<'_, f64>) -> Array1<f64> {
        let n = x.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let xi = x.row(i).to_owned();
            let mut s = 0.0_f64;
            for r in 0..self.support_vectors.nrows() {
                let sv = self.support_vectors.row(r).to_owned();
                s += self.dual_coef[r] * kernel_value(&self.kernel, xi.view(), sv.view());
            }
            out[i] = s - self.rho;
        }
        out
    }

    /// Predict `+1` for inliers, `-1` for outliers.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Array1<i64> {
        self.decision_function(x)
            .map(|z| if *z >= 0.0 { 1 } else { -1 })
    }
}

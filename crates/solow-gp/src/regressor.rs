//! Gaussian-process regression.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::kernels::Kernel;

/// Fitted Gaussian-process regressor.
pub struct GaussianProcessRegressor<K: Kernel> {
    /// Training inputs.
    pub x_train: Array2<f64>,
    /// Training targets.
    pub y_train: Array1<f64>,
    /// Kernel.
    pub kernel: K,
    /// Additive noise on the diagonal `α`.
    pub alpha: f64,
    /// Cholesky factor `L` of `K + α·I` (lower-triangular).
    pub l: Array2<f64>,
    /// α-inv weights `α_w = (K + α·I)⁻¹ y`.
    pub alpha_weights: Array1<f64>,
    /// Log-marginal-likelihood at fit.
    pub log_marginal_likelihood: f64,
    /// Constant mean subtracted at fit (added back on predict).
    pub y_mean: f64,
}

impl<K: Kernel> GaussianProcessRegressor<K> {
    /// Fit with the given kernel and additive noise `α`.
    pub fn fit(
        kernel: K,
        x: ArrayView2<'_, f64>,
        y: &[f64],
        alpha: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape(format!(
                "GaussianProcessRegressor::fit: y has {} rows, x has {n}",
                y.len()
            )));
        }
        if n == 0 {
            return Err(Error::Value(
                "GaussianProcessRegressor::fit: need at least one sample".into(),
            ));
        }
        if alpha < 0.0 || !alpha.is_finite() {
            return Err(Error::Value(format!(
                "GaussianProcessRegressor::fit: alpha must be finite and ≥ 0 (got {alpha})"
            )));
        }
        let x_train = x.to_owned();
        let mean = y.iter().sum::<f64>() / n as f64;
        let y_centered = Array1::from_vec(y.iter().map(|v| v - mean).collect::<Vec<_>>());
        let mut k = kernel.gram(x_train.view());
        for i in 0..n {
            k[[i, i]] += alpha;
        }
        let l = cholesky(&k)?;
        // α = L⁻ᵀ L⁻¹ y  →  first solve L z = y  then Lᵀ α = z
        let z = forward_substitute(&l, y_centered.as_slice().unwrap())?;
        let alpha_w = backward_substitute(&l, &z)?;
        // Log-marginal-likelihood = -½ yᵀ α − Σ log Lᵢᵢ − (n/2)·log 2π
        let mut half_yta = 0.0_f64;
        for i in 0..n {
            half_yta += y_centered[i] * alpha_w[i];
        }
        half_yta *= 0.5;
        let mut log_det = 0.0_f64;
        for i in 0..n {
            log_det += l[[i, i]].ln();
        }
        let lml = -half_yta - log_det - (n as f64) * 0.5 * (2.0 * std::f64::consts::PI).ln();
        Ok(Self {
            x_train,
            y_train: Array1::from_vec(y.to_vec()),
            kernel,
            alpha,
            l,
            alpha_weights: Array1::from_vec(alpha_w),
            log_marginal_likelihood: lml,
            y_mean: mean,
        })
    }

    /// Predict posterior mean.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Array1<f64> {
        let k_star = self.kernel.cross(x, self.x_train.view());
        let n = x.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = 0.0_f64;
            for j in 0..self.x_train.nrows() {
                s += k_star[[i, j]] * self.alpha_weights[j];
            }
            out[i] = s + self.y_mean;
        }
        out
    }

    /// Predict posterior mean and standard deviation.
    pub fn predict_with_std(&self, x: ArrayView2<'_, f64>) -> (Array1<f64>, Array1<f64>) {
        let mean = self.predict(x);
        let n = x.nrows();
        let k_star = self.kernel.cross(x, self.x_train.view());
        let mut var = Array1::<f64>::zeros(n);
        let k_diag = self.kernel.diag(x);
        for i in 0..n {
            let mut v = k_star.row(i).to_vec();
            let z = forward_substitute(&self.l, &v).unwrap_or_default();
            let mut s = 0.0_f64;
            for j in 0..z.len() {
                s += z[j] * z[j];
            }
            let vari = (k_diag[i] - s).max(0.0);
            var[i] = vari.sqrt();
            // Prevent Rust complaining about unused `v`.
            v.clear();
        }
        (mean, var)
    }
}

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
                        "cholesky: matrix not positive definite (increase alpha)".into(),
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

fn forward_substitute(l: &Array2<f64>, b: &[f64]) -> Result<Vec<f64>> {
    let n = l.nrows();
    if b.len() != n {
        return Err(Error::Shape("forward_substitute: shape mismatch".into()));
    }
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for r in 0..i {
            s -= l[[i, r]] * x[r];
        }
        x[i] = s / l[[i, i]];
    }
    Ok(x)
}

fn backward_substitute(l: &Array2<f64>, b: &[f64]) -> Result<Vec<f64>> {
    let n = l.nrows();
    if b.len() != n {
        return Err(Error::Shape("backward_substitute: shape mismatch".into()));
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
    use crate::kernels::Rbf;
    use ndarray::array;

    #[test]
    fn gp_interpolates_training_points_at_small_noise() {
        let x = array![[-2.0_f64], [-1.0], [0.0], [1.0], [2.0]];
        let y_vals = x.iter().map(|xi| xi.sin()).collect::<Vec<_>>();
        let gp = GaussianProcessRegressor::fit(
            Rbf::new(1.0),
            x.view(),
            &y_vals,
            1e-8,
        ).unwrap();
        let pred = gp.predict(x.view());
        for i in 0..5 {
            assert!((pred[i] - y_vals[i]).abs() < 1e-4);
        }
    }
}

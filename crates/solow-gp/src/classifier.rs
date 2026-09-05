//! Laplace-approximate Gaussian-process binary classifier (Rasmussen-
//! Williams Ch. 3, Alg. 3.1).
//!
//! Fits `f | y ∼ 𝒩(f̂, (K⁻¹ + W)⁻¹)` with `W = -∇² log p(y | f)` and the
//! logistic likelihood.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::kernels::Kernel;

/// Fitted GP classifier (binary).
pub struct GaussianProcessClassifier<K: Kernel> {
    /// Training inputs.
    pub x_train: Array2<f64>,
    /// Training labels in {0, 1}.
    pub y_train: Array1<u8>,
    /// Kernel.
    pub kernel: K,
    /// Laplace-mode latent `f̂`.
    pub f_mode: Array1<f64>,
    /// `a = ∇ log p(y | f̂) = y − π̂`, kept for predict.
    pub a: Array1<f64>,
    /// `L = chol(I + √W · K · √W)`.
    pub l: Array2<f64>,
    /// √W at the mode.
    pub sqrt_w: Array1<f64>,
    /// Number of Newton iterations used.
    pub n_iter: usize,
}

impl<K: Kernel> GaussianProcessClassifier<K> {
    /// Fit via Newton iteration.
    pub fn fit(
        kernel: K,
        x: ArrayView2<'_, f64>,
        y: &[u8],
    ) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape("GaussianProcessClassifier: y/x size mismatch".into()));
        }
        for &yi in y {
            if yi > 1 {
                return Err(Error::Value(
                    "GaussianProcessClassifier: only binary {0, 1} labels supported".into(),
                ));
            }
        }
        let k = kernel.gram(x);
        let y_f: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        let mut f = vec![0.0_f64; n];
        let mut l = Array2::<f64>::eye(n);
        let mut sqrt_w = vec![0.0_f64; n];
        let mut a = vec![0.0_f64; n];
        let mut iter = 0_usize;
        for it in 0..50 {
            iter = it + 1;
            // π = σ(f)
            let pi: Vec<f64> = f.iter().map(|&fi| sigmoid(fi)).collect();
            let w: Vec<f64> = pi.iter().map(|&p| p * (1.0 - p)).collect();
            for i in 0..n {
                sqrt_w[i] = w[i].sqrt();
            }
            // B = I + √W K √W
            let mut b = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    b[[i, j]] = sqrt_w[i] * k[[i, j]] * sqrt_w[j];
                    if i == j {
                        b[[i, j]] += 1.0;
                    }
                }
            }
            l = cholesky(&b)?;
            let b_vec: Vec<f64> = (0..n).map(|i| w[i] * f[i] + (y_f[i] - pi[i])).collect();
            // Solve for a = b - √W L⁻ᵀ L⁻¹ √W K b
            let kb = matvec(&k, &b_vec);
            let mut swkb = vec![0.0_f64; n];
            for i in 0..n {
                swkb[i] = sqrt_w[i] * kb[i];
            }
            let z = forward_substitute(&l, &swkb)?;
            let z2 = backward_substitute(&l, &z)?;
            let mut a_new = vec![0.0_f64; n];
            for i in 0..n {
                a_new[i] = b_vec[i] - sqrt_w[i] * z2[i];
            }
            let f_new = matvec(&k, &a_new);
            let mut delta = 0.0_f64;
            for i in 0..n {
                delta += (f_new[i] - f[i]).abs();
            }
            f = f_new;
            a = a_new;
            if delta < 1e-6 * (n as f64).max(1.0) {
                break;
            }
        }
        Ok(Self {
            x_train: x.to_owned(),
            y_train: Array1::from_vec(y.to_vec()),
            kernel,
            f_mode: Array1::from_vec(f),
            a: Array1::from_vec(a),
            l,
            sqrt_w: Array1::from_vec(sqrt_w),
            n_iter: iter,
        })
    }

    /// Predict class probabilities `p(y=1)`.
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Array1<f64> {
        let k_star = self.kernel.cross(x, self.x_train.view());
        let n = x.nrows();
        let mut probs = Array1::<f64>::zeros(n);
        // Approximate p(y*=1) ≈ σ(f̄), using MacKay's variance correction
        // κ(σ_f²) = 1 / sqrt(1 + π · σ_f² / 8).
        for i in 0..n {
            let mut fbar = 0.0_f64;
            for j in 0..self.x_train.nrows() {
                fbar += k_star[[i, j]] * self.a[j];
            }
            let mut v = vec![0.0_f64; self.x_train.nrows()];
            for j in 0..self.x_train.nrows() {
                v[j] = self.sqrt_w[j] * k_star[[i, j]];
            }
            let z = forward_substitute(&self.l, &v).unwrap_or_default();
            let mut var_star = self.kernel.call(x.row(i), x.row(i));
            for j in 0..z.len() {
                var_star -= z[j] * z[j];
            }
            let kappa = 1.0 / (1.0 + std::f64::consts::PI * var_star.max(0.0) / 8.0).sqrt();
            probs[i] = sigmoid(kappa * fbar);
        }
        probs
    }

    /// Predict labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Array1<u8> {
        self.predict_proba(x).map(|p| if *p >= 0.5 { 1 } else { 0 })
    }
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn matvec(m: &Array2<f64>, v: &[f64]) -> Vec<f64> {
    let n = m.nrows();
    let p = m.ncols();
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..p {
            s += m[[i, j]] * v[j];
        }
        out[i] = s;
    }
    out
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
                        "cholesky: matrix not positive definite".into(),
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
    fn gpc_separates_two_bumps() {
        let x = array![
            [-2.0], [-1.5], [-1.0], [-0.5],
            [ 0.5], [ 1.0], [ 1.5], [ 2.0]
        ];
        let y: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let gpc = GaussianProcessClassifier::fit(Rbf::new(0.5), x.view(), &y).unwrap();
        let probs = gpc.predict_proba(x.view());
        for i in 0..4 {
            assert!(probs[i] < 0.5, "left half should predict 0");
        }
        for i in 4..8 {
            assert!(probs[i] > 0.5, "right half should predict 1");
        }
    }
}

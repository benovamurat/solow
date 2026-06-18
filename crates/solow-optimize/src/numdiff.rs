//! Numerical differentiation: gradients and Hessians by finite differences.

use ndarray::{Array1, Array2};

/// Central-difference gradient of `f` at `x`.
///
/// Step size `h_i = base · (1 + |x_i|)` with `base = 1e-6`.
pub fn approx_fprime<F>(x: &Array1<f64>, mut f: F) -> Array1<f64>
where
    F: FnMut(&Array1<f64>) -> f64,
{
    let n = x.len();
    let mut g = Array1::<f64>::zeros(n);
    let base = 1e-6;
    let mut xp = x.clone();
    for i in 0..n {
        let h = base * (1.0 + x[i].abs());
        xp[i] = x[i] + h;
        let fp = f(&xp);
        xp[i] = x[i] - h;
        let fm = f(&xp);
        xp[i] = x[i];
        g[i] = (fp - fm) / (2.0 * h);
    }
    g
}

/// Central-difference Hessian of `f` at `x`.
pub fn approx_hess<F>(x: &Array1<f64>, mut f: F) -> Array2<f64>
where
    F: FnMut(&Array1<f64>) -> f64,
{
    let n = x.len();
    let mut h = Array2::<f64>::zeros((n, n));
    let base = 1e-4;
    let steps: Vec<f64> = (0..n).map(|i| base * (1.0 + x[i].abs())).collect();
    let mut xx = x.clone();
    let f0 = f(&xx);
    // Diagonal.
    for i in 0..n {
        let hi = steps[i];
        xx[i] = x[i] + hi;
        let fp = f(&xx);
        xx[i] = x[i] - hi;
        let fm = f(&xx);
        xx[i] = x[i];
        h[[i, i]] = (fp - 2.0 * f0 + fm) / (hi * hi);
    }
    // Off-diagonal (symmetric).
    for i in 0..n {
        for j in (i + 1)..n {
            let hi = steps[i];
            let hj = steps[j];
            xx[i] = x[i] + hi;
            xx[j] = x[j] + hj;
            let fpp = f(&xx);
            xx[j] = x[j] - hj;
            let fpm = f(&xx);
            xx[i] = x[i] - hi;
            let fmm = f(&xx);
            xx[j] = x[j] + hj;
            let fmp = f(&xx);
            xx[i] = x[i];
            xx[j] = x[j];
            let v = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
            h[[i, j]] = v;
            h[[j, i]] = v;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn gradient_of_quadratic() {
        // f(x) = x0^2 + 3 x1^2 - x0 x1 ; ∇f = (2x0 - x1, 6x1 - x0)
        let f = |x: &Array1<f64>| x[0] * x[0] + 3.0 * x[1] * x[1] - x[0] * x[1];
        let x = array![1.5, -2.0];
        let g = approx_fprime(&x, f);
        assert_abs_diff_eq!(g[0], 2.0 * 1.5 - (-2.0), epsilon = 1e-6);
        assert_abs_diff_eq!(g[1], 6.0 * -2.0 - 1.5, epsilon = 1e-6);
    }

    #[test]
    fn hessian_of_quadratic() {
        let f = |x: &Array1<f64>| x[0] * x[0] + 3.0 * x[1] * x[1] - x[0] * x[1];
        let x = array![0.3, 0.7];
        let h = approx_hess(&x, f);
        assert_abs_diff_eq!(h[[0, 0]], 2.0, epsilon = 1e-3);
        assert_abs_diff_eq!(h[[1, 1]], 6.0, epsilon = 1e-3);
        assert_abs_diff_eq!(h[[0, 1]], -1.0, epsilon = 1e-3);
    }
}

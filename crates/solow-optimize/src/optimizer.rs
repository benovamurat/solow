//! Unconstrained optimizers: Newton–Raphson (analytic derivatives) and BFGS
//! (gradient only, with backtracking line search).
//!
//! Both *minimize*. To maximize a log-likelihood, minimize its negative — or use
//! [`newton_stationary`], which simply drives the gradient to zero (max or min).

use ndarray::{Array1, Array2};
use solow_core::error::Result;
use solow_linalg::{lstsq, solve};

/// Outcome of an optimization run.
#[derive(Clone, Debug)]
pub struct OptimizeResult {
    /// The located point.
    pub x: Array1<f64>,
    /// Objective value there.
    pub fval: f64,
    /// Iterations performed.
    pub iters: usize,
    /// Whether the convergence test was met.
    pub converged: bool,
    /// Final gradient norm.
    pub grad_norm: f64,
}

/// Newton iteration that drives the gradient of an objective to zero.
///
/// `fgh(x)` returns `(value, gradient, hessian)`. The update is
/// `x ← x − H⁻¹ g`, which converges to a local maximum or minimum depending on
/// the curvature. A pseudoinverse fallback is used if `H` is singular.
pub fn newton_stationary<F>(
    start: &Array1<f64>,
    mut fgh: F,
    maxiter: usize,
    gtol: f64,
) -> Result<OptimizeResult>
where
    F: FnMut(&Array1<f64>) -> (f64, Array1<f64>, Array2<f64>),
{
    let mut x = start.clone();
    let mut fval;
    let mut gnorm;
    let mut converged = false;
    let mut it = 0;
    loop {
        let (f, g, h) = fgh(&x);
        fval = f;
        gnorm = g.dot(&g).sqrt();
        if gnorm < gtol {
            converged = true;
            break;
        }
        if it >= maxiter {
            break;
        }
        let step = match solve(&h, &g) {
            Ok(s) => s,
            Err(_) => lstsq(&h, &g)?,
        };
        let step_norm = step.dot(&step).sqrt();
        let xnorm = x.dot(&x).sqrt();
        // Damped Newton: take the largest fraction `t ∈ {1, 1/2, 1/4, …}` of the
        // Newton step that reduces the gradient norm. Near the optimum the full step
        // (`t = 1`) is always accepted on the first try, preserving Newton's quadratic
        // convergence (and identical iterates for well-behaved problems); on ill-scaled
        // designs the backtracking keeps the solver globally convergent instead of
        // overshooting into a non-finite region.
        let mut t = 1.0;
        let mut x_new = &x - &step;
        for _ in 0..40 {
            let g_new = fgh(&x_new).1;
            let gnew_norm = g_new.dot(&g_new).sqrt();
            if gnew_norm.is_finite() && gnew_norm < gnorm {
                break;
            }
            t *= 0.5;
            x_new = &x - &(t * &step);
            if t < 1e-12 {
                break;
            }
        }
        x = x_new;
        it += 1;
        // Scale-aware convergence: the full (undamped) Newton step — the Newton
        // decrement `‖H⁻¹g‖` — has shrunk to nothing relative to the parameters. This
        // catches ill-scaled designs where the raw gradient norm has a floating-point
        // roundoff floor above `gtol` even though the MLE has been reached. The Newton
        // step is small only *near* the optimum, so for well-behaved problems the
        // `gnorm < gtol` test above fires first and the converged iterate is unchanged.
        if step_norm <= 1e-12 * (1.0 + xnorm) {
            converged = true;
            break;
        }
    }
    Ok(OptimizeResult {
        x,
        fval,
        iters: it,
        converged,
        grad_norm: gnorm,
    })
}

/// Minimize `f` with gradient `grad` using BFGS and Armijo backtracking.
pub fn minimize_bfgs<F, G>(
    start: &Array1<f64>,
    mut f: F,
    mut grad: G,
    maxiter: usize,
    gtol: f64,
) -> Result<OptimizeResult>
where
    F: FnMut(&Array1<f64>) -> f64,
    G: FnMut(&Array1<f64>) -> Array1<f64>,
{
    let n = start.len();
    let mut x = start.clone();
    let mut hinv = Array2::<f64>::eye(n); // inverse-Hessian approximation
    let mut g = grad(&x);
    let mut fval = f(&x);
    let mut converged = false;
    let mut it = 0;

    while it < maxiter {
        let gnorm = g.dot(&g).sqrt();
        if gnorm < gtol {
            converged = true;
            break;
        }
        // Search direction d = -Hinv g.
        let d = hinv.dot(&g).mapv(|v| -v);
        // Backtracking line search (Armijo).
        let slope = g.dot(&d);
        let mut alpha = 1.0;
        let c = 1e-4;
        let rho = 0.5;
        let mut x_new = &x + &(alpha * &d);
        let mut f_new = f(&x_new);
        let mut ls = 0;
        while f_new > fval + c * alpha * slope && ls < 50 {
            alpha *= rho;
            x_new = &x + &(alpha * &d);
            f_new = f(&x_new);
            ls += 1;
        }
        let g_new = grad(&x_new);
        // BFGS update of the inverse Hessian.
        let s = &x_new - &x;
        let yv = &g_new - &g;
        let sy = s.dot(&yv);
        if sy > 1e-12 {
            let rho_bfgs = 1.0 / sy;
            let i = Array2::<f64>::eye(n);
            // (I - ρ s yᵀ) Hinv (I - ρ y sᵀ) + ρ s sᵀ
            let sy_outer = outer(&s, &yv).mapv(|v| v * rho_bfgs);
            let ys_outer = outer(&yv, &s).mapv(|v| v * rho_bfgs);
            let left = &i - &sy_outer;
            let right = &i - &ys_outer;
            hinv = left.dot(&hinv).dot(&right) + outer(&s, &s).mapv(|v| v * rho_bfgs);
        }
        x = x_new;
        g = g_new;
        fval = f_new;
        it += 1;
    }

    Ok(OptimizeResult {
        x,
        fval,
        iters: it,
        converged,
        grad_norm: g.dot(&g).sqrt(),
    })
}

fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut o = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            o[[i, j]] = a[i] * b[j];
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numdiff::approx_fprime;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn newton_minimizes_quadratic() {
        // f(x) = (x0-3)^2 + 2 (x1+1)^2, min at (3, -1).
        let fgh = |x: &Array1<f64>| {
            let f = (x[0] - 3.0).powi(2) + 2.0 * (x[1] + 1.0).powi(2);
            let g = array![2.0 * (x[0] - 3.0), 4.0 * (x[1] + 1.0)];
            let h = array![[2.0, 0.0], [0.0, 4.0]];
            (f, g, h)
        };
        let r = newton_stationary(&array![0.0, 0.0], fgh, 50, 1e-10).unwrap();
        assert!(r.converged);
        assert_abs_diff_eq!(r.x[0], 3.0, epsilon = 1e-8);
        assert_abs_diff_eq!(r.x[1], -1.0, epsilon = 1e-8);
    }

    #[test]
    fn bfgs_minimizes_rosenbrock() {
        // Rosenbrock: min at (1,1).
        let f = |x: &Array1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let g = |x: &Array1<f64>| approx_fprime(x, f);
        let r = minimize_bfgs(&array![-1.2, 1.0], f, g, 1000, 1e-8).unwrap();
        assert!(r.converged, "did not converge (grad_norm {})", r.grad_norm);
        assert_abs_diff_eq!(r.x[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(r.x[1], 1.0, epsilon = 1e-4);
    }
}

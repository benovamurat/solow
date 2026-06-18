# Optimization

The `solow-optimize` crate provides the unconstrained numerical optimizers that
sit underneath Solow's maximum-likelihood estimators, together with the
finite-difference derivatives those optimizers and the standard-error machinery
depend on. Two solvers are exposed as free functions — `newton_stationary`, a
damped Newton iteration that uses analytic score and Hessian, and
`minimize_bfgs`, a quasi-Newton BFGS minimizer that needs only a gradient — plus
`approx_fprime` and `approx_hess` for central-difference gradients and Hessians.
Every run returns a single `OptimizeResult`.

This crate is deliberately small relative to the reference's optimizer
collection: it ships Newton–Raphson and BFGS with backtracking line search and
finite-difference derivatives, but does not (yet) include the full menu of
methods such as conjugate gradient, Powell, Nelder–Mead, or L-BFGS-B.

## Background

Maximum-likelihood estimation reduces to an unconstrained smooth optimization
problem. Writing the log-likelihood as \\( \ell(\theta) \\), its gradient (the
*score*) and Hessian are

\\[
g(\theta) = \nabla \ell(\theta), \qquad
H(\theta) = \nabla^2 \ell(\theta).
\\]

A stationary point of \\( \ell \\) satisfies \\( g(\theta) = 0 \\). Both solvers
generate a sequence \\( \theta_0, \theta_1, \dots \\) that drives the gradient
toward zero.

**Newton–Raphson.** The Newton update solves the local quadratic model
exactly,

\\[
\theta_{k+1} = \theta_k - H(\theta_k)^{-1} g(\theta_k),
\\]

giving quadratic convergence near the optimum. The displacement
\\( \Delta_k = H^{-1} g \\) is the *Newton step*, and \\( \lVert \Delta_k
\rVert \\) the *Newton decrement*. `newton_stationary` drives the gradient to
zero regardless of curvature, so it converges to a local maximum or minimum
according to the sign of \\( H \\); for a concave log-likelihood the stationary
point is the MLE. When \\( H \\) is singular the step is computed from a
least-squares (pseudoinverse) solve instead of a direct solve.

To stay globally convergent on ill-scaled problems, the Newton step is *damped*:
the solver takes the largest fraction \\( t \in \{1, \tfrac12, \tfrac14,
\dots\} \\) of the step for which the gradient norm strictly decreases and stays
finite,

\\[
\theta_{k+1} = \theta_k - t\, H(\theta_k)^{-1} g(\theta_k).
\\]

Near the optimum the full step \\( t = 1 \\) is accepted on the first try, so
the quadratic convergence of pure Newton is preserved.

**Quasi-Newton (BFGS).** When the Hessian is unavailable or expensive, BFGS
maintains an approximation \\( B_k \approx H_k^{-1} \\) to the *inverse* Hessian
and forms the search direction \\( d_k = -B_k g_k \\). With the curvature pair

\\[
s_k = \theta_{k+1} - \theta_k, \qquad
y_k = g_{k+1} - g_k, \qquad
\rho_k = \frac{1}{s_k^\top y_k},
\\]

the inverse-Hessian update is

\\[
B_{k+1} = (I - \rho_k\, s_k y_k^\top)\, B_k\, (I - \rho_k\, y_k s_k^\top)
          + \rho_k\, s_k s_k^\top ,
\\]

applied only when the curvature condition \\( s_k^\top y_k > 0 \\) holds (here,
above a small tolerance). The iteration starts from \\( B_0 = I \\).

**Line search.** BFGS chooses the step length \\( \alpha \\) by backtracking
until the Armijo sufficient-decrease condition holds,

\\[
f(\theta_k + \alpha d_k) \le f(\theta_k) + c\,\alpha\, g_k^\top d_k,
\qquad c = 10^{-4},
\\]

starting at \\( \alpha = 1 \\) and halving (\\( \rho = \tfrac12 \\)) on each
failure. Note that `minimize_bfgs` *minimizes*: to maximize a log-likelihood,
minimize its negative.

**Convergence.** A run is declared converged when the gradient norm falls below
the tolerance, \\( \lVert g(\theta_k) \rVert < \texttt{gtol} \\). In addition,
`newton_stationary` accepts a scale-aware test: if the Newton step shrinks to
\\( \lVert \Delta_k \rVert \le 10^{-12}(1 + \lVert \theta_k \rVert) \\) it also
declares convergence, which catches well-conditioned MLEs whose raw gradient
norm has a floating-point roundoff floor just above `gtol`.

**Numerical derivatives.** When analytic derivatives are not at hand, central
differences approximate them. With a per-coordinate step \\( h_i \\),

\\[
[\nabla f]_i \approx \frac{f(\theta + h_i e_i) - f(\theta - h_i e_i)}{2 h_i},
\\]

and the second derivatives use the standard central stencils. `approx_fprime`
uses \\( h_i = 10^{-6}(1 + |\theta_i|) \\) and `approx_hess` uses \\( h_i =
10^{-4}(1 + |\theta_i|) \\); the off-diagonal Hessian entries are computed with
the symmetric four-point formula and the result is symmetrized.

## Example

Two short examples: a BFGS minimization of the Rosenbrock function using a
finite-difference gradient, and a Newton solve of a smooth objective with
analytic derivatives.

### BFGS with a finite-difference gradient

```rust
use ndarray::{array, Array1};
use solow_optimize::{approx_fprime, minimize_bfgs};

// Rosenbrock: f(x) = (1 - x0)^2 + 100 (x1 - x0^2)^2, global min at (1, 1).
let f = |x: &Array1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);

// Supply the gradient by central differences rather than deriving it by hand.
let grad = |x: &Array1<f64>| approx_fprime(x, f);

let start = array![-1.2, 1.0];
let res = minimize_bfgs(&start, f, grad, 1000, 1e-8).unwrap();

assert!(res.converged);
println!("minimizer = {:?}", res.x);     // close to [1.0, 1.0]
println!("fval      = {:.3e}", res.fval); // close to 0
println!("iters     = {}", res.iters);
println!("grad_norm = {:.3e}", res.grad_norm);
```

The fitted `OptimizeResult` exposes the located point `x`, the objective value
`fval` there, the number of iterations `iters`, the `converged` flag, and the
final `grad_norm`. For the Rosenbrock start above the solver converges to the
valley floor at \\( (1, 1) \\) with `fval` near zero.

### Newton with analytic score and Hessian

```rust
use ndarray::{array, Array1, Array2};
use solow_optimize::newton_stationary;

// f(x) = (x0 - 3)^2 + 2 (x1 + 1)^2, stationary point (minimum) at (3, -1).
let fgh = |x: &Array1<f64>| {
    let f = (x[0] - 3.0).powi(2) + 2.0 * (x[1] + 1.0).powi(2);
    let g = array![2.0 * (x[0] - 3.0), 4.0 * (x[1] + 1.0)];
    let h = array![[2.0, 0.0], [0.0, 4.0]];
    (f, g, h)
};

let res = newton_stationary(&array![0.0, 0.0], fgh, 50, 1e-10).unwrap();

assert!(res.converged);
println!("solution = {:?}", res.x); // close to [3.0, -1.0]
```

The closure passed to `newton_stationary` returns the triple `(value, gradient,
hessian)`; the solver applies the damped Newton update until the gradient norm
(or the Newton step) is small enough. This is exactly the pattern Solow's
likelihood-based estimators use internally, swapping in the model's score and
information matrix.

> *Illustrative output.* The printed values above describe where each solver
> lands for these starting points; exact iteration counts and residual norms
> depend on the platform's floating-point arithmetic.

If you only have the objective, build the gradient and Hessian with
`approx_fprime` and `approx_hess` and feed them to whichever solver you prefer.

## Module reference

**Functions**

| Name | Description |
| --- | --- |
| `newton_stationary` | Damped Newton iteration driving the gradient to zero from a `(value, gradient, hessian)` closure; falls back to a least-squares solve when the Hessian is singular. |
| `minimize_bfgs` | BFGS quasi-Newton minimizer with Armijo backtracking line search, taking separate value and gradient closures. |
| `approx_fprime` | Central-difference gradient of a scalar objective. |
| `approx_hess` | Central-difference Hessian of a scalar objective (symmetrized). |

**Results**

| Name | Description |
| --- | --- |
| `OptimizeResult` | Outcome of an optimization run: fields `x`, `fval`, `iters`, `converged`, `grad_norm`. |

Full API: see the generated rustdoc for `solow-optimize`.

## References

- Nocedal, J., and Wright, S. J. *Numerical Optimization*, 2nd ed. Springer,
  2006.
- Fletcher, R. *Practical Methods of Optimization*, 2nd ed. Wiley, 1987.
- Broyden, C. G. "The Convergence of a Class of Double-Rank Minimization
  Algorithms." *Journal of the Institute of Mathematics and Its Applications*,
  6(1):76–90, 1970.
- Dennis, J. E., and Schnabel, R. B. *Numerical Methods for Unconstrained
  Optimization and Nonlinear Equations*. SIAM, 1996.

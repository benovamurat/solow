//! Optimizer convergence.
//!
//! Minimizes two standard test functions with the real solvers from
//! `solow-optimize` and plots how fast each drives the gradient to zero:
//!
//! - the **Rosenbrock** banana, minimized with [`minimize_bfgs`] (gradient only,
//!   the gradient supplied by finite differences via [`approx_fprime`]);
//! - a convex **quadratic**, minimized with [`newton_stationary`] (analytic
//!   gradient and Hessian).
//!
//! Neither optimizer exposes a per-iteration trace, so the convergence curve is
//! built honestly: the solver is re-run from the same start with an increasing
//! iteration cap `k = 0, 1, 2, …`, and the final gradient norm reported by the
//! `OptimizeResult` at each cap is recorded. Plotting on a log y-scale turns the
//! convergence *rate* into a slope: BFGS is roughly linear (a straight-ish line),
//! Newton on a quadratic reaches the minimum in a single step.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin optim_convergence

use ndarray::{array, Array1, Array2};
use solow_optimize::{approx_fprime, minimize_bfgs, newton_stationary};
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker, Scale};

#[path = "../common.rs"]
mod common;

/// Rosenbrock f(x) = (1 - x0)^2 + 100 (x1 - x0^2)^2, global min 0 at (1, 1).
fn rosenbrock(x: &Array1<f64>) -> f64 {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
}

fn main() {
    // ---- Curve 1: BFGS on Rosenbrock, gradient from finite differences ------
    let start_rb = array![-1.2, 1.0];
    let gtol = 1e-10;
    let rb_caps: Vec<usize> = (0..=40).collect();
    let mut rb_iter: Vec<f64> = Vec::new();
    let mut rb_gnorm: Vec<f64> = Vec::new();
    for &k in &rb_caps {
        let f = rosenbrock;
        let g = |x: &Array1<f64>| approx_fprime(x, f);
        let r = minimize_bfgs(&start_rb, f, g, k, gtol).unwrap();
        rb_iter.push(r.iters as f64);
        // Clamp the floor so the log axis stays finite once we hit machine zero.
        rb_gnorm.push(r.grad_norm.max(1e-16));
    }
    let rb_final = minimize_bfgs(
        &start_rb,
        rosenbrock,
        |x: &Array1<f64>| approx_fprime(x, rosenbrock),
        1000,
        gtol,
    )
    .unwrap();

    // ---- Curve 2: Newton on a convex quadratic, analytic g and H ------------
    // f(x) = (x0 - 3)^2 + 2 (x1 + 1)^2, min 0 at (3, -1).
    let fgh = |x: &Array1<f64>| {
        let f = (x[0] - 3.0).powi(2) + 2.0 * (x[1] + 1.0).powi(2);
        let g = array![2.0 * (x[0] - 3.0), 4.0 * (x[1] + 1.0)];
        let h: Array2<f64> = array![[2.0, 0.0], [0.0, 4.0]];
        (f, g, h)
    };
    let start_q = array![0.0, 0.0];
    let q_caps: Vec<usize> = (0..=6).collect();
    let mut q_iter: Vec<f64> = Vec::new();
    let mut q_gnorm: Vec<f64> = Vec::new();
    for &k in &q_caps {
        let r = newton_stationary(&start_q, fgh, k, gtol).unwrap();
        q_iter.push(r.iters as f64);
        q_gnorm.push(r.grad_norm.max(1e-16));
    }
    let q_final = newton_stationary(&start_q, fgh, 50, gtol).unwrap();

    // ---- Printed results (the real OptimizeResult fields) -------------------
    println!("Optimizer convergence");
    println!("=====================================================");
    println!("BFGS on Rosenbrock  (start = [-1.2, 1.0])");
    println!("  x* = [{:.6}, {:.6}]", rb_final.x[0], rb_final.x[1]);
    println!("  f(x*)      = {:.6e}", rb_final.fval);
    println!("  grad_norm  = {:.6e}", rb_final.grad_norm);
    println!("  iters      = {}", rb_final.iters);
    println!("  converged  = {}", rb_final.converged);
    println!("-----------------------------------------------------");
    println!("Newton on quadratic (start = [0.0, 0.0])");
    println!("  x* = [{:.6}, {:.6}]", q_final.x[0], q_final.x[1]);
    println!("  f(x*)      = {:.6e}", q_final.fval);
    println!("  grad_norm  = {:.6e}", q_final.grad_norm);
    println!("  iters      = {}", q_final.iters);
    println!("  converged  = {}", q_final.converged);
    println!("-----------------------------------------------------");
    println!("Gradient norm vs iteration cap (BFGS / Rosenbrock):");
    for k in (0..=40).step_by(5) {
        println!(
            "  cap {:>2} -> iters {:>2}  grad_norm = {:.4e}",
            rb_caps[k], rb_iter[k] as usize, rb_gnorm[k]
        );
    }

    // ---- Plot: gradient norm vs iteration, log y-scale ----------------------
    let mut fig = Figure::new(760, 520);
    {
        let ax = fig.axes();
        ax.set_title("Optimizer convergence: gradient norm vs iteration")
            .set_xlabel("iteration")
            .set_ylabel("gradient norm  (log scale)")
            .set_yscale(Scale::Log)
            .set_grid(true);
        ax.line(
            &rb_iter,
            &rb_gnorm,
            Color::cycle(0),
            2.0,
            LineStyle::Solid,
            Marker::Circle,
            0.9,
            Some("BFGS, Rosenbrock"),
        );
        ax.line(
            &q_iter,
            &q_gnorm,
            Color::RED,
            2.0,
            LineStyle::Dashed,
            Marker::Square,
            0.9,
            Some("Newton, quadratic"),
        );
        ax.legend(LegendLoc::UpperRight);
    }

    let out = common::img_path("optim_convergence.svg");
    fig.save_svg(&out).expect("write optim_convergence.svg");
    eprintln!("wrote {}", out.display());
}

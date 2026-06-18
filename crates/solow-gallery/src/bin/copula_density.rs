//! Clayton copula dependence.
//!
//! A copula couples uniform margins into a joint law whose *only* content is
//! the dependence structure. The Clayton family captures lower-tail dependence:
//! small values of one coordinate strongly pull the other down, while the upper
//! tail is comparatively loose. This example
//!
//!   * pins the dependence parameter `theta` from a target Kendall's tau,
//!   * evaluates the closed-form copula density `c(u, v)` on a grid over the
//!     unit square and renders it as a heatmap,
//!   * draws a deterministic pseudo-random sample from the copula by conditional
//!     inversion, and recovers the rank correlations from that sample to confirm
//!     they match the analytic `tau`.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin copula_density

use solow_copula::{kendalls_tau, spearmans_rho, ClaytonCopula};
use solow_viz::{Color, Colormap, Figure, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Pin theta from a target Kendall's tau ------------------------------
    // Clayton's analytic map is tau = theta / (theta + 2); invert it.
    let target_tau = 0.5;
    let theta = ClaytonCopula::theta_from_tau(target_tau);
    let cop = ClaytonCopula::new(theta);

    println!("Clayton copula");
    println!("  target Kendall's tau : {:.4}", target_tau);
    println!("  implied theta        : {:.6}", theta);
    println!("  analytic tau(theta)  : {:.6}", cop.tau());

    // A few density landmarks (note the heavy lower-left corner).
    println!("  density landmarks c(u, v):");
    for &(u, v) in &[(0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.1, 0.9)] {
        println!("    c({u:.1}, {v:.1}) = {:.4}", cop.pdf(u, v));
    }

    // --- Evaluate the copula density on a grid over the unit square ---------
    // Row 0 of the grid is drawn at the TOP of the extent, so we let v run from
    // high to low as the row index grows. We clip a hair inside the open square
    // because the Clayton density diverges at the (0, 0) corner.
    let n = 80usize;
    let eps = 1.0 / (n as f64 + 1.0); // grid spans (eps, 1 - eps)
    let coord = |k: usize| eps + (1.0 - 2.0 * eps) * (k as f64) / ((n - 1) as f64);

    let mut grid: Vec<Vec<f64>> = Vec::with_capacity(n);
    for r in 0..n {
        let v = coord(n - 1 - r); // top row -> largest v
        let mut row = Vec::with_capacity(n);
        for c in 0..n {
            let u = coord(c);
            // Log density compresses the corner spike into a readable range.
            row.push(cop.pdf(u, v).ln());
        }
        grid.push(row);
    }

    let (mut gmin, mut gmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for row in &grid {
        for &z in row {
            gmin = gmin.min(z);
            gmax = gmax.max(z);
        }
    }
    println!(
        "  grid: {n} x {n} over (u, v) in ({eps:.4}, {:.4})",
        1.0 - eps
    );
    println!("  log-density range    : [{:.4}, {:.4}]", gmin, gmax);

    // --- Draw a reproducible sample by conditional inversion ----------------
    // For Clayton, U ~ Unif and V | U is inverted in closed form:
    //   v = ( u^{-theta} * (w^{-theta/(1+theta)} - 1) + 1 )^{-1/theta},
    // with u, w independent uniforms. We use the deterministic SplitMix64 RNG
    // so the sample (and the recovered rank correlations) never change.
    let mut rng = common::Rng::new(0xC0FFEE_u64);
    let m = 600usize;
    let mut us = Vec::with_capacity(m);
    let mut vs = Vec::with_capacity(m);
    for _ in 0..m {
        let u = rng.uniform().clamp(1e-9, 1.0 - 1e-9);
        let w = rng.uniform().clamp(1e-9, 1.0 - 1e-9);
        let a = w.powf(-theta / (1.0 + theta)) - 1.0;
        let v = (u.powf(-theta) * a + 1.0).powf(-1.0 / theta);
        us.push(u);
        vs.push(v.clamp(1e-9, 1.0 - 1e-9));
    }

    let tau_hat = kendalls_tau(&us, &vs);
    let rho_hat = spearmans_rho(&us, &vs);
    println!("  sample size          : {m}");
    println!(
        "  sample Kendall's tau : {:.4}  (analytic {:.4})",
        tau_hat,
        cop.tau()
    );
    println!("  sample Spearman rho  : {:.4}", rho_hat);

    // --- Render: density heatmap with the sample scattered on top -----------
    let mut fig = Figure::new(720, 640);
    {
        let ax = fig.axes();
        ax.set_title("Clayton copula density (tau = 0.5)")
            .set_xlabel("u")
            .set_ylabel("v");
        ax.heatmap(&grid, Colormap::Viridis, (0.0, 1.0, 0.0, 1.0), true);
        ax.scatter_full(
            &us,
            &vs,
            Color::WHITE,
            2.0,
            Marker::Circle,
            0.55,
            Some("sample"),
        );
        // The independence diagonal u = v, where Clayton concentrates mass.
        ax.line(
            &[0.0, 1.0],
            &[0.0, 1.0],
            Color::RED,
            1.5,
            LineStyle::Dashed,
            Marker::None,
            0.9,
            Some("u = v"),
        );
        ax.set_xlim(0.0, 1.0).set_ylim(0.0, 1.0);
    }

    let out = common::img_path("copula_density.svg");
    fig.save_svg(&out).expect("write copula_density.svg");
    eprintln!("wrote {}", out.display());
}

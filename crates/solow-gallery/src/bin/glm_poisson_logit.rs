//! Generalized linear models: Poisson regression and logistic (Logit) regression.
//!
//! Two GLMs on deterministic synthetic data:
//!
//!   * Poisson count model with a log link: `log E[y] = b0 + b1 x`.
//!   * Bernoulli/Binomial model with a logit link: `logit P(y=1) = b0 + b1 x`.
//!
//! Each model's canonical summary is printed. The saved figure has two panels:
//! left, Poisson observed counts with the fitted mean curve; right, the binary
//! outcomes with the fitted success-probability curve.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin glm_poisson_logit

use ndarray::{Array1, Array2};
use solow_core::tools::{add_constant, HasConstant};
use solow_glm::{Family, Glm};
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    let mut rng = common::Rng::new(101);
    let n = 80usize;

    // x spans a modest range so both mean curves are visibly nonlinear.
    let x_raw: Vec<f64> = (0..n)
        .map(|i| -2.0 + 4.0 * i as f64 / (n - 1) as f64)
        .collect();
    let x = Array2::from_shape_vec((n, 1), x_raw.clone()).unwrap();
    let design = add_constant(&x, true, HasConstant::Add).unwrap();

    // --- Poisson: log mean = 0.4 + 0.7 x ------------------------------------
    let (pb0, pb1) = (0.4, 0.7);
    let y_pois: Vec<f64> = x_raw
        .iter()
        .map(|&xi| rng.poisson((pb0 + pb1 * xi).exp()))
        .collect();
    let pois = Glm::new(
        Array1::from(y_pois.clone()),
        design.clone(),
        Family::Poisson,
    )
    .unwrap()
    .fit()
    .unwrap();

    println!("=== Poisson regression (log link) ===");
    println!("{}", pois.summary_titled("counts", Some(&["const", "x"])));

    // --- Logit: logit P(y=1) = -0.3 + 1.6 x ---------------------------------
    let (lb0, lb1) = (-0.3, 1.6);
    let y_bin: Vec<f64> = x_raw
        .iter()
        .map(|&xi| {
            let p = 1.0 / (1.0 + (-(lb0 + lb1 * xi)).exp());
            rng.bernoulli(p)
        })
        .collect();
    let logit = Glm::new(
        Array1::from(y_bin.clone()),
        design.clone(),
        Family::Binomial,
    )
    .unwrap()
    .fit()
    .unwrap();

    println!("\n=== Logistic regression (logit link) ===");
    println!("{}", logit.summary_titled("y", Some(&["const", "x"])));

    // --- Fitted curves on a dense grid --------------------------------------
    let m = 200usize;
    let grid: Vec<f64> = (0..m)
        .map(|i| -2.0 + 4.0 * i as f64 / (m - 1) as f64)
        .collect();
    let pois_curve: Vec<f64> = grid
        .iter()
        .map(|&xi| (pois.params[0] + pois.params[1] * xi).exp())
        .collect();
    let logit_curve: Vec<f64> = grid
        .iter()
        .map(|&xi| {
            let eta = logit.params[0] + logit.params[1] * xi;
            1.0 / (1.0 + (-eta).exp())
        })
        .collect();

    // --- Two-panel figure ---------------------------------------------------
    let mut fig = Figure::subplots(960, 460, 1, 2);
    fig.suptitle("GLM fitted mean vs observed");
    {
        let ax = fig.ax_at(0, 0).unwrap();
        ax.set_title("Poisson: counts & fitted mean")
            .set_xlabel("x")
            .set_ylabel("y (count)")
            .set_grid(true);
        ax.scatter_full(
            &x_raw,
            &y_pois,
            Color::cycle(0),
            4.0,
            Marker::Circle,
            0.7,
            Some("observed"),
        );
        ax.line(
            &grid,
            &pois_curve,
            Color::RED,
            2.5,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("fitted E[y]"),
        );
        ax.legend(LegendLoc::UpperLeft);
    }
    {
        let ax = fig.ax_at(0, 1).unwrap();
        ax.set_title("Logit: outcomes & fitted P(y=1)")
            .set_xlabel("x")
            .set_ylabel("y / probability")
            .set_grid(true);
        ax.scatter_full(
            &x_raw,
            &y_bin,
            Color::cycle(0),
            4.0,
            Marker::Circle,
            0.6,
            Some("observed (0/1)"),
        );
        ax.line(
            &grid,
            &logit_curve,
            Color::RED,
            2.5,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("fitted P(y=1)"),
        );
        ax.legend(LegendLoc::UpperLeft);
    }

    let out = common::img_path("glm_poisson_logit.svg");
    fig.save_svg(&out).expect("write glm_poisson_logit.svg");
    eprintln!("wrote {}", out.display());
}

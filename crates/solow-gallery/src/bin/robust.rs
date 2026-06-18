//! Robust linear regression (RLM) via M-estimation.
//!
//! Clean linear data is contaminated with a handful of gross outliers. OLS is
//! pulled toward the outliers; an M-estimator with Huber's loss downweights
//! them and recovers the underlying line. The plot shows the data (outliers
//! highlighted) with the OLS and robust fits overlaid.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin robust

use ndarray::{Array1, Array2};
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;
use solow_robust::norms::HuberT;
use solow_robust::Rlm;
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    let mut rng = common::Rng::new(2024);
    let n = 40usize;
    let (b0, b1) = (1.0, 2.0);

    let x_raw: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
    let mut y_vec: Vec<f64> = x_raw
        .iter()
        .map(|&xi| b0 + b1 * xi + 0.6 * rng.normal())
        .collect();

    // Inject a few large vertical outliers.
    let outlier_idx = [5usize, 12, 27, 33];
    for &i in &outlier_idx {
        y_vec[i] += 14.0;
    }
    let is_outlier: Vec<bool> = (0..n).map(|i| outlier_idx.contains(&i)).collect();

    let x = Array2::from_shape_vec((n, 1), x_raw.clone()).unwrap();
    let y = Array1::from(y_vec.clone());
    let design = add_constant(&x, true, HasConstant::Add).unwrap();

    // OLS (sensitive) vs Huber M-estimator (robust).
    let ols = LinearModel::ols(y.clone(), design.clone())
        .unwrap()
        .fit()
        .unwrap();
    let rlm = Rlm::new(y.clone(), design.clone(), HuberT::default())
        .unwrap()
        .fit()
        .unwrap();

    // --- Printed comparison --------------------------------------------------
    println!("Robust linear model (RLM) with Huber's T norm");
    println!("True coefficients:  const = {b0:.3}   x = {b1:.3}\n");
    println!("{:<22}{:>12}{:>12}", "", "const", "x");
    println!(
        "{:<22}{:>12.4}{:>12.4}",
        "OLS params", ols.params[0], ols.params[1]
    );
    println!(
        "{:<22}{:>12.4}{:>12.4}",
        "RLM params (Huber)", rlm.params[0], rlm.params[1]
    );
    println!(
        "{:<22}{:>12.4}{:>12.4}",
        "RLM std err", rlm.bse[0], rlm.bse[1]
    );
    println!(
        "{:<22}{:>12.4}{:>12.4}",
        "RLM z-value", rlm.tvalues[0], rlm.tvalues[1]
    );
    println!(
        "{:<22}{:>12.4}{:>12.4}",
        "RLM P>|z|", rlm.pvalues[0], rlm.pvalues[1]
    );
    println!(
        "\nRobust scale estimate: {:.4}   iterations: {}   converged: {}",
        rlm.scale, rlm.iteration, rlm.converged
    );
    println!(
        "OLS is dragged toward the {} outliers; RLM stays on the true line.",
        outlier_idx.len()
    );

    // --- Plot ----------------------------------------------------------------
    let xline = [x_raw[0], x_raw[n - 1]];
    let ols_line = [
        ols.params[0] + ols.params[1] * xline[0],
        ols.params[0] + ols.params[1] * xline[1],
    ];
    let rlm_line = [
        rlm.params[0] + rlm.params[1] * xline[0],
        rlm.params[0] + rlm.params[1] * xline[1],
    ];

    // Split points into inliers vs outliers for distinct styling.
    let mut inx = Vec::new();
    let mut iny = Vec::new();
    let mut outx = Vec::new();
    let mut outy = Vec::new();
    for i in 0..n {
        if is_outlier[i] {
            outx.push(x_raw[i]);
            outy.push(y_vec[i]);
        } else {
            inx.push(x_raw[i]);
            iny.push(y_vec[i]);
        }
    }

    let mut fig = Figure::new(760, 520);
    {
        let ax = fig.axes();
        ax.set_title("Robust regression resists outliers")
            .set_xlabel("x")
            .set_ylabel("y")
            .set_grid(true);
        ax.scatter_full(
            &inx,
            &iny,
            Color::cycle(0),
            4.0,
            Marker::Circle,
            0.8,
            Some("inliers"),
        );
        ax.scatter_full(
            &outx,
            &outy,
            Color::cycle(3),
            6.0,
            Marker::Cross,
            1.0,
            Some("outliers"),
        );
        ax.line(
            &xline,
            &ols_line,
            Color::cycle(1),
            2.5,
            LineStyle::Dashed,
            Marker::None,
            1.0,
            Some("OLS fit"),
        );
        ax.line(
            &xline,
            &rlm_line,
            Color::RED,
            2.5,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("RLM (Huber)"),
        );
        ax.legend(LegendLoc::UpperLeft);
    }

    let out = common::img_path("robust.svg");
    fig.save_svg(&out).expect("write robust.svg");
    eprintln!("wrote {}", out.display());
}

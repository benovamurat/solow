//! Ordinary least squares (OLS).
//!
//! Fits a simple linear model `y = b0 + b1 * x + e` on deterministic synthetic
//! data, prints the full canonical results summary, and saves a scatter plot of
//! the data overlaid with the fitted regression line.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin ols

use ndarray::{Array1, Array2};
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Example data: y = 2 + 0.5 x + N(0, 1.2), x = 0, 1, ..., 49 ---------
    let mut rng = common::Rng::new(20240617);
    let n = 50usize;
    let beta0 = 2.0;
    let beta1 = 0.5;

    let x_raw: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y_vec: Vec<f64> = x_raw
        .iter()
        .map(|&xi| beta0 + beta1 * xi + 1.2 * rng.normal())
        .collect();

    let x = Array2::from_shape_vec((n, 1), x_raw.clone()).unwrap();
    let y = Array1::from(y_vec.clone());

    // Add an intercept column, then fit by ordinary least squares.
    let design = add_constant(&x, true, HasConstant::Add).unwrap();
    let res = LinearModel::ols(y, design).unwrap().fit().unwrap();

    // --- Printed results summary (the canonical reference layout) -----------
    println!("{}", res.summary_titled("y", "OLS", Some(&["const", "x"])));

    // --- Scatter of observations + the fitted regression line ---------------
    let b0 = res.params[0];
    let b1 = res.params[1];
    let xs_line = [0.0, (n - 1) as f64];
    let ys_line = [b0, b0 + b1 * (n - 1) as f64];

    let mut fig = Figure::new(760, 520);
    {
        let ax = fig.axes();
        ax.set_title("OLS fit: y = b0 + b1 x")
            .set_xlabel("x")
            .set_ylabel("y")
            .set_grid(true);
        ax.scatter_full(
            &x_raw,
            &y_vec,
            Color::cycle(0),
            4.0,
            Marker::Circle,
            0.85,
            Some("observed"),
        );
        ax.line(
            &xs_line,
            &ys_line,
            Color::RED,
            2.5,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("OLS fit"),
        );
        ax.legend(LegendLoc::UpperLeft);
    }

    let out = common::img_path("ols.svg");
    fig.save_svg(&out).expect("write ols.svg");
    eprintln!("wrote {}", out.display());
}

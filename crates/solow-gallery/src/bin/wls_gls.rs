//! Weighted and generalized least squares (WLS / GLS).
//!
//! Builds data with deliberate heteroskedasticity — the error variance grows
//! with `x` — then contrasts three fits:
//!
//!   * OLS, which ignores the non-constant variance;
//!   * WLS, with weights proportional to the inverse error variance;
//!   * GLS, with the same information supplied as a diagonal covariance Sigma.
//!
//! WLS and GLS recover identical coefficients (GLS with a diagonal Sigma *is*
//! WLS). The plot overlays the OLS and WLS fitted lines on the raw scatter.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin wls_gls

use ndarray::{Array1, Array2};
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Heteroskedastic data: Var(e_i) grows linearly with x_i -------------
    let mut rng = common::Rng::new(7);
    let n = 60usize;
    let beta0 = 1.0;
    let beta1 = 0.8;

    let x_raw: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.5).collect();
    // Standard deviation of the noise scales with x => later points are noisier.
    let sd: Vec<f64> = x_raw.iter().map(|&xi| 0.35 * xi).collect();
    let y_vec: Vec<f64> = x_raw
        .iter()
        .zip(&sd)
        .map(|(&xi, &si)| beta0 + beta1 * xi + si * rng.normal())
        .collect();

    let x = Array2::from_shape_vec((n, 1), x_raw.clone()).unwrap();
    let y = Array1::from(y_vec.clone());
    let design = add_constant(&x, true, HasConstant::Add).unwrap();

    // Weights proportional to 1 / variance; Sigma is the diagonal variance matrix.
    let weights = Array1::from(sd.iter().map(|&s| 1.0 / (s * s)).collect::<Vec<_>>());
    let mut sigma = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        sigma[[i, i]] = sd[i] * sd[i];
    }

    let ols = LinearModel::ols(y.clone(), design.clone())
        .unwrap()
        .fit()
        .unwrap();
    let wls = LinearModel::wls(y.clone(), design.clone(), weights.clone())
        .unwrap()
        .fit()
        .unwrap();
    let gls = LinearModel::gls(y.clone(), design.clone(), &sigma)
        .unwrap()
        .fit()
        .unwrap();

    // --- Printed summaries ---------------------------------------------------
    println!("=== OLS (ignores heteroskedasticity) ===");
    println!("{}", ols.summary_titled("y", "OLS", Some(&["const", "x"])));
    println!("\n=== WLS (weights = 1 / variance) ===");
    println!("{}", wls.summary_titled("y", "WLS", Some(&["const", "x"])));
    println!(
        "\nGLS with diagonal Sigma reproduces the WLS coefficients exactly:\n  \
         WLS params = [{:.6}, {:.6}]\n  GLS params = [{:.6}, {:.6}]",
        wls.params[0], wls.params[1], gls.params[0], gls.params[1]
    );

    // --- Plot: scatter + OLS line + WLS line --------------------------------
    let xline = [x_raw[0], x_raw[n - 1]];
    let ols_line = [
        ols.params[0] + ols.params[1] * xline[0],
        ols.params[0] + ols.params[1] * xline[1],
    ];
    let wls_line = [
        wls.params[0] + wls.params[1] * xline[0],
        wls.params[0] + wls.params[1] * xline[1],
    ];

    let mut fig = Figure::new(760, 520);
    {
        let ax = fig.axes();
        ax.set_title("WLS vs OLS under heteroskedasticity")
            .set_xlabel("x")
            .set_ylabel("y")
            .set_grid(true);
        ax.scatter_full(
            &x_raw,
            &y_vec,
            Color::cycle(0),
            4.0,
            Marker::Circle,
            0.75,
            Some("observed"),
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
            &wls_line,
            Color::RED,
            2.5,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("WLS fit"),
        );
        ax.legend(LegendLoc::UpperLeft);
    }

    let out = common::img_path("wls_gls.svg");
    fig.save_svg(&out).expect("write wls_gls.svg");
    eprintln!("wrote {}", out.display());
}

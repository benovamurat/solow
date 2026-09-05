//! Time-series analysis: ACF/PACF, an autoregressive (AutoReg) fit, and a
//! classical seasonal decomposition.
//!
//! A deterministic AR(2) process with a superimposed seasonal cycle is built,
//! then:
//!
//!   * its sample ACF and PACF are computed and drawn as stem plots with
//!     approximate +/- 1.96/sqrt(n) significance bands;
//!   * an AutoReg(2) model is fitted and its summary-style table printed;
//!   * an additive seasonal decomposition splits the series into
//!     trend / seasonal / residual components.
//!
//! The saved figure has four panels: ACF, PACF, observed-vs-fitted (AR), and
//! the seasonal component.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin time_series

use ndarray::Array1;
use solow_tsa::{acf, pacf, seasonal_decompose, AutoReg, PacfMethod, SeasonalModel, Trend};
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Build an AR(2) series plus a 12-period seasonal cycle --------------
    let mut rng = common::Rng::new(424242);
    let n = 240usize;
    let period = 12usize;
    let (phi1, phi2) = (0.6, -0.3);

    let mut y = vec![0.0f64; n];
    for t in 0..n {
        let ar = if t >= 2 {
            phi1 * y[t - 1] + phi2 * y[t - 2]
        } else {
            0.0
        };
        let seasonal = 4.0 * (std::f64::consts::TAU * (t % period) as f64 / period as f64).sin();
        y[t] = ar + seasonal + 0.8 * rng.normal();
    }
    let series = Array1::from(y.clone());

    // --- ACF / PACF ----------------------------------------------------------
    let nlags = 24usize;
    let acf_vals = acf(&series, nlags, false).unwrap();
    let pacf_vals = pacf(&series, nlags, PacfMethod::YuleWalker).unwrap();
    let conf = 1.96 / (n as f64).sqrt();
    println!(
        "Sample ACF (lags 0..5): {:?}",
        &acf_vals.as_slice().unwrap()[..6]
    );
    println!(
        "Sample PACF (lags 0..5): {:?}",
        &pacf_vals.as_slice().unwrap()[..6]
    );
    println!("Approx 95% band: +/- {conf:.4}\n");

    // --- AutoReg(2) fit ------------------------------------------------------
    let ar = AutoReg::new(series.clone(), 2, Trend::C)
        .unwrap()
        .fit()
        .unwrap();
    println!("AutoReg(2) with constant trend");
    println!(
        "{:<14}{:>12}{:>12}{:>12}{:>12}",
        "param", "coef", "std err", "z", "P>|z|"
    );
    let names = ["const", "y.L1", "y.L2"];
    for i in 0..ar.params.len() {
        println!(
            "{:<14}{:>12.4}{:>12.4}{:>12.4}{:>12.4}",
            names.get(i).copied().unwrap_or("?"),
            ar.params[i],
            ar.bse[i],
            ar.tvalues[i],
            ar.pvalues[i],
        );
    }
    println!(
        "sigma2 = {:.4}   llf = {:.3}   aic = {:.3}   bic = {:.3}   nobs = {}",
        ar.sigma2, ar.llf, ar.aic, ar.bic, ar.nobs
    );

    // --- Seasonal decomposition ---------------------------------------------
    let dec = seasonal_decompose(&series, period, SeasonalModel::Additive).unwrap();
    let n_trend = dec.trend.iter().filter(|v| v.is_finite()).count();
    println!("\nAdditive decomposition: period = {period}, finite trend points = {n_trend}");

    // --- Four-panel figure ---------------------------------------------------
    let lags: Vec<f64> = (0..=nlags).map(|i| i as f64).collect();
    let mut fig = Figure::subplots(980, 760, 2, 2);
    fig.suptitle("Time-series diagnostics");

    // ACF stem.
    {
        let ax = fig.ax_at(0, 0).unwrap();
        ax.set_title("ACF")
            .set_xlabel("lag")
            .set_ylabel("acf")
            .set_grid(true);
        stem(ax, &lags, acf_vals.as_slice().unwrap(), Color::cycle(0));
        ax.axhline(conf, Color::GRAY, LineStyle::Dashed);
        ax.axhline(-conf, Color::GRAY, LineStyle::Dashed);
        ax.axhline(0.0, Color::BLACK, LineStyle::Solid);
    }
    // PACF stem.
    {
        let ax = fig.ax_at(0, 1).unwrap();
        ax.set_title("PACF")
            .set_xlabel("lag")
            .set_ylabel("pacf")
            .set_grid(true);
        stem(ax, &lags, pacf_vals.as_slice().unwrap(), Color::cycle(1));
        ax.axhline(conf, Color::GRAY, LineStyle::Dashed);
        ax.axhline(-conf, Color::GRAY, LineStyle::Dashed);
        ax.axhline(0.0, Color::BLACK, LineStyle::Solid);
    }
    // Observed vs AR-fitted (last 80 points so the overlay is legible).
    {
        let ax = fig.ax_at(1, 0).unwrap();
        ax.set_title("AutoReg(2): observed vs fitted")
            .set_xlabel("t")
            .set_ylabel("y")
            .set_grid(true);
        let start = n - ar.nobs; // fitted values start after the AR lags
        let t_obs: Vec<f64> = (start..n).map(|t| t as f64).collect();
        let obs: Vec<f64> = y[start..].to_vec();
        let fit: Vec<f64> = ar.fittedvalues.to_vec();
        ax.line(
            &t_obs,
            &obs,
            Color::cycle(0),
            1.4,
            LineStyle::Solid,
            Marker::None,
            0.9,
            Some("observed"),
        );
        ax.line(
            &t_obs,
            &fit,
            Color::RED,
            1.6,
            LineStyle::Dashed,
            Marker::None,
            0.9,
            Some("fitted"),
        );
        ax.legend(LegendLoc::UpperRight);
    }
    // Seasonal component (one or two cycles shown clearly across the series).
    {
        let ax = fig.ax_at(1, 1).unwrap();
        ax.set_title("Seasonal component")
            .set_xlabel("t")
            .set_ylabel("season")
            .set_grid(true);
        let t_all: Vec<f64> = (0..n).map(|t| t as f64).collect();
        ax.line(
            &t_all,
            dec.seasonal.as_slice().unwrap(),
            Color::cycle(2),
            1.6,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("seasonal"),
        );
    }

    let out = common::img_path("time_series.svg");
    fig.save_svg(&out).expect("write time_series.svg");
    eprintln!("wrote {}", out.display());
}

/// Draw a stem plot (vertical lines from 0 to each value plus a marker on top)
/// using the available step/line primitives.
fn stem(ax: &mut solow_viz::Axes, x: &[f64], y: &[f64], color: Color) {
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        ax.line(
            &[xi, xi],
            &[0.0, yi],
            color,
            1.4,
            LineStyle::Solid,
            Marker::None,
            0.9,
            None,
        );
    }
    ax.scatter_full(x, y, color, 3.0, Marker::Circle, 1.0, None);
}

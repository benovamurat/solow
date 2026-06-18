//! Generalized additive model (penalized B-spline smooth).
//!
//! Fits a Gaussian GAM with one penalized B-spline smooth term to a noisy
//! nonlinear signal by penalized iteratively reweighted least squares (P-IRLS)
//! at a fixed smoothing parameter, prints the key fitted quantities, and saves
//! a scatter of the data overlaid with the fitted smooth curve.
//!
//! Run with:
//!   cargo run -p solow-gallery --bin gam_smooth

use ndarray::Array1;
use solow_gam::GlmGam;
use solow_glm::Family;
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Example data: a smooth nonlinear signal plus deterministic noise ---
    //   x in [0, 1];  f(x) = sin(2 pi x) + 0.5 x;  y = f(x) + N(0, 0.25^2)
    let mut rng = common::Rng::new(20240617);
    let n = 120usize;
    let sigma = 0.25;

    let x = Array1::linspace(0.0, 1.0, n);
    let signal = x.mapv(|xi| (std::f64::consts::TAU * xi).sin() + 0.5 * xi);
    let y: Array1<f64> = Array1::from_iter((0..n).map(|i| signal[i] + sigma * rng.normal()));

    // --- Fit a penalized B-spline GAM (Gaussian, canonical identity link) ---
    //   df = 12 (=> 11 cubic basis columns), smoothing parameter alpha = 0.01.
    let df = 12usize;
    let degree = 3usize;
    let alpha = 0.01;
    let res = GlmGam::new(y.clone(), &x, df, degree, alpha, Family::Gaussian)
        .unwrap()
        .fit()
        .unwrap();

    // --- Printed results (real fitted quantities) ---------------------------
    println!("Generalized additive model (penalized B-spline smooth)");
    println!("------------------------------------------------------");
    println!("family            : Gaussian (identity link)");
    println!(
        "basis             : {} cubic B-spline columns (df=12)",
        res.dim_basis
    );
    println!("smoothing alpha   : {alpha}");
    println!(
        "converged         : {} (in {} iters)",
        res.converged, res.n_iter
    );
    println!("intercept         : {:.6}", res.intercept());
    println!("edf (total)       : {:.6}", res.edf_total);
    println!("df_resid          : {:.6}", res.df_resid);
    println!("scale (sigma^2)   : {:.6}", res.scale);
    println!("deviance          : {:.6}", res.deviance);
    println!("penalized deviance: {:.6}", res.penalized_deviance);

    // Residual standard deviation recovered from the fit, vs the true sigma.
    let rmse = (res.deviance / res.df_resid).sqrt();
    println!("resid. std (est)  : {rmse:.6}   (true sigma = {sigma})");

    // --- Scatter of observations + the fitted smooth curve ------------------
    // fittedvalues are evaluated at the (sorted) observations, so the curve is
    // drawn directly through x without extra basis evaluation.
    let x_vec: Vec<f64> = x.to_vec();
    let y_vec: Vec<f64> = y.to_vec();
    let fit_vec: Vec<f64> = res.fittedvalues.to_vec();

    let mut fig = Figure::new(760, 520);
    {
        let ax = fig.axes();
        ax.set_title("GAM smooth: penalized B-spline fit")
            .set_xlabel("x")
            .set_ylabel("y")
            .set_grid(true);
        ax.scatter_full(
            &x_vec,
            &y_vec,
            Color::cycle(0),
            4.0,
            Marker::Circle,
            0.65,
            Some("observed"),
        );
        ax.line(
            &x_vec,
            &fit_vec,
            Color::RED,
            2.5,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("GAM smooth"),
        );
        ax.legend(LegendLoc::UpperRight);
    }

    let out = common::img_path("gam_smooth.svg");
    fig.save_svg(&out).expect("write gam_smooth.svg");
    eprintln!("wrote {}", out.display());
}

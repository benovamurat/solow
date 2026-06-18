//! The formula interface.
//!
//! Instead of hand-assembling a design matrix and threading column names
//! through by hand, the formula API takes an R/patsy-style string plus a named
//! `DataFrame` and returns a fitted model whose coefficients are already
//! labeled. This example fits
//!
//!   y ~ x1 + x2 + C(group)
//!
//! mixing two numeric predictors with a categorical factor (expanded into
//! treatment-coded dummy columns), prints the labeled summary, then shows that
//! a Poisson GLM can be fit from a formula just as easily. The plot compares
//! the formula model's fitted values against the observed response.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin formula

use solow_fit::{ols, poisson, DataFrame};
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    let mut rng = common::Rng::new(555);
    let n = 90usize;

    // Two numeric predictors and a three-level categorical factor.
    let groups = ["A", "B", "C"];
    let group_effect = [0.0, 2.5, -1.5];
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let gi = i % 3;
        let a = rng.uniform() * 5.0;
        let b = rng.normal();
        // True model: y = 1 + 0.8 x1 - 1.2 x2 + group_effect + noise.
        let yi = 1.0 + 0.8 * a - 1.2 * b + group_effect[gi] + 0.5 * rng.normal();
        x1.push(a);
        x2.push(b);
        g.push(groups[gi].to_string());
        y.push(yi);
    }

    // Assemble a DataFrame with numeric and categorical columns.
    let mut df = DataFrame::new();
    df.add_numeric("y", y.clone());
    df.add_numeric("x1", x1.clone());
    df.add_numeric("x2", x2.clone());
    df.add_categorical("group", g.clone());

    // --- Fit from a formula --------------------------------------------------
    let fit = ols("y ~ x1 + x2 + C(group)", &df).unwrap();
    println!("OLS from formula: y ~ x1 + x2 + C(group)");
    println!("Design columns: {:?}\n", fit.names());
    println!("{}", fit.summary());

    // --- The same ergonomics for a Poisson GLM ------------------------------
    // Build a count response from a log-linear model and fit it by formula.
    let mut dfp = DataFrame::new();
    let mut counts = Vec::with_capacity(n);
    for i in 0..n {
        let mu = (0.3 + 0.25 * x1[i]).exp();
        counts.push(rng.poisson(mu));
    }
    dfp.add_numeric("count", counts);
    dfp.add_numeric("x1", x1.clone());
    let pfit = poisson("count ~ x1", &dfp).unwrap();
    println!("\nPoisson from formula: count ~ x1");
    println!("Design columns: {:?}", pfit.names());
    println!("{}", pfit.summary());

    // --- Plot: observed vs fitted for the OLS formula model -----------------
    let fitted = fit.results.fittedvalues.to_vec();
    // A reference y = x line for the observed-vs-fitted diagonal.
    let lo = y
        .iter()
        .chain(fitted.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let hi = y
        .iter()
        .chain(fitted.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut fig = Figure::new(720, 560);
    {
        let ax = fig.axes();
        ax.set_title("Formula OLS: observed vs fitted")
            .set_xlabel("fitted y-hat")
            .set_ylabel("observed y")
            .set_grid(true);
        ax.line(
            &[lo, hi],
            &[lo, hi],
            Color::GRAY,
            1.5,
            LineStyle::Dashed,
            Marker::None,
            1.0,
            Some("y = y-hat"),
        );
        ax.scatter_full(
            &fitted,
            &y,
            Color::cycle(0),
            4.0,
            Marker::Circle,
            0.8,
            Some("observations"),
        );
        ax.legend(LegendLoc::UpperLeft);
    }

    let out = common::img_path("formula.svg");
    fig.save_svg(&out).expect("write formula.svg");
    eprintln!("wrote {}", out.display());
}

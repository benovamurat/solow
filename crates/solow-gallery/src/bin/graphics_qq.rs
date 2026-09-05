//! Normal Q-Q plot (probability plot).
//!
//! Draws a reproducible standard-normal sample, builds a [`ProbPlot`] from the
//! `solow-graphics` crate, prints the computed quantiles and the fitted
//! reference line, and renders the sample quantiles against the theoretical
//! normal quantiles with an OLS reference line overlaid.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin graphics_qq

use solow_graphics::ProbPlot;
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Reproducible sample: 80 draws from N(0, 1) via SplitMix64 ----------
    let mut rng = common::Rng::new(20240617);
    let n = 80usize;
    let data: Vec<f64> = (0..n).map(|_| rng.normal()).collect();

    // --- Build the probability plot (standard-normal, Weibull positions) ----
    let pp = ProbPlot::new(&data);
    let theo = pp.theoretical_quantiles();
    let samp = pp.sample_quantiles();
    let line = pp.qqline_regression();
    let qline = pp.qqline_quartile();
    let sline = pp.qqline_standardized();

    // --- Printed results (real computed numbers) ----------------------------
    println!("Normal Q-Q plot (ProbPlot)");
    println!("==========================");
    println!("No. observations:        {}", pp.nobs());
    println!(
        "Theoretical quantiles:   [{:.4}, ..., {:.4}]",
        theo[0],
        theo[theo.len() - 1]
    );
    println!(
        "Sample quantiles:        [{:.4}, ..., {:.4}]",
        samp[0],
        samp[samp.len() - 1]
    );
    println!();
    println!("Reference lines (y = slope * x + intercept):");
    println!(
        "  regression (r):  slope = {:.4}   intercept = {:.4}",
        line.slope, line.intercept
    );
    println!(
        "  standardized (s): slope = {:.4}   intercept = {:.4}",
        sline.slope, sline.intercept
    );
    println!(
        "  quartile (q):    slope = {:.4}   intercept = {:.4}",
        qline.slope, qline.intercept
    );

    // --- Render the Q-Q plot ------------------------------------------------
    let theo_v: Vec<f64> = theo.to_vec();
    let samp_v: Vec<f64> = samp.to_vec();
    let lo = theo_v[0];
    let hi = theo_v[theo_v.len() - 1];
    let xs_line = [lo, hi];
    let ys_line = [
        line.slope * lo + line.intercept,
        line.slope * hi + line.intercept,
    ];

    let mut fig = Figure::new(760, 520);
    {
        let ax = fig.axes();
        ax.set_title("Normal Q-Q plot")
            .set_xlabel("Theoretical quantiles")
            .set_ylabel("Sample quantiles")
            .set_grid(true);
        ax.scatter_full(
            &theo_v,
            &samp_v,
            Color::cycle(0),
            4.0,
            Marker::Circle,
            0.85,
            Some("sample quantiles"),
        );
        ax.line(
            &xs_line,
            &ys_line,
            Color::RED,
            2.5,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("OLS reference line"),
        );
        ax.legend(LegendLoc::UpperLeft);
    }

    let out = common::img_path("graphics_qq.svg");
    fig.save_svg(&out).expect("write graphics_qq.svg");
    eprintln!("wrote {}", out.display());
}

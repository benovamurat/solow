//! Empirical-likelihood ratio profile for the mean.
//!
//! Builds a small reproducible sample, then sweeps a grid of hypothesized means
//! `mu` and evaluates the empirical-likelihood-ratio statistic `-2 logELR` at
//! each via [`DescStat::test_mean`]. The resulting profile curve is plotted
//! against `mu`, with a horizontal line at the chi-squared(1) 95% critical
//! value (3.841). Where the profile dips below that line is exactly the 95%
//! empirical-likelihood confidence interval for the mean; the two crossings are
//! cross-checked against [`DescStat::ci_mean`].
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin emplike_ratio

use solow_emplike::DescStat;
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Example data: 30 draws from N(5, 2^2), deterministic noise ----------
    let mut rng = common::Rng::new(20240618);
    let n = 30usize;
    let true_mu = 5.0;
    let true_sd = 2.0;

    let data: Vec<f64> = (0..n).map(|_| true_mu + true_sd * rng.normal()).collect();

    let sample_mean = data.iter().sum::<f64>() / n as f64;

    let d = DescStat::new(&data);

    // --- 95% EL confidence interval for the mean, two ways -------------------
    // Closed-form CI from the crate's gamma root-finder.
    let (ci_lo, ci_hi) = d.ci_mean(0.05);
    // The chi-squared(1) 95% threshold the profile is compared against.
    let crit = 3.841_458_820_694_124_f64; // chi2_ppf(0.95, 1)

    // --- Profile the -2 logELR statistic over a grid of hypothesized means ---
    let grid_lo = sample_mean - 2.2;
    let grid_hi = sample_mean + 2.2;
    let steps = 221usize;
    let mut mus = Vec::with_capacity(steps);
    let mut stats = Vec::with_capacity(steps);
    for k in 0..steps {
        let mu = grid_lo + (grid_hi - grid_lo) * (k as f64) / (steps as f64 - 1.0);
        let r = d.test_mean(mu);
        mus.push(mu);
        stats.push(r.stat);
    }

    // Statistic and p-value at a couple of reference points.
    let at_mean = d.test_mean(sample_mean);
    let at_true = d.test_mean(true_mu);

    // --- Printed results -----------------------------------------------------
    println!("Empirical-likelihood ratio for the mean");
    println!("========================================");
    println!("nobs              : {}", d.nobs());
    println!("sample mean       : {:.6}", sample_mean);
    println!();
    println!(
        "-2 logELR at sample mean : {:.6}  (p = {:.6})",
        at_mean.stat, at_mean.pvalue
    );
    println!(
        "-2 logELR at true mean 5 : {:.6}  (p = {:.6})",
        at_true.stat, at_true.pvalue
    );
    println!();
    println!("chi2(1) 95% threshold    : {:.6}", crit);
    println!("95% EL CI (ci_mean)      : [{:.6}, {:.6}]", ci_lo, ci_hi);
    println!(
        "-2 logELR at CI endpoints: {:.6}, {:.6}  (should equal threshold)",
        d.test_mean(ci_lo).stat,
        d.test_mean(ci_hi).stat
    );

    // --- Plot the profile curve with the threshold and CI band ---------------
    let mut fig = Figure::new(760, 520);
    {
        let ax = fig.axes();
        ax.set_title("Empirical-likelihood ratio profile for the mean")
            .set_xlabel("hypothesized mean  mu")
            .set_ylabel("-2 log ELR")
            .set_grid(true);

        // Shade the 95% EL confidence interval (where the profile is below crit).
        ax.axvspan(ci_lo, ci_hi, Color::BLUE, 0.08);

        // The profile curve.
        ax.line(
            &mus,
            &stats,
            Color::cycle(0),
            2.5,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("-2 log ELR(mu)"),
        );

        // chi2(1) 95% critical value: crossings define the 95% CI.
        ax.axhline(crit, Color::RED, LineStyle::Dashed);
        ax.annotate_styled(
            grid_lo,
            crit + 0.15,
            "chi2(1) 95% = 3.841",
            Color::RED,
            11.0,
        );

        // Mark the CI endpoints on the threshold line.
        ax.scatter_full(
            &[ci_lo, ci_hi],
            &[crit, crit],
            Color::RED,
            5.0,
            Marker::Circle,
            1.0,
            Some("95% EL CI endpoints"),
        );

        // Mark the sample mean, where the statistic is (numerically) zero.
        ax.scatter_full(
            &[sample_mean],
            &[at_mean.stat],
            Color::GREEN,
            6.0,
            Marker::Diamond,
            1.0,
            Some("sample mean"),
        );

        ax.set_ylim(0.0, 12.0);
        ax.legend(LegendLoc::UpperRight);
    }

    let out = common::img_path("emplike_ratio.svg");
    fig.save_svg(&out).expect("write emplike_ratio.svg");
    eprintln!("wrote {}", out.display());
}

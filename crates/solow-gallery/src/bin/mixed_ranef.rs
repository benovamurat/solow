//! Linear mixed effects: a random-intercept model.
//!
//! Builds grouped data where each group has its own intercept drawn around a
//! common mean, fits a random-intercept `MixedLm` by REML, prints the key
//! estimates, and plots each group's observations together with its fitted
//! line. Every group line shares the same fixed-effect slope but gets its own
//! intercept = (fixed intercept) + (group random intercept). The random
//! intercepts are *shrunk* toward zero relative to the per-group means — the
//! hallmark of partial pooling.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin mixed_ranef

use ndarray::{Array1, Array2};
use solow_mixed::{MixedLm, RemlMethod};
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Example data --------------------------------------------------------
    // True model: y = b0 + b1*x + u_g + e, with x in [0, 10], a shared slope
    // b1 = 1.5 and intercept b0 = 3.0, group random intercepts u_g ~ N(0, 2^2),
    // and residual noise e ~ N(0, 1.1^2). Six groups, eight observations each.
    let mut rng = common::Rng::new(20240617);
    let n_groups = 6usize;
    let per_group = 8usize;
    let b0_true = 3.0;
    let b1_true = 1.5;
    let sd_group = 2.0;
    let sd_resid = 1.1;

    // One random intercept per group, drawn once and reused within the group.
    let group_intercepts: Vec<f64> = (0..n_groups).map(|_| sd_group * rng.normal()).collect();

    let mut x_raw: Vec<f64> = Vec::new();
    let mut y_raw: Vec<f64> = Vec::new();
    let mut group_labels: Vec<i64> = Vec::new();
    for g in 0..n_groups {
        for k in 0..per_group {
            // Spread x across the range so each group line is identifiable.
            let xi = 0.5 + 9.0 * (k as f64) / (per_group as f64 - 1.0);
            let yi = b0_true + b1_true * xi + group_intercepts[g] + sd_resid * rng.normal();
            x_raw.push(xi);
            y_raw.push(yi);
            group_labels.push(g as i64);
        }
    }

    let n = x_raw.len();
    // Fixed-effects design: intercept column + the single covariate x.
    let mut exog = Array2::<f64>::ones((n, 2));
    for i in 0..n {
        exog[[i, 1]] = x_raw[i];
    }
    let endog = Array1::from(y_raw.clone());

    // --- Fit the random-intercept model by REML -----------------------------
    let res = MixedLm::new(endog, exog.clone(), &group_labels)
        .unwrap()
        .method(RemlMethod::Reml)
        .fit()
        .unwrap();

    let b0_hat = res.fe_params[0];
    let b1_hat = res.fe_params[1];

    // Best linear unbiased predictions of the group random intercepts. For a
    // random-intercept model the BLUP shrinks the group's mean residual toward
    // zero by the factor (n_g*psi)/(1 + n_g*psi):
    //     u_g_hat = (n_g*psi)/(1 + n_g*psi) * mean_g( y - X*beta ).
    // This shrinkage is exactly the partial pooling we want to visualize.
    let psi = res.psi;
    let mut group_mean_resid = vec![0.0f64; n_groups];
    let mut group_count = vec![0usize; n_groups];
    for i in 0..n {
        let g = group_labels[i] as usize;
        let fitted_fe = b0_hat + b1_hat * x_raw[i];
        group_mean_resid[g] += y_raw[i] - fitted_fe;
        group_count[g] += 1;
    }
    let blup: Vec<f64> = (0..n_groups)
        .map(|g| {
            let ng = group_count[g] as f64;
            let mean_r = group_mean_resid[g] / ng;
            (ng * psi) / (1.0 + ng * psi) * mean_r
        })
        .collect();

    // --- Printed results -----------------------------------------------------
    println!("Mixed Linear Model (random intercept), REML");
    println!("================================================");
    println!("No. Observations: {n}");
    println!("No. Groups:       {n_groups}");
    println!("Group size:       {per_group} (balanced)");
    println!();
    println!("Fixed effects:");
    println!(
        "  const   coef={:>8.4}   std err={:>7.4}   z={:>7.3}",
        res.fe_params[0],
        res.bse_fe[0],
        res.tvalues()[0]
    );
    println!(
        "  x       coef={:>8.4}   std err={:>7.4}   z={:>7.3}",
        res.fe_params[1],
        res.bse_fe[1],
        res.tvalues()[1]
    );
    println!();
    println!("Variance components:");
    println!("  Group Var (cov_re) = {:.4}", res.cov_re);
    println!("  Residual Var (scale) = {:.4}", res.scale);
    println!("  psi = cov_re/scale   = {:.4}", res.psi);
    println!("  REML log-likelihood  = {:.4}", res.llf);
    println!();
    println!("Predicted random intercepts (BLUP, shrunk toward 0):");
    for g in 0..n_groups {
        let raw_mean = group_mean_resid[g] / group_count[g] as f64;
        println!(
            "  group {g}:  raw mean resid = {:>7.4}   BLUP u_g = {:>7.4}",
            raw_mean, blup[g]
        );
    }

    // --- Plot: data colored by group + per-group fitted lines ---------------
    let mut fig = Figure::new(820, 560);
    {
        let ax = fig.axes();
        ax.set_title("Random-intercept mixed model: partial pooling across groups")
            .set_xlabel("x")
            .set_ylabel("y")
            .set_grid(true);

        let x_lo = 0.0;
        let x_hi = 10.0;
        for g in 0..n_groups {
            let color = Color::cycle(g);
            // Scatter this group's observations.
            let xs: Vec<f64> = (0..n)
                .filter(|&i| group_labels[i] as usize == g)
                .map(|i| x_raw[i])
                .collect();
            let ys: Vec<f64> = (0..n)
                .filter(|&i| group_labels[i] as usize == g)
                .map(|i| y_raw[i])
                .collect();
            ax.scatter_full(
                &xs,
                &ys,
                color,
                5.0,
                Marker::Circle,
                0.9,
                Some(&format!("group {g}")),
            );
            // Group fitted line: shared slope, group-specific intercept.
            let intercept_g = b0_hat + blup[g];
            let line_x = [x_lo, x_hi];
            let line_y = [intercept_g + b1_hat * x_lo, intercept_g + b1_hat * x_hi];
            ax.line(
                &line_x,
                &line_y,
                color,
                1.8,
                LineStyle::Solid,
                Marker::None,
                0.9,
                None,
            );
        }

        // The population (fixed-effect only) line, in bold black dashes.
        let pop_x = [x_lo, x_hi];
        let pop_y = [b0_hat + b1_hat * x_lo, b0_hat + b1_hat * x_hi];
        ax.line(
            &pop_x,
            &pop_y,
            Color::BLACK,
            2.8,
            LineStyle::Dashed,
            Marker::None,
            1.0,
            Some("population (fixed) fit"),
        );

        ax.legend(LegendLoc::UpperLeft);
    }

    // Save under the docs image folder when run from the repo root.
    let out = "docs/book/src/examples/img/mixed_ranef.svg";
    if let Some(parent) = std::path::Path::new(out).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    fig.save_svg(out).expect("write mixed_ranef.svg");
    eprintln!("wrote {out}");
}

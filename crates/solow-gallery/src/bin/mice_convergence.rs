//! Multiple imputation by chained equations (the deterministic core).
//!
//! Builds a small data set `y = b0 + b1 * x + e` and deletes some `y` values,
//! then runs `m` imputations. Each imputation fills the missing `y` with the
//! conditional mean from [`conditional_mean_impute`] plus one residual-scale
//! draw (the stochastic step a full MICE run performs; here it is supplied by
//! the gallery's deterministic RNG so the example is reproducible). Every
//! completed data set is refit by OLS, and the per-imputation slope estimates
//! and their covariances are pooled with Rubin's rules via [`combine`].
//!
//! The plot shows the observed `(x, y)` points, the imputed points (one marker
//! per missing row, at its first-imputation value), and the pooled slope's
//! 95% confidence interval drawn as a point-with-error-bar.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin mice_convergence

use ndarray::{Array1, Array2};
use solow_impute::{combine, conditional_mean_impute};
use solow_regression::LinearModel;
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Complete data: y = 1.5 + 0.8 x + N(0, 1.0), x = 0, 1, ..., 39 ------
    let mut rng = common::Rng::new(20240617);
    let n = 40usize;
    let beta0 = 1.5;
    let beta1 = 0.8;

    let x_raw: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y_full: Vec<f64> = x_raw
        .iter()
        .map(|&xi| beta0 + beta1 * xi + 1.0 * rng.normal())
        .collect();

    // Delete every fifth y (MCAR): rows 0, 5, 10, ... are missing.
    let missing: Vec<usize> = (0..n).filter(|i| i % 5 == 0).collect();
    let observed: Vec<usize> = (0..n).filter(|i| i % 5 != 0).collect();
    let n_miss = missing.len();
    let n_obs = observed.len();

    // Design blocks (intercept + x) for the observed and missing rows.
    let exog_obs = Array2::from_shape_vec(
        (n_obs, 2),
        observed.iter().flat_map(|&i| [1.0, x_raw[i]]).collect(),
    )
    .unwrap();
    let exog_miss = Array2::from_shape_vec(
        (n_miss, 2),
        missing.iter().flat_map(|&i| [1.0, x_raw[i]]).collect(),
    )
    .unwrap();
    let endog_obs = Array1::from(observed.iter().map(|&i| y_full[i]).collect::<Vec<_>>());

    // One deterministic conditional-mean fit shared by every imputation.
    let imp = conditional_mean_impute(endog_obs.clone(), exog_obs.clone(), &exog_miss).unwrap();
    let resid_sd = imp.scale.sqrt();
    println!("conditional-mean imputation");
    println!(
        "  fit on {n_obs} observed rows: y_hat = {:.4} + {:.4} x   (residual sd = {:.4})",
        imp.params[0], imp.params[1], resid_sd
    );
    println!("  {n_miss} missing rows imputed at their conditional means:");
    for (k, &i) in missing.iter().enumerate() {
        println!(
            "    row {i:2}: x = {:>4.1}   y_imputed = {:>7.4}",
            x_raw[i], imp.imputed_missing[k]
        );
    }

    // --- m imputations: conditional mean + one residual-scale draw ----------
    let m = 20usize;
    let design_full =
        Array2::from_shape_vec((n, 2), (0..n).flat_map(|i| [1.0, x_raw[i]]).collect()).unwrap();
    let dfcom = (n_obs - 2) as f64; // complete-data residual df from the observed fit.

    let mut params_list: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut cov_list: Vec<Array2<f64>> = Vec::with_capacity(m);
    // Keep the first imputation's completed y values for the plot.
    let mut imputed_first: Vec<f64> = vec![0.0; n_miss];

    for d in 0..m {
        // Completed y: observed values where present, conditional mean + draw
        // where missing.
        let mut y = y_full.clone();
        for (k, &i) in missing.iter().enumerate() {
            let draw = imp.imputed_missing[k] + resid_sd * rng.normal();
            y[i] = draw;
            if d == 0 {
                imputed_first[k] = draw;
            }
        }
        let yv = Array1::from(y);
        let res = LinearModel::ols(yv, design_full.clone())
            .unwrap()
            .fit()
            .unwrap();
        params_list.push(res.params.clone());
        cov_list.push(res.cov_params.clone());
    }

    // --- Pool with Rubin's rules --------------------------------------------
    let pooled = combine(&params_list, &cov_list, dfcom).unwrap();
    let ci = pooled.conf_int(0.05);
    let names = ["const", "x"];

    println!();
    println!(
        "pooled estimate over m = {} imputations (Rubin's rules)",
        pooled.m
    );
    println!(
        "  {:<6} {:>10} {:>10} {:>10} {:>10}   {:>20}",
        "param", "coef", "std err", "fmi", "df", "95% conf. int."
    );
    for j in 0..pooled.params.len() {
        println!(
            "  {:<6} {:>10.4} {:>10.4} {:>10.4} {:>10.2}   [{:>8.4}, {:>8.4}]",
            names[j],
            pooled.params[j],
            pooled.bse[j],
            pooled.fmi[j],
            pooled.df[j],
            ci[[j, 0]],
            ci[[j, 1]],
        );
    }

    // --- Plot: observed + imputed points, and the pooled slope CI -----------
    let x_obs: Vec<f64> = observed.iter().map(|&i| x_raw[i]).collect();
    let y_obs: Vec<f64> = observed.iter().map(|&i| y_full[i]).collect();
    let x_mis: Vec<f64> = missing.iter().map(|&i| x_raw[i]).collect();

    let mut fig = Figure::new(820, 560);
    {
        let ax = fig.axes();
        ax.set_title("Multiple imputation: observed vs imputed y, with pooled slope CI")
            .set_xlabel("x")
            .set_ylabel("y")
            .set_grid(true);

        // Pooled regression line over the full x range.
        let (b0, b1) = (pooled.params[0], pooled.params[1]);
        let x_lo = 0.0;
        let x_hi = (n - 1) as f64;
        ax.line(
            &[x_lo, x_hi],
            &[b0 + b1 * x_lo, b0 + b1 * x_hi],
            Color::GRAY,
            2.0,
            LineStyle::Dashed,
            Marker::None,
            1.0,
            Some("pooled fit"),
        );

        ax.scatter_full(
            &x_obs,
            &y_obs,
            Color::BLUE,
            5.0,
            Marker::Circle,
            0.85,
            Some("observed"),
        );
        ax.scatter_full(
            &x_mis,
            &imputed_first,
            Color::RED,
            7.0,
            Marker::Diamond,
            0.95,
            Some("imputed (draw 1)"),
        );

        // Pooled slope (b1) with its 95% CI drawn as a point-with-error-bar.
        // Place the marker near the right edge so it does not collide with the
        // cloud of data points; its value is the pooled slope estimate.
        let slope = pooled.params[1];
        let slope_half = (ci[[1, 1]] - ci[[1, 0]]) / 2.0;
        ax.errorbar(
            &[x_hi + 0.5],
            &[slope],
            &[slope_half],
            Color::GREEN,
            Some("pooled slope b1 (95% CI)"),
        );
        ax.annotate_styled(
            x_hi - 9.0,
            slope + 1.2,
            &format!("b1 = {slope:.3}"),
            Color::GREEN,
            12.0,
        );

        ax.legend(LegendLoc::UpperLeft);
    }

    let out = common::img_path("mice_convergence.svg");
    fig.save_svg(&out).expect("write mice_convergence.svg");
    eprintln!("wrote {}", out.display());
}

//! Beta regression.
//!
//! Fits a `BetaModel` (maximum-likelihood beta regression) to a response that
//! lives strictly in `(0, 1)` and depends on a single covariate through the
//! default logit mean link, with an intercept-only log precision submodel.
//! Prints the real coefficient table and overlays the fitted mean curve
//! `μ(x) = logistic(β₀ + β₁ x)` on the scatter of the data.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin beta_reg

use ndarray::{Array1, Array2};
use solow_othermod::BetaModel;
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

/// The logistic (inverse-logit) function — the default mean link's inverse.
fn logistic(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

fn main() {
    // --- Example data ------------------------------------------------------
    // Mean submodel:  logit(μ_i) = b0 + b1 * x_i,  with true (b0, b1) = (-0.4, 1.3).
    // The Beta precision is fixed at φ = 18, so each response is a draw from a
    // Beta with shape (μφ, (1−μ)φ). We synthesize the draw deterministically by
    // perturbing the true mean on the logit scale, which keeps every y strictly
    // inside (0, 1) without any rejection sampling.
    let mut rng = common::Rng::new(20240617);
    let n = 60usize;
    let b0_true = -0.4;
    let b1_true = 1.3;
    let phi_true = 18.0;

    let x_raw: Vec<f64> = (0..n)
        .map(|i| -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0))
        .collect();
    let y_vec: Vec<f64> = x_raw
        .iter()
        .map(|&xi| {
            let mu = logistic(b0_true + b1_true * xi);
            // Logit-scale noise whose spread shrinks as the Beta precision grows;
            // this mimics Beta variance μ(1−μ)/(1+φ) while guaranteeing y ∈ (0, 1).
            let sd = (1.0_f64 / (1.0 + phi_true)).sqrt() / (mu * (1.0 - mu)).sqrt();
            let eta = (mu / (1.0 - mu)).ln() + sd * rng.normal();
            logistic(eta).clamp(1e-4, 1.0 - 1e-4)
        })
        .collect();

    // Mean design: intercept + x. Precision design: intercept only.
    let mut exog = Array2::<f64>::ones((n, 2));
    for i in 0..n {
        exog[[i, 1]] = x_raw[i];
    }
    let exog_precision = Array2::<f64>::ones((n, 1));
    let y = Array1::from(y_vec.clone());

    // --- Fit by maximum likelihood -----------------------------------------
    let res = BetaModel::new(y, exog, exog_precision)
        .unwrap()
        .fit()
        .unwrap();

    // --- Printed results (real estimates) ----------------------------------
    let beta = res.params_mean();
    let gamma = res.params_precision();
    // The precision submodel uses a log link, so φ = exp(γ₀).
    let phi_hat = gamma[0].exp();

    println!("Beta regression (logit mean link, log precision link)");
    println!("  converged      : {}", res.converged);
    println!("  nobs           : {}", res.nobs as usize);
    println!("  log-likelihood : {:.4}", res.llf);
    println!();
    println!("  mean coefficients (β)        coef     std err          z       P>|z|");
    let names = ["const", "x"];
    for j in 0..beta.len() {
        println!(
            "  {:<10} {:12.4} {:12.4} {:12.4} {:11.4}",
            names[j], res.params[j], res.bse[j], res.tvalues[j], res.pvalues[j]
        );
    }
    let k = beta.len();
    println!("  precision coefficient (γ)    coef     std err          z       P>|z|");
    println!(
        "  {:<10} {:12.4} {:12.4} {:12.4} {:11.4}",
        "log(phi)", res.params[k], res.bse[k], res.tvalues[k], res.pvalues[k]
    );
    println!();
    println!("  implied precision  phi = exp(gamma0) = {:.4}", phi_hat);
    println!(
        "  recovered mean model: logit(mu) = {:.4} + {:.4} x   (true: {:.4} + {:.4} x)",
        beta[0], beta[1], b0_true, b1_true
    );

    // --- Scatter of observations + fitted mean curve -----------------------
    // The fitted mean μ̂(x) = logistic(β̂₀ + β̂₁ x) traced over the covariate range.
    let m = 200usize;
    let (xlo, xhi) = (-2.0, 2.0);
    let xs_curve: Vec<f64> = (0..m)
        .map(|i| xlo + (xhi - xlo) * (i as f64) / (m as f64 - 1.0))
        .collect();
    let ys_curve: Vec<f64> = xs_curve
        .iter()
        .map(|&xv| logistic(beta[0] + beta[1] * xv))
        .collect();

    let mut fig = Figure::new(760, 520);
    {
        let ax = fig.axes();
        ax.set_title("Beta regression: mean of y in (0, 1) vs x")
            .set_xlabel("x")
            .set_ylabel("y")
            .set_grid(true);
        ax.set_ylim(0.0, 1.0);
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
            &xs_curve,
            &ys_curve,
            Color::RED,
            2.5,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("fitted mean"),
        );
        ax.legend(LegendLoc::UpperLeft);
    }

    let out = common::img_path("beta_reg.svg");
    fig.save_svg(&out).expect("write beta_reg.svg");
    eprintln!("wrote {}", out.display());
}

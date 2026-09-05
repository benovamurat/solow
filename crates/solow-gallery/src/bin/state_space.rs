//! State space: the Kalman filter (local-level unobserved-components model).
//!
//! Simulates a local-level series — a random-walk level `mu_t` observed
//! through additive irregular noise — fits an `UnobservedComponents` model by
//! maximum likelihood, runs the exact Kalman filter at the estimated variances,
//! and plots the noisy observations overlaid with the filtered (one-step-ahead)
//! level estimate.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin state_space

use ndarray::Array1;
use solow_statespace::{Level, UcSpec, UnobservedComponents};
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Simulate a local level: mu_t = mu_{t-1} + xi_t,  y_t = mu_t + eps_t -
    //
    //   xi_t  ~ N(0, 0.30)   (level disturbance)
    //   eps_t ~ N(0, 1.50)   (irregular / measurement noise)
    //
    // All randomness is the deterministic SplitMix64 RNG, so the series — and
    // therefore the whole example — is fully reproducible.
    let mut rng = common::Rng::new(20240617);
    let n = 120usize;
    let sig_level = 0.30_f64.sqrt();
    let sig_irreg = 1.50_f64.sqrt();

    let mut mu = 5.0_f64;
    let mut level = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    for _ in 0..n {
        mu += sig_level * rng.normal();
        level.push(mu);
        y_vec.push(mu + sig_irreg * rng.normal());
    }
    let t_raw: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y = Array1::from(y_vec.clone());

    // --- Fit the local-level model by maximum likelihood --------------------
    let spec = UcSpec::new(Level::LocalLevel);
    let model = UnobservedComponents::new(y.clone(), spec).expect("build model");
    let res = model.fit().expect("fit");

    // params = [sigma2.irregular, sigma2.level] for a local level.
    let s2_irregular = res.params[0];
    let s2_level = res.params[1];

    println!("Local-level unobserved-components fit (Kalman filter MLE)");
    println!("  nobs            = {}", res.nobs);
    println!("  converged       = {}", res.converged);
    println!("  sigma2.irregular= {:.6}  (true 1.500000)", s2_irregular);
    println!("  sigma2.level    = {:.6}  (true 0.300000)", s2_level);
    println!("  log-likelihood  = {:.6}", res.llf);
    println!("  AIC             = {:.6}", res.aic);
    println!("  BIC             = {:.6}", res.bic);
    println!("  HQIC            = {:.6}", res.hqic);

    // --- Run the Kalman filter at the estimated variances -------------------
    // The first `burn` (= number of nonstationary states = 1) observations are
    // excluded from the log-likelihood, matching the reference implementation.
    let ss = model.build_state_space(&res.params);
    let out = ss.filter(&y, 1);
    // Filtered one-step-ahead state estimate a_{t|t}; the local level is the
    // single state (column 0).
    let filtered: Vec<f64> = out.filtered_state.column(0).to_vec();

    // Diagnostic: mean squared one-step-ahead forecast error vs the raw series.
    let mse_filt: f64 = filtered
        .iter()
        .zip(level.iter())
        .map(|(&f, &m)| (f - m) * (f - m))
        .sum::<f64>()
        / n as f64;
    let mse_obs: f64 = y_vec
        .iter()
        .zip(level.iter())
        .map(|(&o, &m)| (o - m) * (o - m))
        .sum::<f64>()
        / n as f64;
    println!("  MSE(filtered vs true level) = {:.6}", mse_filt);
    println!("  MSE(observed vs true level) = {:.6}", mse_obs);
    println!(
        "  first filtered states: {:.4}, {:.4}, {:.4}",
        filtered[0], filtered[1], filtered[2]
    );

    // --- Plot observations + the filtered level estimate --------------------
    let mut fig = Figure::new(820, 520);
    {
        let ax = fig.axes();
        ax.set_title("Local level: observations and filtered Kalman state")
            .set_xlabel("t")
            .set_ylabel("y")
            .set_grid(true);
        ax.scatter_full(
            &t_raw,
            &y_vec,
            Color::cycle(0),
            3.5,
            Marker::Circle,
            0.55,
            Some("observed"),
        );
        ax.line(
            &t_raw,
            &filtered,
            Color::RED,
            2.0,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("filtered level"),
        );
        ax.legend(LegendLoc::UpperLeft);
    }

    let out_path = common::img_path("state_space.svg");
    fig.save_svg(&out_path).expect("write state_space.svg");
    eprintln!("wrote {}", out_path.display());
}

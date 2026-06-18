//! Markov switching regime probabilities.
//!
//! Synthesizes a single series whose mean shifts between two hidden regimes
//! (a low-mean state and a high-mean state) driven by a deterministic Markov
//! chain, fits a 2-regime [`MarkovRegression`] by maximum likelihood (the
//! Hamilton filter / Kim smoother), prints the key fitted quantities, and plots
//! the smoothed probability of being in regime 1 across the sample.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin regime

use ndarray::Array1;
use solow_regime::MarkovRegression;
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Example data: a two-state hidden Markov chain driving the mean -------
    //
    // Regime 0 has mean 0, regime 1 has mean 4; both share noise std 1.0. The
    // chain is persistent (it tends to stay put), so the series shows clear
    // stretches in each state. Everything is generated from the deterministic
    // SplitMix64 RNG in common.rs, so the run is fully reproducible.
    let mut rng = common::Rng::new(0xA11CE);
    let n = 200usize;
    let mu = [0.0f64, 4.0f64];
    let sigma = 1.0f64;
    // Probability of staying in the current regime each step.
    let stay = 0.95f64;

    let mut state = 0usize;
    let mut true_state = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    for _ in 0..n {
        // Persist or switch.
        if rng.uniform() > stay {
            state = 1 - state;
        }
        true_state.push(state as f64);
        y_vec.push(mu[state] + sigma * rng.normal());
    }

    let y = Array1::from(y_vec.clone());

    // --- Fit a 2-regime switching regression (switching mean & variance) ------
    let model = MarkovRegression::new(y, 2, true).expect("build MarkovRegression");
    let res = model.fit().expect("fit MarkovRegression");

    // --- Printed results (real fitted quantities) -----------------------------
    println!("Markov switching regression (2 regimes)");
    println!("  converged           : {}", res.converged);
    println!("  nobs                : {}", res.nobs);
    println!("  k_params            : {}", res.k_params);
    println!("  log-likelihood      : {:.4}", res.llf);
    println!("  AIC                 : {:.4}", res.aic);
    println!("  BIC                 : {:.4}", res.bic);
    println!();
    println!("  estimated parameters:");
    for (name, val) in res.param_names.iter().zip(res.params.iter()) {
        println!("    {name:<12} = {val:>9.4}");
    }
    println!();
    println!("  transition matrix P[i<-j] (columns sum to 1):");
    let (k, _) = res.transition.dim();
    for i in 0..k {
        let row: Vec<String> = (0..k)
            .map(|j| format!("{:.4}", res.transition[[i, j]]))
            .collect();
        println!("    [{}]", row.join(", "));
    }
    println!(
        "  steady-state probs  : [{:.4}, {:.4}]",
        res.initial_probabilities[0], res.initial_probabilities[1]
    );
    println!(
        "  expected durations  : [{:.2}, {:.2}] periods",
        res.expected_durations[0], res.expected_durations[1]
    );

    // --- Smoothed probability of regime 1 over the sample ---------------------
    // `smoothed_marginal_probabilities` is (nobs, k); column 1 is Pr(S_t = 1 | Y_T).
    let t_axis: Vec<f64> = (0..res.nobs).map(|t| t as f64).collect();
    let p_regime1: Vec<f64> = (0..res.nobs)
        .map(|t| res.smoothed_marginal_probabilities[[t, 1]])
        .collect();

    let frac_high = p_regime1.iter().sum::<f64>() / res.nobs as f64;
    println!();
    println!("  mean smoothed Pr(regime 1) : {frac_high:.4}");

    let mut fig = Figure::new(820, 520);
    {
        let ax = fig.axes();
        ax.set_title("Markov switching: smoothed Pr(regime 1)")
            .set_xlabel("t")
            .set_ylabel("Pr(S_t = 1 | Y)")
            .set_grid(true);
        ax.set_ylim(-0.02, 1.02);
        // The true hidden state (0/1) as a faint step-like reference.
        ax.line(
            &t_axis,
            &true_state,
            Color::GRAY,
            1.0,
            LineStyle::Dashed,
            Marker::None,
            0.6,
            Some("true regime"),
        );
        // The smoothed probability curve.
        ax.line(
            &t_axis,
            &p_regime1,
            Color::cycle(3),
            2.0,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("smoothed Pr(regime 1)"),
        );
        ax.legend(LegendLoc::UpperRight);
    }

    let out = common::img_path("regime.svg");
    fig.save_svg(&out).expect("write regime.svg");
    eprintln!("wrote {}", out.display());
}

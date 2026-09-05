//! Survival analysis: the Kaplan-Meier product-limit estimator.
//!
//! Right-censored survival times are generated from an exponential model with
//! independent exponential censoring. The Kaplan-Meier estimate of S(t) is
//! computed and drawn as a step function, with Greenwood pointwise 95% bands.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin survival

use solow_duration::SurvfuncRight;
use solow_viz::{Color, Figure, LineStyle, StepWhere};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Right-censored data: event ~ Exp(0.08), censoring ~ Exp(0.03) ------
    let mut rng = common::Rng::new(99);
    let n = 120usize;
    let mut time = Vec::with_capacity(n);
    let mut status = Vec::with_capacity(n);
    let mut n_events = 0usize;
    for _ in 0..n {
        let t_event = rng.exponential(0.08);
        let t_cens = rng.exponential(0.03);
        if t_event <= t_cens {
            time.push(t_event);
            status.push(1.0); // observed event
            n_events += 1;
        } else {
            time.push(t_cens);
            status.push(0.0); // right-censored
        }
    }

    let km = SurvfuncRight::new(&time, &status).unwrap();
    let k = km.surv_times.len();

    // --- Printed summary (a few rows of the survival table) -----------------
    println!("Kaplan-Meier survival estimate");
    println!("Observations: {n}   events: {n_events}   distinct event times: {k}\n");
    println!(
        "{:>10}{:>10}{:>10}{:>12}",
        "time", "n_risk", "n_event", "S(t)"
    );
    let show: Vec<usize> = (0..k).step_by((k / 10).max(1)).collect();
    for &i in &show {
        println!(
            "{:>10.3}{:>10.0}{:>10.0}{:>12.4}",
            km.surv_times[i], km.n_risk[i], km.n_events[i], km.surv_prob[i]
        );
    }
    println!(
        "\nMedian survival ~ first time with S(t) <= 0.5: {}",
        median_survival(&km)
            .map(|m| format!("{m:.3}"))
            .unwrap_or_else(|| "not reached".into())
    );

    // --- Step-function plot with Greenwood 95% bands ------------------------
    // Prepend t=0, S=1 so the curve starts at the origin of the survival axis.
    let mut t = vec![0.0];
    let mut s = vec![1.0];
    let mut lo = vec![1.0];
    let mut hi = vec![1.0];
    for i in 0..k {
        t.push(km.surv_times[i]);
        let si = km.surv_prob[i];
        s.push(si);
        let se = km.surv_prob_se[i];
        let (l, h) = if se.is_finite() {
            ((si - 1.96 * se).max(0.0), (si + 1.96 * se).min(1.0))
        } else {
            (si, si)
        };
        lo.push(l);
        hi.push(h);
    }

    let mut fig = Figure::new(760, 520);
    {
        let ax = fig.axes();
        ax.set_title("Kaplan-Meier survival function")
            .set_xlabel("time")
            .set_ylabel("S(t)")
            .set_grid(true);
        ax.set_ylim(0.0, 1.02);
        // Confidence band first (drawn underneath the step curve).
        ax.fill_between(&t, &lo, &hi, Color::cycle(0), 0.18, Some("95% band"));
        ax.step(&t, &s, Color::cycle(0), StepWhere::Post, Some("S(t)"));
        let _ = LineStyle::Solid;
        ax.legend(solow_viz::LegendLoc::UpperRight);
    }

    let out = common::img_path("survival.svg");
    fig.save_svg(&out).expect("write survival.svg");
    eprintln!("wrote {}", out.display());
}

/// First event time at which the survival estimate drops to 0.5 or below.
fn median_survival(km: &SurvfuncRight) -> Option<f64> {
    for i in 0..km.surv_times.len() {
        if km.surv_prob[i] <= 0.5 {
            return Some(km.surv_times[i]);
        }
    }
    None
}

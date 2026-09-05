//! Population-averaged GEE for clustered count data.
//!
//! Fits a marginal Poisson model `log E[y] = b0 + b1 * x` with an
//! *exchangeable* working correlation to deterministic synthetic data in which
//! observations share a cluster-level effect (so within-cluster responses are
//! genuinely correlated). Prints the real fitted parameters, robust standard
//! errors, and the estimated working correlation, then overlays the fitted
//! marginal mean on a scatter of the observed counts.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin gee_marginal

use ndarray::{Array1, Array2};
use solow_gee::{CovStruct, Gee};
use solow_glm::Family;
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Clustered count data ------------------------------------------------
    // 20 clusters of 5 observations each (n = 100). The marginal mean follows
    // log(mu) = b0 + b1 * x with b0 = 0.5, b1 = 0.30. Each cluster carries a
    // shared Gaussian "frailty" on the log scale, which makes the five counts
    // within a cluster positively correlated -- exactly the situation the
    // exchangeable working correlation is meant to capture.
    let mut rng = common::Rng::new(0xC0FFEE);
    let n_clusters = 20usize;
    let per_cluster = 5usize;
    let beta0 = 0.5;
    let beta1 = 0.30;

    let mut x_raw: Vec<f64> = Vec::new();
    let mut y_vec: Vec<f64> = Vec::new();
    let mut group_labels: Vec<i64> = Vec::new();

    for c in 0..n_clusters {
        // One shared cluster effect for all rows in this cluster.
        let u = 0.40 * rng.normal();
        for k in 0..per_cluster {
            // Covariate spread over [0, 4] within each cluster.
            let x = k as f64 * (4.0 / (per_cluster - 1) as f64);
            let eta = beta0 + beta1 * x + u;
            let mu = eta.exp();
            let y = rng.poisson(mu);
            x_raw.push(x);
            y_vec.push(y);
            group_labels.push(c as i64);
        }
    }
    let n = x_raw.len();

    // Design matrix with an explicit intercept column [1, x].
    let mut design = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        design[[i, 0]] = 1.0;
        design[[i, 1]] = x_raw[i];
    }
    let y = Array1::from(y_vec.clone());

    // --- Fit the population-averaged Poisson GEE (exchangeable) --------------
    let res = Gee::new(
        y,
        design,
        &group_labels,
        Family::Poisson,
        CovStruct::Exchangeable,
    )
    .unwrap()
    .fit()
    .unwrap();

    // --- Real fitted quantities ---------------------------------------------
    let b0 = res.params[0];
    let b1 = res.params[1];
    println!("Population-averaged GEE (Poisson, log link, exchangeable)");
    println!("  observations         : {}", n);
    println!("  clusters             : {}", n_clusters);
    println!("  converged            : {}", res.converged);
    println!("  score-equation norm  : {:.3e}", res.score_norm);
    println!();
    println!("  param        estimate     robust SE    naive SE       z");
    let names = ["const", "x"];
    for j in 0..res.params.len() {
        println!(
            "  {:<10} {:>11.6} {:>12.6} {:>11.6} {:>8.3}",
            names[j], res.params[j], res.bse[j], res.bse_naive[j], res.tvalues[j]
        );
    }
    println!();
    println!("  working corr (alpha) : {:.6}", res.dep_params);
    println!("  scale                : {:.6}", res.scale);
    println!(
        "  marginal mean        : log(mu) = {:.4} + {:.4} * x",
        b0, b1
    );

    // --- Plot: observed counts + fitted marginal mean curve -----------------
    // The fitted marginal mean is mu(x) = exp(b0 + b1 * x).
    let n_line = 100usize;
    let x_max = 4.0;
    let xs_line: Vec<f64> = (0..n_line)
        .map(|i| x_max * i as f64 / (n_line - 1) as f64)
        .collect();
    let ys_line: Vec<f64> = xs_line.iter().map(|&x| (b0 + b1 * x).exp()).collect();

    let mut fig = Figure::new(760, 520);
    {
        let ax = fig.axes();
        ax.set_title("GEE marginal mean: Poisson, exchangeable working correlation")
            .set_xlabel("x")
            .set_ylabel("count y")
            .set_grid(true);
        ax.scatter_full(
            &x_raw,
            &y_vec,
            Color::cycle(0),
            4.0,
            Marker::Circle,
            0.55,
            Some("observed (20 clusters)"),
        );
        ax.line(
            &xs_line,
            &ys_line,
            Color::RED,
            2.5,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("fitted marginal mean"),
        );
        ax.legend(LegendLoc::UpperLeft);
    }

    let out = common::img_path("gee_marginal.svg");
    fig.save_svg(&out).expect("write gee_marginal.svg");
    eprintln!("wrote {}", out.display());
}

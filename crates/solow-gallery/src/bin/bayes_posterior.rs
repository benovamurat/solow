//! Bayesian mixed GLM by variational Bayes (binomial / logit link).
//!
//! Builds a balanced random-intercept logistic design: `n_groups` groups, each
//! with `per` observations, a shared intercept and slope (the fixed effects),
//! and one Gaussian random intercept per group. The 0/1 responses are drawn
//! deterministically from the true generative model with the gallery's
//! SplitMix64 RNG. The model is then fit by mean-field variational Bayes
//! (`BayesMixedGlm::fit_vb`), which maximizes the evidence lower bound (ELBO) of
//! a factored Gaussian variational posterior.
//!
//! Variational Bayes returns a Gaussian *approximate* posterior, summarized by a
//! posterior mean and posterior standard deviation per parameter — not MCMC
//! draws. We honor that: the plot shows the two fixed-effect posterior means as
//! points with +/- 2 posterior-sd credible bands (the natural variational
//! analogue of a 95% interval), with the data-generating truth marked for
//! reference.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin bayes_posterior

use ndarray::{Array1, Array2};
use solow_bayes::{BayesMixedGlm, Family};
use solow_viz::{Color, Figure, LegendLoc, LineStyle};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Generative model ----------------------------------------------------
    // logit P(y=1) = b0 + b1 * x + u_group, u_group ~ N(0, sigma_u^2).
    let n_groups = 8usize;
    let per = 12usize;
    let n = n_groups * per;
    let true_b0 = -0.4_f64; // true intercept (fixed effect 0)
    let true_b1 = 1.3_f64; // true slope     (fixed effect 1)
    let true_sigma_u = 0.8_f64; // sd of the group random intercepts

    let mut rng = common::Rng::new(20240618);

    // One random intercept per group, drawn once.
    let group_u: Vec<f64> = (0..n_groups).map(|_| true_sigma_u * rng.normal()).collect();

    // Fixed-effects design: [1, x]. Random-effects design: one indicator column
    // per group (a random intercept). `ident` ties all group columns to a single
    // variance component (one shared log-sd parameter).
    let mut exog = Array2::<f64>::zeros((n, 2));
    let mut exog_vc = Array2::<f64>::zeros((n, n_groups));
    let mut endog = Array1::<f64>::zeros(n);
    let mut x_raw = Vec::with_capacity(n);
    let mut grp = Vec::with_capacity(n);

    for i in 0..n {
        let g = i / per;
        // x spans [-1.5, 1.5] within each group.
        let xi = -1.5 + 3.0 * (i % per) as f64 / (per as f64 - 1.0);
        let eta = true_b0 + true_b1 * xi + group_u[g];
        let p = 1.0 / (1.0 + (-eta).exp());
        let y = rng.bernoulli(p);

        exog[[i, 0]] = 1.0;
        exog[[i, 1]] = xi;
        exog_vc[[i, g]] = 1.0;
        endog[i] = y;
        x_raw.push(xi);
        grp.push(g);
    }

    let ident = vec![0usize; n_groups];

    // --- Fit by variational Bayes -------------------------------------------
    // vcp_p, fe_p are the prior sds for the variance-component log-sd and the
    // fixed effects, respectively.
    let model = BayesMixedGlm::new(
        Family::Binomial,
        endog.clone(),
        exog,
        exog_vc,
        ident,
        1.0,
        4.0,
    )
    .unwrap();
    let res = model.fit_vb(None, None, 100_000, 1e-6).unwrap();

    // --- Real results --------------------------------------------------------
    let fe_names = ["intercept (b0)", "slope (b1)"];
    println!("Bayesian mixed GLM (binomial / logit) by variational Bayes");
    println!(
        "  observations: {}   groups: {}   per group: {}",
        n, n_groups, per
    );
    println!(
        "  converged: {}   iters: {}   |grad|: {:.3e}   ELBO: {:.6}",
        res.converged, res.iters, res.grad_norm, res.elbo
    );
    println!();
    println!("Fixed effects (approximate Gaussian posterior):");
    println!(
        "  {:<16} {:>12} {:>12} {:>22}",
        "name", "post. mean", "post. sd", "post. mean +/- 2 sd"
    );
    for k in 0..res.fe_mean.len() {
        let m = res.fe_mean[k];
        let s = res.fe_sd[k];
        println!(
            "  {:<16} {:>12.5} {:>12.5} {:>10.5} .. {:>8.5}",
            fe_names[k],
            m,
            s,
            m - 2.0 * s,
            m + 2.0 * s
        );
    }
    println!();
    // The variance component is parameterized as a log standard deviation, so
    // exp(vcp_mean) is the posterior-mean group-intercept sd.
    let vcp_m = res.vcp_mean[0];
    let vcp_s = res.vcp_sd[0];
    println!("Variance component (group random intercept, log-sd):");
    println!(
        "  vcp posterior mean: {:.5}   posterior sd: {:.5}",
        vcp_m, vcp_s
    );
    println!(
        "  implied sigma_u = exp(vcp_mean): {:.5}   (true {:.3})",
        vcp_m.exp(),
        true_sigma_u
    );
    println!();
    println!("Random intercept posterior means by group (vc_mean):");
    for g in 0..n_groups {
        println!(
            "  group {:>2}: post. mean {:>8.5}  post. sd {:>7.5}  (true u {:>8.5})",
            g, res.vc_mean[g], res.vc_sd[g], group_u[g]
        );
    }

    // --- Plot: fixed-effect posterior means with credible bands -------------
    // x positions 0, 1 for the two fixed effects; y is the posterior mean; the
    // error bar is +/- 2 posterior sds (the variational 95%-style band).
    let xs = [0.0_f64, 1.0];
    let means = [res.fe_mean[0], res.fe_mean[1]];
    let err2 = [2.0 * res.fe_sd[0], 2.0 * res.fe_sd[1]];
    let truth = [true_b0, true_b1];

    let mut fig = Figure::new(760, 520);
    {
        let ax = fig.axes();
        ax.set_title("Bayesian mixed GLM (VB): fixed-effect posteriors")
            .set_xlabel("fixed effect")
            .set_ylabel("coefficient")
            .set_grid(true);
        ax.set_xlim(-0.6, 1.6);

        // Zero reference line.
        ax.axhline(0.0, Color::GRAY, LineStyle::Dotted);

        // Posterior means with +/- 2 sd credible bands.
        ax.errorbar(
            &xs,
            &means,
            &err2,
            Color::cycle(0),
            Some("posterior mean +/- 2 sd"),
        );

        // The data-generating truth, for reference.
        ax.scatter_full(
            &xs,
            &truth,
            Color::RED,
            6.0,
            solow_viz::Marker::Diamond,
            1.0,
            Some("true value"),
        );

        ax.legend(LegendLoc::UpperLeft);
    }

    let out = common::img_path("bayes_posterior.svg");
    fig.save_svg(&out).expect("write bayes_posterior.svg");
    eprintln!("wrote {}", out.display());
}

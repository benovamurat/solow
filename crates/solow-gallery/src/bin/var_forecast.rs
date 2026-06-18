//! Vector autoregression (VAR).
//!
//! Simulates a deterministic 2-variable VAR(1) system, fits a VAR(1) with
//! `solow-var`, prints the key estimated quantities (intercept, coefficient
//! matrix, residual covariance, log-likelihood and information criteria), and
//! plots the two observed series together with their in-sample fitted values
//! and a short out-of-sample forecast iterated from the estimated parameters.
//!
//! Run with:
//!   cargo run -p solow-gallery --bin var_forecast

use ndarray::Array2;
use solow_var::Var;
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Simulate a stable bivariate VAR(1): y_t = nu + A y_{t-1} + u_t -----
    // The companion matrix A has spectral radius < 1, so the process is
    // stationary. Noise is the deterministic SplitMix64 stream from common.rs,
    // so the whole example is reproducible bit-for-bit.
    let mut rng = common::Rng::new(20240617);
    let n = 60usize;

    let nu = [0.6_f64, -0.3_f64]; // true intercept
    let a = [[0.5_f64, 0.2_f64], [-0.1_f64, 0.4_f64]]; // true A_1

    let mut y = Array2::<f64>::zeros((n, 2));
    // Start the recursion from the unconditional-ish point [1.0, -0.5].
    y[[0, 0]] = 1.0;
    y[[0, 1]] = -0.5;
    for t in 1..n {
        let y0 = y[[t - 1, 0]];
        let y1 = y[[t - 1, 1]];
        y[[t, 0]] = nu[0] + a[0][0] * y0 + a[0][1] * y1 + 0.20 * rng.normal();
        y[[t, 1]] = nu[1] + a[1][0] * y0 + a[1][1] * y1 + 0.20 * rng.normal();
    }

    // --- Fit a VAR(1) by equation-by-equation OLS --------------------------
    let res = Var::new(y.clone()).unwrap().fit(1).unwrap();

    // --- Print the key real results ----------------------------------------
    println!("VAR(1) fit on a simulated bivariate system");
    println!("neqs (K)        = {}", res.neqs);
    println!("k_ar (p)        = {}", res.k_ar);
    println!("nobs (T)        = {}", res.nobs);
    println!("df_model        = {}", res.df_model);
    println!("df_resid        = {}", res.df_resid);
    println!();
    println!(
        "intercept nu    = [{:.4}, {:.4}]   (true [{:.1}, {:.1}])",
        res.intercept[0], res.intercept[1], nu[0], nu[1]
    );
    let a_hat = &res.coefs[0];
    println!("estimated A_1   =");
    println!(
        "  [{:>8.4}  {:>8.4}]   (true [{:>4.1}  {:>4.1}])",
        a_hat[[0, 0]],
        a_hat[[0, 1]],
        a[0][0],
        a[0][1]
    );
    println!(
        "  [{:>8.4}  {:>8.4}]   (true [{:>4.1}  {:>4.1}])",
        a_hat[[1, 0]],
        a_hat[[1, 1]],
        a[1][0],
        a[1][1]
    );
    println!();
    println!("sigma_u (resid cov, df-adjusted) =");
    println!(
        "  [{:>9.5}  {:>9.5}]",
        res.sigma_u[[0, 0]],
        res.sigma_u[[0, 1]]
    );
    println!(
        "  [{:>9.5}  {:>9.5}]",
        res.sigma_u[[1, 0]],
        res.sigma_u[[1, 1]]
    );
    println!();
    println!("log-likelihood  = {:.4}", res.llf);
    println!(
        "AIC = {:.4}   BIC = {:.4}   HQIC = {:.4}   FPE = {:.6}",
        res.aic, res.bic, res.hqic, res.fpe
    );

    // --- Multi-step forecast iterated from the fitted parameters -----------
    // y_hat_{T+h} = nu_hat + A_hat * y_hat_{T+h-1}, seeded at the last sample.
    let h = 10usize;
    let mut fc = Array2::<f64>::zeros((h, 2));
    let mut prev = [y[[n - 1, 0]], y[[n - 1, 1]]];
    for k in 0..h {
        let f0 = res.intercept[0] + a_hat[[0, 0]] * prev[0] + a_hat[[0, 1]] * prev[1];
        let f1 = res.intercept[1] + a_hat[[1, 0]] * prev[0] + a_hat[[1, 1]] * prev[1];
        fc[[k, 0]] = f0;
        fc[[k, 1]] = f1;
        prev = [f0, f1];
    }
    println!();
    println!("forecast horizon h = {h}");
    println!("y_hat[T+1]      = [{:.4}, {:.4}]", fc[[0, 0]], fc[[0, 1]]);
    println!(
        "y_hat[T+{h}]     = [{:.4}, {:.4}]",
        fc[[h - 1, 0]],
        fc[[h - 1, 1]]
    );

    // --- Plot: observed series, in-sample fit, and the forecast ------------
    // The fitted values cover observations p..n (one lag is consumed).
    let t_obs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y0_obs: Vec<f64> = (0..n).map(|i| y[[i, 0]]).collect();
    let y1_obs: Vec<f64> = (0..n).map(|i| y[[i, 1]]).collect();

    let p = res.k_ar;
    let t_fit: Vec<f64> = (p..n).map(|i| i as f64).collect();
    let y0_fit: Vec<f64> = (0..res.nobs).map(|i| res.fittedvalues[[i, 0]]).collect();
    let y1_fit: Vec<f64> = (0..res.nobs).map(|i| res.fittedvalues[[i, 1]]).collect();

    // Forecast x runs from the last observation through T+h so the line joins
    // visually onto the observed path.
    let mut t_fc: Vec<f64> = vec![(n - 1) as f64];
    t_fc.extend((0..h).map(|k| (n + k) as f64));
    let mut y0_fc: Vec<f64> = vec![y[[n - 1, 0]]];
    y0_fc.extend((0..h).map(|k| fc[[k, 0]]));
    let mut y1_fc: Vec<f64> = vec![y[[n - 1, 1]]];
    y1_fc.extend((0..h).map(|k| fc[[k, 1]]));

    let mut fig = Figure::new(820, 540);
    {
        let ax = fig.axes();
        ax.set_title("VAR(1): observed series, in-sample fit, and forecast")
            .set_xlabel("t")
            .set_ylabel("y")
            .set_grid(true);

        // Observed series 1.
        ax.scatter_full(
            &t_obs,
            &y0_obs,
            Color::cycle(0),
            3.0,
            Marker::Circle,
            0.6,
            Some("y1 observed"),
        );
        ax.line(
            &t_fit,
            &y0_fit,
            Color::cycle(0),
            2.0,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("y1 fitted"),
        );
        ax.line(
            &t_fc,
            &y0_fc,
            Color::cycle(0),
            2.0,
            LineStyle::Dashed,
            Marker::None,
            1.0,
            Some("y1 forecast"),
        );

        // Observed series 2.
        ax.scatter_full(
            &t_obs,
            &y1_obs,
            Color::cycle(1),
            3.0,
            Marker::Square,
            0.6,
            Some("y2 observed"),
        );
        ax.line(
            &t_fit,
            &y1_fit,
            Color::cycle(1),
            2.0,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("y2 fitted"),
        );
        ax.line(
            &t_fc,
            &y1_fc,
            Color::cycle(1),
            2.0,
            LineStyle::Dashed,
            Marker::None,
            1.0,
            Some("y2 forecast"),
        );

        // Mark where the sample ends and the forecast begins.
        ax.axvline((n - 1) as f64, Color::GRAY, LineStyle::Dotted);
        ax.legend(LegendLoc::UpperLeft);
    }

    let out = common::img_path("var_forecast.svg");
    fig.save_svg(&out).expect("write var_forecast.svg");
    eprintln!("wrote {}", out.display());
}

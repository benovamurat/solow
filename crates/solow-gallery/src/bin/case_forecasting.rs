//! Production case study: a demand-forecasting service in pure Rust.
//!
//! Scenario. A fulfillment center needs a day-ahead and week-ahead forecast of
//! daily order volume so it can pre-stage temporary staff. The forecasting step
//! must run inside an existing Rust service — no Python sidecar, no FFI, no
//! network call to a model server. This binary is that forecasting step in
//! miniature:
//!
//!   1. Build a deterministic daily-demand series with a rising trend and a
//!      strong weekly (period-7) cycle, using the gallery's SplitMix64 RNG so
//!      the whole run is reproducible bit-for-bit.
//!   2. Hold out the final 21 days as a backtest window the model never sees.
//!   3. Fit a seasonal autoregression on the training days with
//!      `solow_tsa::AutoReg` (lags 1..=7 capture both day-to-day persistence
//!      and the weekly lag; a constant-plus-linear trend captures growth).
//!   4. Produce a genuine multi-step forecast by iterating the estimated
//!      parameters forward, with a Gaussian 95% prediction band whose width
//!      grows with the accumulated one-step innovation variance.
//!   5. Score the forecast against the held-out actuals (MAE / RMSE / MAPE) and
//!      turn the week-ahead number into a concrete staffing decision.
//!   6. Render the whole story to an SVG with `solow-viz`.
//!
//! Everything printed below is computed at run time from the fitted model;
//! nothing is hard-coded.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin case_forecasting

use ndarray::Array1;
use solow_distributions::norm_ppf;
use solow_tsa::{AutoReg, Trend};
use solow_viz::{Color, Figure, LegendLoc, LineStyle, Marker};

#[path = "../common.rs"]
mod common;

/// Orders processed per worker-shift; drives the staffing decision.
const ORDERS_PER_SHIFT: f64 = 55.0;

fn main() {
    // --- 1. Synthesize a realistic daily-demand series ----------------------
    // level + linear growth + weekly seasonal profile + AR(1) demand shocks.
    let mut rng = common::Rng::new(20260616);
    let n_total = 140usize; // 20 weeks of daily data
    let period = 7usize;

    // A plausible Monday..Sunday multiplicative-ish weekly profile (additive
    // here), peaking mid-week and dipping on the weekend.
    let weekly = [18.0, 26.0, 30.0, 24.0, 12.0, -22.0, -28.0];

    let base = 180.0; // baseline orders/day
    let growth = 0.45; // +0.45 orders/day trend (~3/week)
    let phi = 0.5; // persistence of demand shocks
    let mut shock = 0.0f64;

    let mut demand = vec![0.0f64; n_total];
    for t in 0..n_total {
        shock = phi * shock + 9.0 * rng.normal();
        let trend = base + growth * t as f64;
        let season = weekly[t % period];
        demand[t] = (trend + season + shock).max(0.0).round();
    }

    // --- 2. Train / backtest split ------------------------------------------
    let h = 21usize; // backtest horizon: forecast the final 3 weeks
    let n_train = n_total - h;
    let train = Array1::from(demand[..n_train].to_vec());
    let actual_future = &demand[n_train..]; // held-out truth

    println!("Demand-forecasting service  (pure-Rust inference, no Python)");
    println!("------------------------------------------------------------");
    println!("daily series: {n_total} days  ->  train on {n_train}, backtest the last {h}");
    println!("weekly period = {period} days\n");

    // --- 3. Fit a seasonal autoregression on the training window ------------
    // lags 1..=7: lag-7 carries the weekly signal; lags 1..3 carry the
    // short-run persistence. Trend::Ct = constant + linear growth term.
    let lags = 7usize;
    let model = AutoReg::new(train.clone(), lags, Trend::Ct).unwrap();
    let fit = model.fit().unwrap();

    // params layout for Ct: [const, trend, y.L1, ..., y.L7]
    let mut names = vec!["const".to_string(), "trend".to_string()];
    for l in 1..=lags {
        names.push(format!("y.L{l}"));
    }

    println!("AutoReg(7) + linear trend, conditional least squares");
    println!(
        "{:<10}{:>12}{:>12}{:>12}{:>12}",
        "param", "coef", "std err", "z", "P>|z|"
    );
    for i in 0..fit.params.len() {
        println!(
            "{:<10}{:>12.4}{:>12.4}{:>12.4}{:>12.4}",
            names[i], fit.params[i], fit.bse[i], fit.tvalues[i], fit.pvalues[i]
        );
    }
    let rmse_in = fit.sigma2.sqrt();
    println!(
        "\nsigma2 = {:.3}  (in-sample resid sd = {:.3})   llf = {:.2}   aic = {:.2}   bic = {:.2}",
        fit.sigma2, rmse_in, fit.llf, fit.aic, fit.bic
    );
    println!("nobs used = {}   regressors = {}\n", fit.nobs, fit.df_model);

    // --- 4. Iterated multi-step forecast with a growing prediction band -----
    // Recursion: y_hat[t] = const + trend*(t+1) + sum_l phi_l * y[t-l],
    // feeding forecasts back in as lags become future-dated.
    let c = fit.params[0];
    let trend_coef = fit.params[1];
    let ar: Vec<f64> = (0..lags).map(|j| fit.params[2 + j]).collect();

    // history we extend with forecasts (1-indexed time via t+1 inside loop)
    let mut hist: Vec<f64> = train.to_vec();

    // For Gaussian h-step prediction intervals under an AR model the forecast
    // error variance accumulates the squared psi-weights of the MA(inf) form.
    // We approximate with the recursive psi-weights of the fitted AR poly.
    let z95 = norm_ppf(0.975);
    let mut psi = vec![0.0f64; h]; // psi[0] = 1 handled inline
    let mut forecast = vec![0.0f64; h];
    let mut lo = vec![0.0f64; h];
    let mut hi = vec![0.0f64; h];

    for k in 0..h {
        let t = n_train + k; // absolute time index of the day being forecast
        let time_term = trend_coef * (t as f64 + 1.0);
        let mut yhat = c + time_term;
        for (l, &phi_l) in ar.iter().enumerate() {
            yhat += phi_l * hist[t - 1 - l];
        }
        hist.push(yhat);
        forecast[k] = yhat;

        // psi-weight recursion: psi_k = sum_{j=1..p} phi_j * psi_{k-j}, psi_0=1
        let psi_k = if k == 0 {
            1.0
        } else {
            let mut s = 0.0;
            for (j, &phi_j) in ar.iter().enumerate() {
                let idx = k as isize - 1 - j as isize;
                let prev = if idx < 0 { 0.0 } else { psi[idx as usize] };
                s += phi_j * prev;
            }
            s
        };
        psi[k] = psi_k;
        // var of h-step error = sigma2 * sum_{i=0..k} psi_i^2
        let var: f64 = (0..=k).map(|i| psi[i] * psi[i]).sum::<f64>() * fit.sigma2;
        let band = z95 * var.sqrt();
        lo[k] = yhat - band;
        hi[k] = yhat + band;
    }

    // --- 5. Backtest accuracy on the held-out window ------------------------
    let mut abs_err = 0.0;
    let mut sq_err = 0.0;
    let mut pct_err = 0.0;
    let mut covered = 0usize;
    for k in 0..h {
        let e = forecast[k] - actual_future[k];
        abs_err += e.abs();
        sq_err += e * e;
        pct_err += (e / actual_future[k]).abs();
        if actual_future[k] >= lo[k] && actual_future[k] <= hi[k] {
            covered += 1;
        }
    }
    let mae = abs_err / h as f64;
    let rmse = (sq_err / h as f64).sqrt();
    let mape = 100.0 * pct_err / h as f64;
    let coverage = 100.0 * covered as f64 / h as f64;

    println!("Backtest on the held-out {h} days (model never saw these):");
    println!("  MAE   = {mae:.2} orders/day");
    println!("  RMSE  = {rmse:.2} orders/day");
    println!("  MAPE  = {mape:.2}%");
    println!("  95% band coverage = {coverage:.1}%  ({covered}/{h} actuals inside the interval)\n");

    // --- 6. Turn the forecast into an operational decision ------------------
    // Day-ahead (T+1) and week-ahead (T+7) point forecasts and a staffing
    // recommendation sized to the *upper* 95% bound so we are unlikely to be
    // understaffed.
    let day1 = forecast[0];
    let week1 = forecast[6];
    let week1_hi = hi[6];
    let shifts_point = (week1 / ORDERS_PER_SHIFT).ceil();
    let shifts_safe = (week1_hi / ORDERS_PER_SHIFT).ceil();

    println!("Operational forecast:");
    println!(
        "  day-ahead (T+1)   = {day1:.0} orders   [95% {:.0} .. {:.0}]",
        lo[0], hi[0]
    );
    println!(
        "  week-ahead (T+7)  = {week1:.0} orders   [95% {:.0} .. {:.0}]",
        lo[6], hi[6]
    );
    println!(
        "  staffing @ {ORDERS_PER_SHIFT:.0} orders/shift: plan {shifts_point:.0} shifts for the \
         point forecast,"
    );
    println!(
        "    or {shifts_safe:.0} shifts to cover the 95% upper bound on the week-ahead demand."
    );

    // --- 7. Plot: history, backtest forecast, band, and held-out actuals ----
    let t_all: Vec<f64> = (0..n_total).map(|i| i as f64).collect();

    // training history (solid observed)
    let t_train: Vec<f64> = (0..n_train).map(|i| i as f64).collect();
    let y_train: Vec<f64> = demand[..n_train].to_vec();

    // in-sample fitted values (start after the 7 consumed lags)
    let fit_start = n_train - fit.nobs;
    let t_fit: Vec<f64> = (fit_start..n_train).map(|i| i as f64).collect();
    let y_fit: Vec<f64> = fit.fittedvalues.to_vec();

    // forecast window x: chain from last training day so the line connects
    let mut t_fc: Vec<f64> = vec![(n_train - 1) as f64];
    t_fc.extend((0..h).map(|k| (n_train + k) as f64));
    let mut y_fc: Vec<f64> = vec![*y_train.last().unwrap()];
    y_fc.extend(forecast.iter().copied());

    // held-out actuals
    let t_act: Vec<f64> = (n_train..n_total).map(|i| i as f64).collect();
    let y_act: Vec<f64> = actual_future.to_vec();

    // band over the forecast window
    let t_band: Vec<f64> = (n_train..n_total).map(|i| i as f64).collect();

    let mut fig = Figure::new(960, 560);
    {
        let ax = fig.axes();
        ax.set_title("Daily demand forecast: 20-week history, 3-week backtest")
            .set_xlabel("day")
            .set_ylabel("orders / day")
            .set_grid(true);

        // 95% prediction band (drawn first so it sits behind the lines).
        ax.fill_between(
            &t_band,
            &lo,
            &hi,
            Color::cycle(1),
            0.18,
            Some("95% prediction band"),
        );

        // Observed training history.
        ax.line(
            &t_train,
            &y_train,
            Color::cycle(0),
            1.6,
            LineStyle::Solid,
            Marker::None,
            0.9,
            Some("observed (train)"),
        );

        // In-sample fit.
        ax.line(
            &t_fit,
            &y_fit,
            Color::GREEN,
            1.3,
            LineStyle::Dotted,
            Marker::None,
            0.9,
            Some("in-sample fit"),
        );

        // Forecast path.
        ax.line(
            &t_fc,
            &y_fc,
            Color::RED,
            2.0,
            LineStyle::Dashed,
            Marker::None,
            1.0,
            Some("forecast"),
        );

        // Held-out actuals as points.
        ax.scatter_full(
            &t_act,
            &y_act,
            Color::cycle(3),
            3.5,
            Marker::Circle,
            0.95,
            Some("held-out actual"),
        );

        // Split marker.
        ax.axvline((n_train - 1) as f64, Color::GRAY, LineStyle::Dotted);
        ax.legend(LegendLoc::UpperLeft);
    }

    // keep the unused t_all binding meaningful: assert lengths line up.
    debug_assert_eq!(t_all.len(), n_total);

    let out = common::img_path("case_forecasting.svg");
    fig.save_svg(&out).expect("write case_forecasting.svg");
    eprintln!("wrote {}", out.display());
}

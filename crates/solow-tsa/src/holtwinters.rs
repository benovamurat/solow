//! Exponential smoothing (Holt-Winters) state-space-free recursions.
//!
//! Provides [`SimpleExpSmoothing`], [`Holt`] and the seasonal
//! [`ExponentialSmoothing`] models. Models are fitted by minimising the sum of
//! squared one-step errors over the smoothing parameters, with the initial
//! states fixed by the deterministic heuristic of Hyndman et al. (matching the
//! reference `initialization_method="heuristic"`).
//!
//! The recursions, fitted values and forecasts reproduce the reference
//! `holtwinters` implementation exactly for the supported configurations
//! (additive trend; additive or multiplicative seasonality).

use ndarray::Array1;
use solow_core::error::{Error, Result};
use solow_linalg::lstsq;
use solow_optimize::minimize_bfgs;

const LOWER_BOUND: f64 = 1.4901161193847656e-8; // sqrt(eps)

/// Seasonal component type for [`ExponentialSmoothing`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seasonal {
    /// Additive seasonality `Y = (level + trend) + season`.
    Additive,
    /// Multiplicative seasonality `Y = (level + trend) * season`.
    Multiplicative,
}

/// A fitted exponential-smoothing model.
///
/// Holds the optimal smoothing parameters, the fixed initial states, the
/// in-sample fitted values and the achieved sum of squared errors.
#[derive(Debug, Clone)]
pub struct SmoothingResult {
    /// Smoothing level `alpha`.
    pub alpha: f64,
    /// Smoothing trend `beta` (zero when the model has no trend).
    pub beta: f64,
    /// Smoothing seasonal `gamma` (zero when the model has no seasonality).
    pub gamma: f64,
    /// Initial level `l0`.
    pub initial_level: f64,
    /// Initial trend `b0` (zero when the model has no trend).
    pub initial_trend: f64,
    /// Initial seasonal factors `s0` (empty when the model has no seasonality).
    pub initial_seasons: Array1<f64>,
    /// In-sample one-step fitted values, length `nobs`.
    pub fittedvalues: Array1<f64>,
    /// Sum of squared one-step errors.
    pub sse: f64,
    has_trend: bool,
    seasonal: Option<Seasonal>,
    seasonal_periods: usize,
    endog: Array1<f64>,
}

impl SmoothingResult {
    /// Estimated smoothing parameters in the canonical reference order
    /// `[alpha, beta, gamma, l0, b0, s0...]` restricted to the parameters that
    /// are relevant for this model.
    pub fn params(&self) -> Array1<f64> {
        let mut v = vec![self.alpha];
        if self.has_trend {
            v.push(self.beta);
        }
        if self.seasonal.is_some() {
            v.push(self.gamma);
        }
        v.push(self.initial_level);
        if self.has_trend {
            v.push(self.initial_trend);
        }
        v.extend(self.initial_seasons.iter().copied());
        Array1::from_vec(v)
    }

    /// Out-of-sample point forecasts for horizons `1..=h`.
    pub fn forecast(&self, h: usize) -> Array1<f64> {
        predict(
            &self.endog,
            self.alpha,
            self.beta,
            self.gamma,
            self.initial_level,
            self.initial_trend,
            &self.initial_seasons,
            self.has_trend,
            self.seasonal,
            self.seasonal_periods,
            h,
        )
        .1
    }
}

/// Core recursion reproducing the reference `_predict`. Returns
/// `(fittedvalues, forecasts)` where `fittedvalues` has length `nobs` and
/// `forecasts` has length `h`.
#[allow(clippy::too_many_arguments)]
fn predict(
    y: &Array1<f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    l0: f64,
    b0: f64,
    s0: &Array1<f64>,
    has_trend: bool,
    seasonal: Option<Seasonal>,
    m: usize,
    h: usize,
) -> (Array1<f64>, Array1<f64>) {
    let nobs = y.len();
    let phi = 1.0; // non-damped
    let mut lvls = Array1::<f64>::zeros(nobs + h + 1);
    let mut b = Array1::<f64>::zeros(nobs + h + 1);
    let mut s = Array1::<f64>::zeros(nobs + h + m + 1);
    lvls[0] = l0;
    b[0] = b0;
    for j in 0..m {
        s[j] = s0[j];
    }
    let ac = 1.0 - alpha;
    let bc = 1.0 - beta;
    let gc = 1.0 - gamma;

    // trended(l, db) and detrend depend on whether the model has a trend.
    let trended = |l: f64, db: f64| if has_trend { l + db } else { l };

    match seasonal {
        Some(Seasonal::Multiplicative) => {
            for i in 1..=nobs {
                let prev_t = trended(lvls[i - 1], phi * b[i - 1]);
                lvls[i] = alpha * y[i - 1] / s[i - 1] + ac * prev_t;
                if has_trend {
                    b[i] = beta * (lvls[i] - lvls[i - 1]) + bc * phi * b[i - 1];
                }
                s[i + m - 1] = gamma * y[i - 1] / prev_t + gc * s[i - 1];
            }
        }
        Some(Seasonal::Additive) => {
            for i in 1..=nobs {
                let prev_t = trended(lvls[i - 1], phi * b[i - 1]);
                lvls[i] = alpha * y[i - 1] - alpha * s[i - 1] + ac * prev_t;
                if has_trend {
                    b[i] = beta * (lvls[i] - lvls[i - 1]) + bc * phi * b[i - 1];
                }
                s[i + m - 1] = gamma * y[i - 1] - gamma * prev_t + gc * s[i - 1];
            }
        }
        None => {
            for i in 1..=nobs {
                let prev_t = trended(lvls[i - 1], phi * b[i - 1]);
                lvls[i] = alpha * y[i - 1] + ac * prev_t;
                if has_trend {
                    b[i] = beta * (lvls[i] - lvls[i - 1]) + bc * phi * b[i - 1];
                }
            }
        }
    }

    // Freeze the level and project the trend for the forecast horizon.
    for i in nobs..nobs + h + 1 {
        lvls[i] = lvls[nobs];
    }
    if has_trend {
        // b[:nobs] *= phi (phi=1, no-op); b[nobs:] = b[nobs] * phi_h with
        // phi_h = [1, 2, ..., h+1] for the non-damped case.
        for (k, i) in (nobs..nobs + h + 1).enumerate() {
            b[i] = b[nobs] * (k as f64 + 1.0);
        }
    }

    // Assemble the trend (level + trend) series.
    let mut trend = Array1::<f64>::zeros(nobs + h + 1);
    for i in 0..nobs + h + 1 {
        trend[i] = if has_trend { lvls[i] + b[i] } else { lvls[i] };
    }

    let fitted = match seasonal {
        Some(seas) => {
            // Fill the forecast seasonal slots by repeating the last cycle.
            for j in 0..h + 2 {
                s[nobs + m - 1 + j] = s[(nobs - 1) + j % m];
            }
            // fitted = trend (op) s[:-m]
            let mut f = Array1::<f64>::zeros(nobs + h + 1);
            for i in 0..nobs + h + 1 {
                f[i] = match seas {
                    Seasonal::Additive => trend[i] + s[i],
                    Seasonal::Multiplicative => trend[i] * s[i],
                };
            }
            f
        }
        None => trend,
    };

    let fittedvalues = fitted.slice(ndarray::s![..nobs]).to_owned();
    let forecasts = fitted.slice(ndarray::s![nobs..nobs + h]).to_owned();
    (fittedvalues, forecasts)
}

/// Deterministic heuristic initialization of the states (Hyndman et al.,
/// Section 2.6), matching the reference `_initialization_heuristic`.
fn heuristic_init(
    y: &Array1<f64>,
    has_trend: bool,
    seasonal: Option<Seasonal>,
    m: usize,
) -> Result<(f64, f64, Array1<f64>)> {
    let nobs = y.len();
    if nobs < 10 {
        return Err(Error::Value(
            "Cannot use heuristic method with less than 10 observations.".into(),
        ));
    }

    let mut initial_seasonal = Array1::<f64>::zeros(0);
    // The series on which level/trend are estimated: either y, or, for the
    // seasonal models, the centered moving-average trend of the first cycles.
    let level_series: Array1<f64>;

    if let Some(seas) = seasonal {
        if nobs < 2 * m {
            return Err(Error::Value(
                "Cannot compute initial seasonals with less than two cycles.".into(),
            ));
        }
        let min_obs = 10 + 2 * (m / 2);
        if nobs < min_obs {
            return Err(Error::Value(
                "Cannot use heuristic method: need 10 + 2*(period//2) datapoints.".into(),
            ));
        }
        let mut k_cycles = (5).min(nobs / m);
        let need = (min_obs as f64 / m as f64).ceil() as usize;
        k_cycles = k_cycles.max(need);

        let series_len = m * k_cycles;
        let series = y.slice(ndarray::s![..series_len]).to_owned();

        // Centered rolling mean of window `m`; for even `m`, an extra 2-window
        // average of the shifted series (matching pandas center+shift logic).
        let trend = rolling_center_trend(&series, m);

        // Detrend.
        let detrended: Array1<f64> = match seas {
            Seasonal::Additive => &series - &trend,
            Seasonal::Multiplicative => &series / &trend,
        };

        // Average seasonal effect across cycles (ignoring NaNs).
        let mut seas_avg = Array1::<f64>::zeros(m);
        for (p, sv) in seas_avg.iter_mut().enumerate() {
            let mut sum = 0.0;
            let mut cnt = 0usize;
            let mut idx = p;
            while idx < detrended.len() {
                let v = detrended[idx];
                if v.is_finite() {
                    sum += v;
                    cnt += 1;
                }
                idx += m;
            }
            *sv = if cnt == 0 { f64::NAN } else { sum / cnt as f64 };
        }
        match seas {
            Seasonal::Additive => {
                let mean = seas_avg.sum() / m as f64;
                seas_avg.mapv_inplace(|v| v - mean);
            }
            Seasonal::Multiplicative => {
                let mean = seas_avg.sum() / m as f64;
                seas_avg.mapv_inplace(|v| v / mean);
            }
        }
        initial_seasonal = seas_avg;

        // The level/trend regression uses the (non-NaN) trend values.
        level_series = trend.iter().copied().filter(|v| v.is_finite()).collect();
    } else {
        level_series = y.clone();
    }

    // Regress the first 10 points of level_series on [1, t], t=1..10.
    if level_series.len() < 10 {
        return Err(Error::Value(
            "heuristic initialization requires at least 10 usable points".into(),
        ));
    }
    let mut exog = ndarray::Array2::<f64>::zeros((10, 2));
    for i in 0..10 {
        exog[[i, 0]] = 1.0;
        exog[[i, 1]] = (i + 1) as f64;
    }
    let target = level_series.slice(ndarray::s![..10]).to_owned();
    let beta = lstsq(&exog, &target)?;
    let initial_level = beta[0];
    let initial_trend = if has_trend { beta[1] } else { 0.0 };

    Ok((initial_level, initial_trend, initial_seasonal))
}

/// Centered rolling mean used by the seasonal heuristic. For even `m` the
/// pandas recipe is `rolling(m, center).mean()` followed by
/// `shift(-1).rolling(2).mean()`.
fn rolling_center_trend(series: &Array1<f64>, m: usize) -> Array1<f64> {
    let n = series.len();
    let mut t = Array1::<f64>::from_elem(n, f64::NAN);
    // pandas center offset: for window m, label index = right_edge - (m-1)//2.
    let off = (m - 1) / 2;
    for end in (m - 1)..n {
        let start = end + 1 - m;
        let sum: f64 = series.slice(ndarray::s![start..=end]).sum();
        let label = end - off;
        t[label] = sum / m as f64;
    }
    if m % 2 == 0 {
        // shift(-1) then rolling(2).mean(): t2[i] = (t[i] + t[i+1]) / 2.
        let mut t2 = Array1::<f64>::from_elem(n, f64::NAN);
        for i in 0..n {
            let j = i + 1;
            if j < n && t[i].is_finite() && t[j].is_finite() {
                t2[i] = (t[i] + t[j]) / 2.0;
            }
        }
        t2
    } else {
        t
    }
}

/// Sum of squared one-step errors for a parameter triple.
#[allow(clippy::too_many_arguments)]
fn sse_for(
    y: &Array1<f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    l0: f64,
    b0: f64,
    s0: &Array1<f64>,
    has_trend: bool,
    seasonal: Option<Seasonal>,
    m: usize,
) -> f64 {
    let (fitted, _) = predict(y, alpha, beta, gamma, l0, b0, s0, has_trend, seasonal, m, 0);
    let mut sse = 0.0;
    for i in 0..y.len() {
        let e = y[i] - fitted[i];
        sse += e * e;
    }
    sse
}

// Map an unconstrained real to the open unit interval and back, used so that
// the [0,1] "unrestricted" reference coordinates can be optimised by
// unconstrained BFGS.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply the reference `to_restricted` map to `(u_alpha, u_beta, u_gamma)`
/// where each `u_*` lies in `(0, 1)`.
fn to_restricted(ua: f64, ub: f64, ug: f64, free_b: bool, free_g: bool) -> (f64, f64, f64) {
    let lb_a = LOWER_BOUND;
    let ub_a = 1.0 - LOWER_BOUND;
    let alpha = lb_a + ua * (ub_a - lb_a);
    let beta = if free_b {
        0.0 + ub * (alpha - 0.0)
    } else {
        0.0
    };
    let gamma = if free_g {
        0.0 + ug * ((1.0 - alpha) - 0.0)
    } else {
        0.0
    };
    (alpha, beta, gamma)
}

#[allow(clippy::too_many_arguments)]
fn optimize(
    y: &Array1<f64>,
    has_trend: bool,
    seasonal: Option<Seasonal>,
    m: usize,
    l0: f64,
    b0: f64,
    s0: &Array1<f64>,
) -> (f64, f64, f64) {
    let free_b = has_trend;
    let free_g = seasonal.is_some();
    let nfree = 1 + free_b as usize + free_g as usize;

    // Objective in unconstrained R^nfree (mapped through sigmoid -> to_restricted).
    let eval = |z: &Array1<f64>| -> f64 {
        let ua = sigmoid(z[0]);
        let mut idx = 1;
        let ub = if free_b {
            let v = sigmoid(z[idx]);
            idx += 1;
            v
        } else {
            0.0
        };
        let ug = if free_g { sigmoid(z[idx]) } else { 0.0 };
        let (a, b, g) = to_restricted(ua, ub, ug, free_b, free_g);
        sse_for(y, a, b, g, l0, b0, s0, has_trend, seasonal, m)
    };

    // Coarse grid search over the restricted unit coordinates to seed BFGS,
    // mirroring the reference brute-force start.
    let grid = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
    let mut best_u = (0.5, 0.5, 0.5);
    let mut best_val = f64::INFINITY;
    for &ua in &grid {
        let bs: &[f64] = if free_b { &grid } else { &[0.0] };
        for &ub in bs {
            let gs: &[f64] = if free_g { &grid } else { &[0.0] };
            for &ug in gs {
                let (a, b, g) = to_restricted(ua, ub, ug, free_b, free_g);
                let v = sse_for(y, a, b, g, l0, b0, s0, has_trend, seasonal, m);
                if v < best_val {
                    best_val = v;
                    best_u = (ua, ub, ug);
                }
            }
        }
    }

    let logit = |p: f64| {
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln()
    };
    let mut z0 = Vec::with_capacity(nfree);
    z0.push(logit(best_u.0));
    if free_b {
        z0.push(logit(best_u.1));
    }
    if free_g {
        z0.push(logit(best_u.2));
    }
    let start = Array1::from_vec(z0);

    let grad = |z: &Array1<f64>| -> Array1<f64> {
        let mut g = Array1::<f64>::zeros(z.len());
        let f0 = eval(z);
        for j in 0..z.len() {
            let step = 1e-7 * (1.0 + z[j].abs());
            let mut zp = z.clone();
            zp[j] += step;
            g[j] = (eval(&zp) - f0) / step;
        }
        g
    };

    let res = minimize_bfgs(&start, |z| eval(z), grad, 2000, 1e-10).expect("bfgs should not fail");

    // Decode the optimum.
    let z = &res.x;
    let ua = sigmoid(z[0]);
    let mut idx = 1;
    let ub = if free_b {
        let v = sigmoid(z[idx]);
        idx += 1;
        v
    } else {
        0.0
    };
    let ug = if free_g { sigmoid(z[idx]) } else { 0.0 };
    to_restricted(ua, ub, ug, free_b, free_g)
}

/// Build a fitted [`SmoothingResult`] for the given configuration.
fn fit_model(
    y: Array1<f64>,
    has_trend: bool,
    seasonal: Option<Seasonal>,
    m: usize,
) -> Result<SmoothingResult> {
    if y.len() < 2 {
        return Err(Error::Value("need at least 2 observations".into()));
    }
    if (seasonal == Some(Seasonal::Multiplicative)) && y.iter().any(|&v| v <= 0.0) {
        return Err(Error::Value(
            "endog must be strictly positive for multiplicative seasonality".into(),
        ));
    }
    let (l0, b0, s0) = heuristic_init(&y, has_trend, seasonal, m)?;
    let (alpha, beta, gamma) = optimize(&y, has_trend, seasonal, m, l0, b0, &s0);
    let (fittedvalues, _) = predict(
        &y, alpha, beta, gamma, l0, b0, &s0, has_trend, seasonal, m, 0,
    );
    let mut sse = 0.0;
    for i in 0..y.len() {
        let e = y[i] - fittedvalues[i];
        sse += e * e;
    }
    Ok(SmoothingResult {
        alpha,
        beta,
        gamma,
        initial_level: l0,
        initial_trend: b0,
        initial_seasons: s0,
        fittedvalues,
        sse,
        has_trend,
        seasonal,
        seasonal_periods: m,
        endog: y,
    })
}

/// Simple exponential smoothing (no trend, no seasonality).
#[derive(Debug, Clone)]
pub struct SimpleExpSmoothing {
    endog: Array1<f64>,
}

impl SimpleExpSmoothing {
    /// Create a model for the given series.
    pub fn new(endog: Array1<f64>) -> Self {
        Self { endog }
    }

    /// Fit by minimising the sum of squared one-step errors over `alpha`.
    pub fn fit(&self) -> Result<SmoothingResult> {
        fit_model(self.endog.clone(), false, None, 0)
    }
}

/// Holt's linear-trend method (additive trend, no seasonality).
#[derive(Debug, Clone)]
pub struct Holt {
    endog: Array1<f64>,
}

impl Holt {
    /// Create a model for the given series.
    pub fn new(endog: Array1<f64>) -> Self {
        Self { endog }
    }

    /// Fit by minimising the sum of squared one-step errors over
    /// `(alpha, beta)`.
    pub fn fit(&self) -> Result<SmoothingResult> {
        fit_model(self.endog.clone(), true, None, 0)
    }
}

/// Holt-Winters exponential smoothing with additive trend and optional
/// additive or multiplicative seasonality.
#[derive(Debug, Clone)]
pub struct ExponentialSmoothing {
    endog: Array1<f64>,
    trend: bool,
    seasonal: Option<Seasonal>,
    seasonal_periods: usize,
}

impl ExponentialSmoothing {
    /// Create a Holt-Winters model.
    ///
    /// `trend` enables an additive trend component. `seasonal` (with
    /// `seasonal_periods > 1`) enables additive/multiplicative seasonality.
    pub fn new(
        endog: Array1<f64>,
        trend: bool,
        seasonal: Option<Seasonal>,
        seasonal_periods: usize,
    ) -> Result<Self> {
        if seasonal.is_some() && seasonal_periods <= 1 {
            return Err(Error::Value(
                "seasonal_periods must be larger than 1".into(),
            ));
        }
        Ok(Self {
            endog,
            trend,
            seasonal,
            seasonal_periods,
        })
    }

    /// Fit by minimising the sum of squared one-step errors over the active
    /// smoothing parameters.
    pub fn fit(&self) -> Result<SmoothingResult> {
        let m = if self.seasonal.is_some() {
            self.seasonal_periods
        } else {
            0
        };
        fit_model(self.endog.clone(), self.trend, self.seasonal, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_series(n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |i| 10.0 + 0.5 * i as f64)
    }

    #[test]
    fn ses_fits_and_forecasts_flat() {
        let y = linear_series(20);
        let res = SimpleExpSmoothing::new(y).fit().unwrap();
        assert!(res.sse >= 0.0);
        let fc = res.forecast(3);
        assert_eq!(fc.len(), 3);
        // SES forecasts are flat.
        assert!((fc[0] - fc[1]).abs() < 1e-9);
        assert!((fc[1] - fc[2]).abs() < 1e-9);
    }

    #[test]
    fn holt_forecast_is_linear() {
        let y = linear_series(20);
        let res = Holt::new(y).fit().unwrap();
        let fc = res.forecast(3);
        // Holt forecasts grow linearly: equal first differences.
        let d1 = fc[1] - fc[0];
        let d2 = fc[2] - fc[1];
        assert!((d1 - d2).abs() < 1e-6);
    }

    #[test]
    fn seasonal_requires_period_gt_one() {
        let y = linear_series(20);
        assert!(ExponentialSmoothing::new(y, true, Some(Seasonal::Additive), 1).is_err());
    }

    #[test]
    fn additive_seasonal_fits() {
        let pattern = [2.0, -1.0, -3.0, 2.0];
        let y = Array1::from_shape_fn(40, |i| 20.0 + 0.3 * i as f64 + pattern[i % 4]);
        let res = ExponentialSmoothing::new(y, true, Some(Seasonal::Additive), 4)
            .unwrap()
            .fit()
            .unwrap();
        assert_eq!(res.initial_seasons.len(), 4);
        assert!(res.sse >= 0.0);
        assert_eq!(res.forecast(8).len(), 8);
    }
}

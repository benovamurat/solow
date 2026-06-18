//! # solow-graphics
//!
//! Statistical-graphics helpers that compute the data behind a plot and render
//! it through the [`solow_viz`] SVG backend. Every routine returns *both* the
//! rendered [`Figure`] and the computed arrays, so the numerics can be tested
//! independently of the (intentionally un-pixel-exact) SVG output.
//!
//! Provided:
//!
//! * [`ProbPlot`] / [`qqplot`] — theoretical vs. sample quantiles of a
//!   probability plot, plus the fitted reference line ([`QqLine`]).
//! * [`plot_acf`] / [`plot_pacf`] — the (biased) autocorrelation and the
//!   Yule-Walker partial autocorrelation, with a white-noise confidence band.
//! * [`plot_resid_fitted`] — a residuals-vs-fitted diagnostic scatter.
//!
//! ```
//! use solow_graphics::ProbPlot;
//! let data = [-1.2, 0.3, 0.1, 1.4, -0.7, 2.1, -0.2, 0.9];
//! let pp = ProbPlot::new(&data);
//! assert_eq!(pp.sample_quantiles().len(), data.len());
//! let line = pp.qqline_regression();
//! let svg = pp.qqplot().to_svg();
//! assert!(svg.starts_with("<svg"));
//! let _ = line.slope;
//! ```

use ndarray::Array1;
use solow_distributions::norm_ppf;
use solow_viz::{Color, Figure};

mod influence;
pub use influence::{influence_plot, mosaic, plot_fit, plot_regress_exog, Influence, MosaicData};

/// The fitted reference line of a probability plot, `y = slope * x + intercept`,
/// where `x` are the theoretical quantiles and `y` the sample quantiles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QqLine {
    /// Slope of the reference line.
    pub slope: f64,
    /// Intercept of the reference line.
    pub intercept: f64,
}

/// A normal probability plot (Q-Q plot against the standard normal).
///
/// Mirrors the reference `ProbPlot` for the default standard-normal
/// distribution: the theoretical percentiles are the plotting positions
/// `(i - a) / (n + 1 - 2a)` for `i = 1..=n`, the theoretical quantiles are the
/// normal inverse-CDF of those percentiles, and the sample quantiles are simply
/// the sorted data.
#[derive(Clone, Debug)]
pub struct ProbPlot {
    sorted: Vec<f64>,
    a: f64,
}

impl ProbPlot {
    /// Build a probability plot from `data`, using plotting-position parameter
    /// `a = 0` (the reference default).
    pub fn new(data: &[f64]) -> Self {
        Self::with_a(data, 0.0)
    }

    /// Build a probability plot with an explicit plotting-position parameter `a`.
    ///
    /// Common choices: `0.0` (Weibull, the default), `0.375` (Blom),
    /// `0.5` (Hazen).
    pub fn with_a(data: &[f64], a: f64) -> Self {
        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|x, y| x.partial_cmp(y).expect("data must not contain NaN"));
        ProbPlot { sorted, a }
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.sorted.len()
    }

    /// The theoretical plotting positions (percentiles in `(0, 1)`).
    ///
    /// `p_i = (i - a) / (n + 1 - 2a)` for `i = 1..=n`.
    pub fn theoretical_percentiles(&self) -> Array1<f64> {
        let n = self.sorted.len() as f64;
        let denom = n + 1.0 - 2.0 * self.a;
        Array1::from_iter((1..=self.sorted.len()).map(|i| (i as f64 - self.a) / denom))
    }

    /// The theoretical quantiles: the standard-normal inverse CDF of the
    /// plotting positions.
    pub fn theoretical_quantiles(&self) -> Array1<f64> {
        self.theoretical_percentiles().mapv(norm_ppf)
    }

    /// The sample quantiles: the sorted data.
    pub fn sample_quantiles(&self) -> Array1<f64> {
        Array1::from_vec(self.sorted.clone())
    }

    /// The regression reference line ("r"): an ordinary least-squares fit of
    /// the sample quantiles on the theoretical quantiles (with intercept).
    pub fn qqline_regression(&self) -> QqLine {
        let x = self.theoretical_quantiles();
        let y = self.sample_quantiles();
        // SAFETY: owned contiguous arrays from the quantile helpers.
        let (slope, intercept) = ols_line(x.as_slice().unwrap_or(&[]), y.as_slice().unwrap_or(&[]));
        QqLine { slope, intercept }
    }

    /// The standardized reference line ("s"): slope = sample standard deviation
    /// (population, `1/n`), intercept = sample mean.
    pub fn qqline_standardized(&self) -> QqLine {
        let y = &self.sorted;
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let var = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        QqLine {
            slope: var.sqrt(),
            intercept: mean,
        }
    }

    /// The quartile reference line ("q"): the line through the first and third
    /// quartiles of the sample versus the theoretical normal quartiles.
    pub fn qqline_quartile(&self) -> QqLine {
        let q25 = score_at_percentile(&self.sorted, 25.0);
        let q75 = score_at_percentile(&self.sorted, 75.0);
        let t25 = norm_ppf(0.25);
        let t75 = norm_ppf(0.75);
        let slope = (q75 - q25) / (t75 - t25);
        let intercept = q25 - slope * t25;
        QqLine { slope, intercept }
    }

    /// Render the Q-Q plot to a [`Figure`]: a scatter of (theoretical, sample)
    /// quantiles overlaid with the regression reference line.
    pub fn qqplot(&self) -> Figure {
        let theo = self.theoretical_quantiles();
        let samp = self.sample_quantiles();
        let line = self.qqline_regression();

        let mut fig = Figure::new(640, 480);
        let ax = fig.axes();
        ax.set_title("Q-Q plot")
            .set_xlabel("Theoretical quantiles")
            .set_ylabel("Sample quantiles")
            .set_grid(true);
        // SAFETY: owned contiguous arrays from the quantile helpers.
        let theo_s = theo.as_slice().unwrap_or(&[]);
        ax.scatter(theo_s, samp.as_slice().unwrap_or(&[]));
        if let (Some(&lo), Some(&hi)) = (theo_s.first(), theo_s.last()) {
            let xs = [lo, hi];
            let ys = [
                line.slope * lo + line.intercept,
                line.slope * hi + line.intercept,
            ];
            ax.plot_styled(&xs, &ys, Color::RED, 1.6);
        }
        fig
    }
}

/// Convenience wrapper: build a [`ProbPlot`] and render its Q-Q plot.
pub fn qqplot(data: &[f64]) -> (Figure, ProbPlot) {
    let pp = ProbPlot::new(data);
    let fig = pp.qqplot();
    (fig, pp)
}

/// Result of an autocorrelation/partial-autocorrelation computation.
#[derive(Clone, Debug)]
pub struct AcfResult {
    /// The correlation values, lag `0..=nlags` (index 0 is always `1.0`).
    pub values: Array1<f64>,
    /// The (symmetric) confidence-band half-width `z_{alpha/2} / sqrt(n)`.
    pub conf_band: f64,
}

/// The biased autocorrelation function for lags `0..=nlags`.
///
/// Uses the biased (divide-by-`n`) autocovariance estimator
/// `gamma_k = (1/n) sum_{t=k}^{n-1} (x_t - xbar)(x_{t-k} - xbar)`, then
/// `acf_k = gamma_k / gamma_0`.
pub fn acf(x: &[f64], nlags: usize) -> Array1<f64> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let xc: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let g0: f64 = xc.iter().map(|v| v * v).sum::<f64>() / n as f64;
    let mut out = Vec::with_capacity(nlags + 1);
    for k in 0..=nlags {
        let mut s = 0.0;
        for t in k..n {
            s += xc[t] * xc[t - k];
        }
        out.push((s / n as f64) / g0);
    }
    Array1::from_vec(out)
}

/// The Yule-Walker partial autocorrelation function (adjusted / unbiased
/// autocovariances), lags `0..=nlags`.
///
/// For each order `k`, solves the Yule-Walker equations using autocovariances
/// estimated with divisor `n - lag` (the "adjusted" estimator), and takes the
/// last coefficient as the partial autocorrelation at lag `k`. Index 0 is
/// `1.0` by convention.
pub fn pacf_yw(x: &[f64], nlags: usize) -> Array1<f64> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let xc: Vec<f64> = x.iter().map(|v| v - mean).collect();
    // Adjusted autocovariances: divisor (n - k).
    let mut acov = vec![0.0_f64; nlags + 1];
    for (k, ac) in acov.iter_mut().enumerate() {
        let mut s = 0.0;
        for t in k..n {
            s += xc[t] * xc[t - k];
        }
        *ac = s / (n - k) as f64;
    }
    let mut out = vec![0.0_f64; nlags + 1];
    for (k, slot) in out.iter_mut().enumerate() {
        *slot = if k == 0 {
            1.0
        } else {
            yule_walker_last(&acov, k)
        };
    }
    Array1::from_vec(out)
}

/// Solve the order-`k` Yule-Walker system `R phi = r` for autocovariances
/// `acov[0..=k]` and return the last coefficient `phi_k` (the PACF at lag `k`).
fn yule_walker_last(acov: &[f64], k: usize) -> f64 {
    // Toeplitz system: R[i][j] = acov[|i-j|], rhs r[i] = acov[i+1].
    let mut r = vec![vec![0.0_f64; k]; k];
    let mut rhs = vec![0.0_f64; k];
    for i in 0..k {
        for j in 0..k {
            r[i][j] = acov[i.abs_diff(j)];
        }
        rhs[i] = acov[i + 1];
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..k {
        let mut piv = col;
        for row in (col + 1)..k {
            if r[row][col].abs() > r[piv][col].abs() {
                piv = row;
            }
        }
        r.swap(col, piv);
        rhs.swap(col, piv);
        let pivot_row = r[col].clone();
        let d = pivot_row[col];
        let pivot_rhs = rhs[col];
        for row in (col + 1)..k {
            let f = r[row][col] / d;
            for (rc, &pc) in r[row].iter_mut().zip(pivot_row.iter()).skip(col) {
                *rc -= f * pc;
            }
            rhs[row] -= f * pivot_rhs;
        }
    }
    // Back-substitution; we only need phi[k-1].
    let mut phi = vec![0.0_f64; k];
    for row in (0..k).rev() {
        let mut s = rhs[row];
        for c in (row + 1)..k {
            s -= r[row][c] * phi[c];
        }
        phi[row] = s / r[row][row];
    }
    phi[k - 1]
}

/// Compute the ACF and a `(1 - alpha)` white-noise confidence band, and render
/// a stem-style plot. Returns the [`Figure`] and the [`AcfResult`].
pub fn plot_acf(x: &[f64], nlags: usize, alpha: f64) -> (Figure, AcfResult) {
    let values = acf(x, nlags);
    let band = conf_band(x.len(), alpha);
    let fig = render_corr(&values, band, "Autocorrelation");
    (
        fig,
        AcfResult {
            values,
            conf_band: band,
        },
    )
}

/// Compute the (Yule-Walker) PACF and a `(1 - alpha)` white-noise confidence
/// band, and render a stem-style plot. Returns the [`Figure`] and the
/// [`AcfResult`].
pub fn plot_pacf(x: &[f64], nlags: usize, alpha: f64) -> (Figure, AcfResult) {
    let values = pacf_yw(x, nlags);
    let band = conf_band(x.len(), alpha);
    let fig = render_corr(&values, band, "Partial Autocorrelation");
    (
        fig,
        AcfResult {
            values,
            conf_band: band,
        },
    )
}

/// The symmetric white-noise confidence half-width `z_{alpha/2} / sqrt(n)`.
pub fn conf_band(n: usize, alpha: f64) -> f64 {
    let z = norm_ppf(1.0 - alpha / 2.0);
    z / (n as f64).sqrt()
}

/// A residuals-vs-fitted diagnostic. Takes the model's fitted values and
/// residuals, returns the rendered [`Figure`] (a zero-reference line is drawn
/// at `resid = 0`).
pub fn plot_resid_fitted(fitted: &[f64], resid: &[f64]) -> Figure {
    assert_eq!(
        fitted.len(),
        resid.len(),
        "fitted and resid length mismatch"
    );
    let mut fig = Figure::new(640, 480);
    let ax = fig.axes();
    ax.set_title("Residuals vs Fitted")
        .set_xlabel("Fitted values")
        .set_ylabel("Residuals")
        .set_grid(true);
    ax.scatter(fitted, resid);
    if let (Some(&lo), Some(&hi)) = (
        fitted.iter().min_by(|a, b| a.total_cmp(b)),
        fitted.iter().max_by(|a, b| a.total_cmp(b)),
    ) {
        ax.plot_styled(&[lo, hi], &[0.0, 0.0], Color::GRAY, 1.0);
    }
    fig
}

// --- internal helpers ------------------------------------------------------

/// Render a stem-style correlation plot (markers at each lag plus a horizontal
/// confidence band).
fn render_corr(values: &Array1<f64>, band: f64, title: &str) -> Figure {
    let lags: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
    let mut fig = Figure::new(640, 480);
    let ax = fig.axes();
    ax.set_title(title).set_xlabel("Lag").set_grid(true);
    // Stems.
    for (i, &v) in values.iter().enumerate() {
        ax.plot_styled(&[i as f64, i as f64], &[0.0, v], Color::BLUE, 1.2);
    }
    // SAFETY: owned contiguous correlation array.
    ax.scatter_styled(&lags, values.as_slice().unwrap_or(&[]), Color::BLUE, 3.0);
    let xmax = (values.len() - 1) as f64;
    ax.plot_styled(&[0.0, xmax], &[band, band], Color::GRAY, 1.0);
    ax.plot_styled(&[0.0, xmax], &[-band, -band], Color::GRAY, 1.0);
    fig
}

/// Ordinary least squares of `y` on `x` with an intercept, returning
/// `(slope, intercept)`.
fn ols_line(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        sxx += (xi - mx) * (xi - mx);
        sxy += (xi - mx) * (yi - my);
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    (slope, intercept)
}

/// The score at the given `percentile` of `sorted` data, using linear
/// interpolation between order statistics (matching the reference
/// `scoreatpercentile` default, `interpolation_method="fraction"`).
fn score_at_percentile(sorted: &[f64], percentile: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = percentile / 100.0 * (n as f64 - 1.0);
    let lo = idx.floor() as usize;
    let frac = idx - lo as f64;
    if lo + 1 >= n {
        sorted[n - 1]
    } else {
        sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn acf_lag0_is_one() {
        let x = [1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0];
        let a = acf(&x, 4);
        assert_relative_eq!(a[0], 1.0, max_relative = 1e-15);
        for &v in a.iter() {
            assert!((-1.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn acf_matches_manual() {
        let x = [0.5, -1.0, 2.0, 0.3, -0.7, 1.1];
        let n = x.len();
        let mean = x.iter().sum::<f64>() / n as f64;
        let xc: Vec<f64> = x.iter().map(|v| v - mean).collect();
        let g0: f64 = xc.iter().map(|v| v * v).sum::<f64>() / n as f64;
        let g1: f64 = (1..n).map(|t| xc[t] * xc[t - 1]).sum::<f64>() / n as f64;
        let a = acf(&x, 1);
        assert_relative_eq!(a[1], g1 / g0, max_relative = 1e-13);
    }

    #[test]
    fn pacf_lag1_matches_adjusted_acov_ratio() {
        let x = [0.5, -1.0, 2.0, 0.3, -0.7, 1.1, 0.9, -0.2];
        let p = pacf_yw(&x, 3);
        assert_relative_eq!(p[0], 1.0, max_relative = 1e-15);
        let n = x.len();
        let mean = x.iter().sum::<f64>() / n as f64;
        let xc: Vec<f64> = x.iter().map(|v| v - mean).collect();
        let a0: f64 = xc.iter().map(|v| v * v).sum::<f64>() / n as f64;
        let a1: f64 = (1..n).map(|t| xc[t] * xc[t - 1]).sum::<f64>() / (n - 1) as f64;
        assert_relative_eq!(p[1], a1 / a0, max_relative = 1e-12);
    }

    #[test]
    fn probplot_sorts_and_sizes() {
        let data = [3.0, 1.0, 2.0, -1.0];
        let pp = ProbPlot::new(&data);
        let s = pp.sample_quantiles();
        assert_eq!(s.as_slice().unwrap(), &[-1.0, 1.0, 2.0, 3.0]);
        assert_eq!(pp.theoretical_quantiles().len(), 4);
        let t = pp.theoretical_quantiles();
        assert_relative_eq!(t[0], -t[3], max_relative = 1e-12);
    }

    #[test]
    fn conf_band_known_value() {
        let b = conf_band(100, 0.05);
        assert_relative_eq!(b, 1.959963984540054 / 10.0, max_relative = 1e-12);
    }

    #[test]
    fn qqplot_svg_structural() {
        let data = [-1.0, 0.0, 0.5, 1.5, -0.3, 0.8];
        let (fig, _pp) = qqplot(&data);
        let svg = fig.to_svg();
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("circle") || svg.contains("<line"));
    }

    #[test]
    fn ols_line_recovers_exact() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [3.0, 5.0, 7.0, 9.0, 11.0];
        let (m, b) = ols_line(&x, &y);
        assert_relative_eq!(m, 2.0, max_relative = 1e-12);
        assert_relative_eq!(b, 3.0, max_relative = 1e-12);
    }

    #[test]
    fn resid_fitted_svg_structural() {
        let fitted = [1.0, 2.0, 3.0, 4.0];
        let resid = [0.1, -0.2, 0.05, -0.1];
        let svg = plot_resid_fitted(&fitted, &resid).to_svg();
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("</svg>"));
    }
}

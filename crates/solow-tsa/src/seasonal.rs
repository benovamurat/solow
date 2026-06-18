//! Classical seasonal decomposition by moving averages.
//!
//! Mirrors the reference `seasonal_decompose(x, period, model)` for the
//! two-sided (centered) moving-average filter, returning the trend, seasonal
//! and residual components. Components that cannot be estimated near the
//! sample edges are returned as `NaN`, exactly as the reference does.

use ndarray::Array1;
use solow_core::error::{Error, Result};

/// Decomposition model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeasonalModel {
    /// `Y[t] = T[t] + S[t] + e[t]`.
    Additive,
    /// `Y[t] = T[t] * S[t] * e[t]`.
    Multiplicative,
}

impl SeasonalModel {
    /// Parse a model code (`"additive"`/`"add"` or `"multiplicative"`/`"mul"`).
    pub fn parse(s: &str) -> Result<Self> {
        if s.starts_with('a') {
            Ok(SeasonalModel::Additive)
        } else if s.starts_with('m') {
            Ok(SeasonalModel::Multiplicative)
        } else {
            Err(Error::Value(format!("unknown model '{s}'")))
        }
    }
}

/// The three components produced by [`seasonal_decompose`].
#[derive(Debug, Clone)]
pub struct DecomposeResult {
    /// Estimated trend-cycle (NaN where the centered filter has no support).
    pub trend: Array1<f64>,
    /// Estimated seasonal component (defined for every observation).
    pub seasonal: Array1<f64>,
    /// Residual component (NaN wherever the trend is NaN).
    pub resid: Array1<f64>,
}

fn nanmean(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &v in values {
        if v.is_finite() {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Centered moving-average trend, matching the reference `convolution_filter`
/// with `nsides = 2`. The filter weights are `[0.5, 1, ..., 1, 0.5] / period`
/// when `period` is even, and `[1/period; period]` when odd.
fn centered_trend(x: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = x.len();
    let filt: Vec<f64> = if period % 2 == 0 {
        let mut f = vec![1.0 / period as f64; period + 1];
        f[0] = 0.5 / period as f64;
        f[period] = 0.5 / period as f64;
        f
    } else {
        vec![1.0 / period as f64; period]
    };
    let nf = filt.len();
    // trim_head = ceil(nf/2) - 1; trim_tail = ceil(nf/2) - (nf % 2)
    let ceil_half = nf.div_ceil(2);
    let trim_head = ceil_half - 1;
    let trim_tail = ceil_half - (nf % 2);

    let mut trend = Array1::<f64>::from_elem(n, f64::NAN);
    // Valid convolution: for output index i in [trim_head, n - trim_tail),
    // trend[i] = sum_k filt[k] * x[i - trim_head + k].
    let last = n.saturating_sub(trim_tail);
    for i in trim_head..last {
        let mut acc = 0.0;
        for (k, &w) in filt.iter().enumerate() {
            acc += w * x[i - trim_head + k];
        }
        trend[i] = acc;
    }
    trend
}

/// Classical seasonal decomposition via centered moving averages.
///
/// `period` is the seasonal period and must satisfy `x.len() >= 2 * period`.
/// For the multiplicative model all observations must be strictly positive.
pub fn seasonal_decompose(
    x: &Array1<f64>,
    period: usize,
    model: SeasonalModel,
) -> Result<DecomposeResult> {
    let n = x.len();
    if period < 2 {
        return Err(Error::Value("period must be at least 2".into()));
    }
    if n < 2 * period {
        return Err(Error::Value(
            "x must have 2 complete cycles (len >= 2 * period)".into(),
        ));
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(Error::Value(
            "This function does not handle missing values".into(),
        ));
    }
    if model == SeasonalModel::Multiplicative && x.iter().any(|&v| v <= 0.0) {
        return Err(Error::Value(
            "Multiplicative seasonality is not appropriate for zero and negative values".into(),
        ));
    }

    let trend = centered_trend(x, period);

    // Detrend.
    let detrended: Array1<f64> = match model {
        SeasonalModel::Additive => x - &trend,
        SeasonalModel::Multiplicative => x / &trend,
    };

    // Average each phase of the cycle (NaNs ignored).
    let mut period_averages = vec![0.0_f64; period];
    for (p, avg) in period_averages.iter_mut().enumerate() {
        let phase: Vec<f64> = (p..n).step_by(period).map(|i| detrended[i]).collect();
        *avg = nanmean(&phase);
    }
    // Normalise the seasonal averages.
    match model {
        SeasonalModel::Additive => {
            let m = period_averages.iter().sum::<f64>() / period as f64;
            for v in period_averages.iter_mut() {
                *v -= m;
            }
        }
        SeasonalModel::Multiplicative => {
            let m = period_averages.iter().sum::<f64>() / period as f64;
            for v in period_averages.iter_mut() {
                *v /= m;
            }
        }
    }

    // Tile across the whole sample.
    let seasonal = Array1::from_shape_fn(n, |i| period_averages[i % period]);

    let resid: Array1<f64> = match model {
        SeasonalModel::Additive => &detrended - &seasonal,
        SeasonalModel::Multiplicative => x / &seasonal / &trend,
    };

    Ok(DecomposeResult {
        trend,
        seasonal,
        resid,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn additive_recovers_seasonality() {
        // Period-4 deterministic seasonal pattern with a linear trend.
        let pattern = [2.0, -1.0, -3.0, 2.0];
        let n = 24;
        let x = Array1::from_shape_fn(n, |i| 10.0 + 0.5 * i as f64 + pattern[i % 4]);
        let r = seasonal_decompose(&x, 4, SeasonalModel::Additive).unwrap();
        // Seasonal component sums to ~0 over one cycle.
        let s: f64 = (0..4).map(|i| r.seasonal[i]).sum();
        assert!(s.abs() < 1e-9);
        // Trend is NaN at the very ends but finite in the middle.
        assert!(r.trend[0].is_nan());
        assert!(r.trend[n / 2].is_finite());
    }

    #[test]
    fn rejects_short_series() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(seasonal_decompose(&x, 4, SeasonalModel::Additive).is_err());
    }

    #[test]
    fn rejects_nonpositive_multiplicative() {
        let x = Array1::from_shape_fn(12, |i| if i == 3 { -1.0 } else { 1.0 + i as f64 });
        assert!(seasonal_decompose(&x, 4, SeasonalModel::Multiplicative).is_err());
    }
}

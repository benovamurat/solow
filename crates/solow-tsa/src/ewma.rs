//! Exponentially-weighted moving average control chart (Roberts 1959).

use ndarray::{Array1, ArrayView1};
use solow_core::{Error, Result};

/// A single EWMA alarm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EwmaAlarm {
    /// Sample index where the alarm fires.
    pub index: usize,
    /// EWMA statistic at the alarm.
    pub statistic: f64,
    /// Sign of the excursion (`+1` = above upper limit, `-1` = below lower).
    pub direction: i32,
}

/// Fitted EWMA control-chart run.
#[derive(Clone, Debug, PartialEq)]
pub struct EwmaResult {
    /// EWMA statistic series (length `n`).
    pub ewma: Array1<f64>,
    /// Upper control limit series.
    pub upper: Array1<f64>,
    /// Lower control limit series.
    pub lower: Array1<f64>,
    /// Every out-of-control alarm.
    pub alarms: Vec<EwmaAlarm>,
    /// Smoothing constant λ used.
    pub lambda: f64,
    /// Alarm width `L` in standard-deviation units.
    pub l: f64,
}

/// Run an EWMA control chart.
///
/// * `target` is the in-control mean.
/// * `sigma` is the in-control standard deviation.
/// * `lambda ∈ (0, 1]` is the smoothing constant (0.2 is a common default).
/// * `l` is the alarm width in standard-deviation units (3 is a common default).
pub fn ewma(
    x: ArrayView1<'_, f64>,
    target: f64,
    sigma: f64,
    lambda: f64,
    l: f64,
) -> Result<EwmaResult> {
    if x.is_empty() {
        return Err(Error::Value("ewma: empty series".into()));
    }
    if !(0.0..=1.0).contains(&lambda) || lambda == 0.0 {
        return Err(Error::Value("ewma: lambda must be in (0, 1]".into()));
    }
    if sigma <= 0.0 {
        return Err(Error::Value("ewma: sigma must be > 0".into()));
    }
    let n = x.len();
    let mut z = Array1::<f64>::zeros(n);
    let mut upper = Array1::<f64>::zeros(n);
    let mut lower = Array1::<f64>::zeros(n);
    let mut alarms = Vec::new();
    let mut prev = target;
    for i in 0..n {
        let zi = lambda * x[i] + (1.0 - lambda) * prev;
        z[i] = zi;
        // Time-varying limit — narrows to lambda / (2 - lambda) as i → ∞.
        let one_minus_lambda = 1.0 - lambda;
        let factor = (lambda / (2.0 - lambda))
            * (1.0 - one_minus_lambda.powi(2 * (i + 1) as i32));
        let half_width = l * sigma * factor.sqrt();
        upper[i] = target + half_width;
        lower[i] = target - half_width;
        if zi > upper[i] {
            alarms.push(EwmaAlarm { index: i, statistic: zi, direction: 1 });
        } else if zi < lower[i] {
            alarms.push(EwmaAlarm { index: i, statistic: zi, direction: -1 });
        }
        prev = zi;
    }
    Ok(EwmaResult { ewma: z, upper, lower, alarms, lambda, l })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn ewma_alarms_on_a_clear_shift() {
        let mut v = vec![0.0_f64; 30];
        v.extend(vec![2.0_f64; 30]);
        let x = Array1::from(v);
        let r = ewma(x.view(), 0.0, 1.0, 0.2, 3.0).unwrap();
        assert!(!r.alarms.is_empty());
        assert_eq!(r.alarms[0].direction, 1);
    }
}

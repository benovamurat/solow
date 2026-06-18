//! Linear band-pass and trend/cycle filters for time series.
//!
//! This module ports three classical filters used in business-cycle analysis:
//!
//! - [`hpfilter`]: the Hodrick-Prescott two-sided trend/cycle decomposition,
//!   computed as the ridge solution `trend = (I + lamb K'K)^{-1} x`.
//! - [`bkfilter`]: the Baxter-King symmetric fixed-length band-pass filter,
//!   returning the cyclical component as a centred weighted moving average.
//! - [`cffilter`]: the Christiano-Fitzgerald asymmetric random-walk band-pass
//!   filter, returning the cycle and the implied trend.
//!
//! Each function mirrors the corresponding reference routine in
//! `tsa.filters` to the last bit; see the crate's reference tests for the
//! validated tolerances.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::solve;

/// Hodrick-Prescott filter.
///
/// Separates the series `x` into a smooth trend `T` and a residual cycle by
/// minimising
/// `sum_t (x_t - T_t)^2 + lamb * sum_t ((T_{t+1} - T_t) - (T_t - T_{t-1}))^2`.
/// The closed-form solution is `T = (I + lamb K'K)^{-1} x`, where `K` is the
/// `(nobs - 2) × nobs` second-difference operator. The returned pair is
/// `(cycle, trend)` with `cycle = x - trend`.
///
/// `lamb = 1600` is the usual choice for quarterly data.
pub fn hpfilter(x: &Array1<f64>, lamb: f64) -> Result<(Array1<f64>, Array1<f64>)> {
    let nobs = x.len();
    if nobs < 3 {
        return Err(Error::Value(
            "hpfilter requires at least 3 observations".into(),
        ));
    }
    // A = I + lamb K'K. K'K is the symmetric pentadiagonal Gram matrix of the
    // second-difference operator; we accumulate it row by row so the dense
    // solve matches the reference sparse solve to machine precision.
    let mut a = Array2::<f64>::zeros((nobs, nobs));
    for i in 0..nobs {
        a[[i, i]] += 1.0;
    }
    // Each row r of K (r = 0..nobs-3) has entries K[r,r]=1, K[r,r+1]=-2,
    // K[r,r+2]=1. (K'K)[i,j] = sum_r K[r,i] K[r,j].
    let kcoef = [1.0_f64, -2.0, 1.0];
    for r in 0..(nobs - 2) {
        for (di, &ci) in kcoef.iter().enumerate() {
            for (dj, &cj) in kcoef.iter().enumerate() {
                a[[r + di, r + dj]] += lamb * ci * cj;
            }
        }
    }
    let trend = solve(&a, x)?;
    let cycle = x - &trend;
    Ok((cycle, trend))
}

/// Symmetric weights of the Baxter-King band-pass filter.
///
/// Returns the `2K + 1` weights `a[-K..K]`, already demeaned so they sum to
/// zero. `low`/`high` are the minimum/maximum oscillation periods.
fn bk_weights(low: f64, high: f64, k: usize) -> Array1<f64> {
    let omega_1 = 2.0 * std::f64::consts::PI / high;
    let omega_2 = 2.0 * std::f64::consts::PI / low;
    let n = 2 * k + 1;
    let mut b = Array1::<f64>::zeros(n);
    b[k] = (omega_2 - omega_1) / std::f64::consts::PI;
    for j in 1..=k {
        let jf = j as f64;
        let w = 1.0 / (std::f64::consts::PI * jf) * ((omega_2 * jf).sin() - (omega_1 * jf).sin());
        b[k + j] = w;
        b[k - j] = w; // symmetric
    }
    let mean = b.sum() / n as f64;
    b.mapv_inplace(|v| v - mean);
    b
}

/// Baxter-King fixed-length symmetric band-pass filter.
///
/// Returns the cyclical component as the `mode='valid'` convolution of `x`
/// with the symmetric, demeaned weights; the output has `nobs - 2K`
/// observations (the `K` leading and `K` trailing points are dropped).
///
/// `low`/`high` are the minimum/maximum periodicities retained and `k` is the
/// lead-lag truncation length.
pub fn bkfilter(x: &Array1<f64>, low: f64, high: f64, k: usize) -> Result<Array1<f64>> {
    let nobs = x.len();
    if nobs <= 2 * k {
        return Err(Error::Value(
            "bkfilter requires nobs > 2K observations".into(),
        ));
    }
    let weights = bk_weights(low, high, k);
    let m = weights.len(); // 2K+1
    let out_len = nobs - 2 * k;
    let mut y = Array1::<f64>::zeros(out_len);
    // 'valid' correlation: because the weights are symmetric, convolution and
    // correlation coincide. Output i aligns x[i..i+m] with the weight window.
    for i in 0..out_len {
        let mut s = 0.0;
        for j in 0..m {
            s += weights[j] * x[i + j];
        }
        y[i] = s;
    }
    Ok(y)
}

/// Christiano-Fitzgerald asymmetric random-walk band-pass filter.
///
/// Returns `(cycle, trend)` where `cycle` is the extracted band-pass component
/// and `trend = x_adj - cycle` (with `x_adj` the optionally drift-adjusted
/// input). When `drift` is true a linear random-walk drift
/// `t * (x[n-1] - x[0]) / (n - 1)` is removed before filtering.
///
/// `low` must be at least 2.
pub fn cffilter(
    x: &Array1<f64>,
    low: f64,
    high: f64,
    drift: bool,
) -> Result<(Array1<f64>, Array1<f64>)> {
    if low < 2.0 {
        return Err(Error::Value("low must be >= 2".into()));
    }
    let nobs = x.len();
    if nobs < 2 {
        return Err(Error::Value(
            "cffilter requires at least 2 observations".into(),
        ));
    }
    let a = 2.0 * std::f64::consts::PI / high;
    let b = 2.0 * std::f64::consts::PI / low;

    // Drift adjustment: x_adj[t] = x[t] - t*(x[n-1]-x[0])/(n-1).
    let mut xa = x.clone();
    if drift {
        let slope = (x[nobs - 1] - x[0]) / (nobs as f64 - 1.0);
        for t in 0..nobs {
            xa[t] = x[t] - (t as f64) * slope;
        }
    }

    // Bj has length nobs+1: Bj[0]=B0, Bj[j]=(sin(b j)-sin(a j))/(pi j).
    let mut bj = Array1::<f64>::zeros(nobs + 1);
    bj[0] = (b - a) / std::f64::consts::PI;
    for j in 1..=nobs {
        let jf = j as f64;
        bj[j] = ((b * jf).sin() - (a * jf).sin()) / (std::f64::consts::PI * jf);
    }

    let mut y = Array1::<f64>::zeros(nobs);
    // Reproduce the reference loop exactly, including its Python negative-slice
    // semantics. For observation i:
    //   mid = Bj[1 .. nobs-1-i]   (this is Python Bj[1:-i-2] on a length nobs+1
    //                              array; the end index is nobs+1 - (i+2) = nobs-1-i)
    //   B = -0.5*Bj[0] - sum(mid)
    //   A = -Bj[0] - sum(mid) - sum(Bj[1:i]) - B
    //   y[i] = Bj[0]*x[i] + dot(mid, x[i+1:nobs-1]) + B*x[nobs-1]
    //          + dot(Bj[1:i], reverse(x[1:i])) + A*x[0]
    let xv = &xa;
    for i in 0..nobs {
        // mid slice end (exclusive) for Bj, in the [1, ...] range:
        // Python Bj[1:-i-2] over an array of length nobs+1 -> stop = nobs+1 - (i+2)
        // = nobs - 1 - i, clamped to >= 1 (empty if <= 1).
        let mid_stop_signed = nobs as isize - 1 - i as isize;
        let mid_stop = if mid_stop_signed > 1 {
            mid_stop_signed as usize
        } else {
            1
        };
        // sum over Bj[1..mid_stop]
        let mut sum_mid = 0.0;
        for idx in 1..mid_stop {
            sum_mid += bj[idx];
        }
        let bcoef = -0.5 * bj[0] - sum_mid;
        // sum over Bj[1..i]
        let mut sum_low = 0.0;
        for idx in 1..i {
            sum_low += bj[idx];
        }
        let acoef = -bj[0] - sum_mid - sum_low - bcoef;

        let mut val = bj[0] * xv[i] + bcoef * xv[nobs - 1] + acoef * xv[0];
        // dot(Bj[1..mid_stop], x[i+1 .. i+1 + (mid_stop-1) ]) aligning with
        // Python x[i+1:-1] (i.e. x[i+1 .. nobs-1]); both have mid_stop-1 elems.
        for (off, idx) in (1..mid_stop).enumerate() {
            val += bj[idx] * xv[i + 1 + off];
        }
        // dot(Bj[1..i], reverse(x[1..i])): pairs Bj[1] with x[i-1], ...,
        // Bj[i-1] with x[1].
        for idx in 1..i {
            val += bj[idx] * xv[i - idx];
        }
        y[i] = val;
    }

    let trend = xa - &y;
    Ok((y, trend))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn hp_cycle_plus_trend_reconstructs_input() {
        let x = Array1::from_vec(vec![
            1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5, 7.0,
        ]);
        let (cycle, trend) = hpfilter(&x, 1600.0).unwrap();
        for i in 0..x.len() {
            assert!((cycle[i] + trend[i] - x[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn hp_trend_is_linear_for_linear_input() {
        // A perfectly linear series has zero second differences, so the HP
        // trend reproduces it exactly and the cycle is ~0 for any lambda.
        let x: Array1<f64> = Array1::from_iter((0..20).map(|i| 2.0 * i as f64 + 1.0));
        let (cycle, trend) = hpfilter(&x, 1600.0).unwrap();
        for i in 0..x.len() {
            assert!(cycle[i].abs() < 1e-7, "cycle[{i}] = {}", cycle[i]);
            assert!((trend[i] - x[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn bk_weights_sum_to_zero() {
        let w = bk_weights(6.0, 32.0, 12);
        assert_eq!(w.len(), 25);
        assert!(w.sum().abs() < 1e-12);
        // Symmetry.
        for j in 1..=12 {
            assert!((w[12 + j] - w[12 - j]).abs() < 1e-15);
        }
    }

    #[test]
    fn bk_output_length() {
        let x: Array1<f64> = Array1::from_iter((0..40).map(|i| (i as f64 * 0.3).sin()));
        let c = bkfilter(&x, 6.0, 32.0, 12).unwrap();
        assert_eq!(c.len(), 40 - 24);
    }

    #[test]
    fn cf_cycle_plus_trend_reconstructs_adjusted_input() {
        // cycle + trend == drift-adjusted input by construction.
        let x = Array1::from_vec(vec![
            1.0, 1.4, 0.9, 2.1, 1.7, 2.6, 2.0, 3.1, 2.5, 3.6, 3.0, 4.1,
        ]);
        let (cycle, trend) = cffilter(&x, 6.0, 32.0, true).unwrap();
        let slope = (x[x.len() - 1] - x[0]) / (x.len() as f64 - 1.0);
        for i in 0..x.len() {
            let xa = x[i] - i as f64 * slope;
            assert!((cycle[i] + trend[i] - xa).abs() < 1e-12);
        }
    }

    #[test]
    fn cf_rejects_low_below_two() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(cffilter(&x, 1.5, 10.0, true).is_err());
    }
}

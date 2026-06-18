//! Theoretical second-moment properties of stationary ARMA processes.
//!
//! Mirrors the reference `arima_process` helpers
//! [`arma_acovf`](crate::arma_acovf), [`arma_acf`](crate::arma_acf) and
//! [`arma_pacf`](crate::arma_pacf).
//!
//! The coefficient conventions follow the reference: both `ar` and `ma`
//! include the zero-lag term, and the AR polynomial is written
//! `1 + ar[1] L + ar[2] L^2 + ...` (i.e. the lag polynomial that multiplies
//! the series). For a process `y_t = phi y_{t-1} + e_t`, pass
//! `ar = [1, -phi]`.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::solve;

/// MA(infinity) / impulse-response coefficients of an ARMA process.
///
/// Equivalent to the reference `arma_impulse_response(ar, ma, leads)`: it
/// applies the recursive filter `lfilter(ma, ar, impulse)` to a unit impulse,
/// producing `leads` coefficients `psi_0, psi_1, ...`.
pub fn arma_impulse_response(ar: &[f64], ma: &[f64], leads: usize) -> Array1<f64> {
    // y[n] = (sum_{k} ma[k] x[n-k] - sum_{k>=1} ar[k] y[n-k]) / ar[0],
    // with x = impulse (1 at n=0, else 0).
    let a0 = ar[0];
    let mut psi = Array1::<f64>::zeros(leads);
    for n in 0..leads {
        let mut acc = if n < ma.len() { ma[n] } else { 0.0 };
        for k in 1..ar.len() {
            if n >= k {
                acc -= ar[k] * psi[n - k];
            }
        }
        psi[n] = acc / a0;
    }
    psi
}

/// Theoretical autocovariance function of a stationary ARMA process.
///
/// Returns `nobs` autocovariances `gamma_0, gamma_1, ...`. The innovation
/// variance is `sigma2`. Errors if the AR polynomial is non-stationary.
///
/// Mirrors the reference `arma_acovf(ar, ma, nobs, sigma2)` (Brockwell &
/// Davis, eq. 3.3.8 / 3.3.9).
pub fn arma_acovf(ar: &[f64], ma: &[f64], nobs: usize, sigma2: f64) -> Result<Array1<f64>> {
    if sigma2 < 0.0 {
        return Err(Error::Value(
            "Must have positive innovation variance.".into(),
        ));
    }
    let p = ar.len() - 1;
    let q = ma.len() - 1;
    let m = p.max(q) + 1;

    // Pure white noise / MA(0) corner-case.
    if p == 0 && q == 0 {
        let mut out = Array1::<f64>::zeros(nobs);
        if nobs > 0 {
            out[0] = sigma2;
        }
        return Ok(out);
    }

    // MA representation coefficients we need (psi_0 .. psi_{m-1}).
    let ma_coeffs = arma_impulse_response(ar, ma, m);

    // Linear system A x = b for the first m autocovariances (BD eq. 3.3.8).
    let mut a = Array2::<f64>::zeros((m, m));
    let mut b = Array1::<f64>::zeros(m);
    let mut tmp_ar = vec![0.0; m];
    for (i, &v) in ar.iter().enumerate().take((p + 1).min(m)) {
        tmp_ar[i] = v;
    }
    for k in 0..m {
        // A[k, :k+1] = reversed(tmp_ar[:k+1])
        for j in 0..=k {
            a[[k, j]] = tmp_ar[k - j];
        }
        // A[k, 1:m-k] += tmp_ar[k+1:m]
        let upper = m - k;
        for (off, j) in (1..upper).enumerate() {
            // tmp_ar[(k+1) + off]
            a[[k, j]] += tmp_ar[k + 1 + off];
        }
        // b[k] = sigma2 * dot(ma[k:q+1], ma_coeffs[:q+1-k])
        let count = (q + 1).saturating_sub(k);
        let mut dotv = 0.0;
        for j in 0..count {
            dotv += ma[k + j] * ma_coeffs[j];
        }
        b[k] = sigma2 * dotv;
    }

    let solved = solve(&a, &b).map_err(|_| {
        Error::Value("The provided ar polynomial does not give a stationary process.".into())
    })?;

    let total = nobs.max(m);
    let mut acovf = Array1::<f64>::zeros(total);
    for k in 0..m {
        acovf[k] = solved[k];
    }
    // Recurse with BD eq. 3.3.9: gamma[k] = -sum_{j=1}^p ar[j] gamma[k-j].
    if total > m {
        for k in m..total {
            let mut acc = 0.0;
            for j in 1..=p {
                acc -= ar[j] * acovf[k - j];
            }
            acovf[k] = acc / ar[0];
        }
    }
    Ok(acovf.slice(ndarray::s![..nobs]).to_owned())
}

/// Theoretical autocorrelation function of an ARMA process for `lags` terms
/// (including the unit lag-0 term).
///
/// Mirrors the reference `arma_acf(ar, ma, lags)`.
pub fn arma_acf(ar: &[f64], ma: &[f64], lags: usize) -> Result<Array1<f64>> {
    let acovf = arma_acovf(ar, ma, lags, 1.0)?;
    let g0 = acovf[0];
    Ok(acovf.mapv(|v| v / g0))
}

/// Theoretical partial autocorrelation function of an ARMA process for `lags`
/// terms (including the unit lag-0 term).
///
/// Solves the Yule-Walker / Toeplitz system at each order, exactly as the
/// reference `arma_pacf(ar, ma, lags)`.
pub fn arma_pacf(ar: &[f64], ma: &[f64], lags: usize) -> Result<Array1<f64>> {
    let mut apacf = Array1::<f64>::zeros(lags);
    let acov = arma_acf(ar, ma, lags + 1)?;
    if lags == 0 {
        return Ok(apacf);
    }
    apacf[0] = 1.0;
    for k in 2..=lags {
        // r = acov[:k]; solve Toeplitz(r[:-1]) phi = r[1:]; take last entry.
        let r = acov.slice(ndarray::s![..k]);
        let order = k - 1;
        let mut t = Array2::<f64>::zeros((order, order));
        for i in 0..order {
            for j in 0..order {
                t[[i, j]] = r[(i as isize - j as isize).unsigned_abs()];
            }
        }
        let rhs = r.slice(ndarray::s![1..]).to_owned();
        let phi = solve(&t, &rhs)?;
        apacf[k - 1] = phi[order - 1];
    }
    Ok(apacf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ar1_impulse_response_is_geometric() {
        // AR(1) with phi = 0.8 => psi_k = 0.8^k.
        let psi = arma_impulse_response(&[1.0, -0.8], &[1.0], 6);
        for k in 0..6 {
            assert!((psi[k] - 0.8_f64.powi(k as i32)).abs() < 1e-12);
        }
    }

    #[test]
    fn white_noise_acf_is_delta() {
        let acf = arma_acf(&[1.0], &[1.0], 5).unwrap();
        assert!((acf[0] - 1.0).abs() < 1e-12);
        for k in 1..5 {
            assert!(acf[k].abs() < 1e-12);
        }
    }

    #[test]
    fn ar1_acf_is_geometric() {
        // AR(1) phi=0.5 => acf[k] = 0.5^k.
        let acf = arma_acf(&[1.0, -0.5], &[1.0], 6).unwrap();
        for k in 0..6 {
            assert!((acf[k] - 0.5_f64.powi(k as i32)).abs() < 1e-10);
        }
    }

    #[test]
    fn ar1_pacf_cuts_off_after_lag1() {
        let pacf = arma_pacf(&[1.0, -0.5], &[1.0], 5).unwrap();
        assert!((pacf[0] - 1.0).abs() < 1e-12);
        assert!((pacf[1] - 0.5).abs() < 1e-10);
        for k in 2..5 {
            assert!(pacf[k].abs() < 1e-10);
        }
    }
}

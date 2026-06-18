//! The innovations algorithm for one-step linear prediction.
//!
//! [`innovations_algo`] computes the coefficients of the best linear one-step
//! predictor of a stationary process directly from its autocovariance
//! sequence, together with the sequence of one-step prediction-error
//! variances. It is the classical Brockwell-Davis recursion (Proposition
//! 5.2.2), matching the reference
//! `tsa.innovations.arma_innovations.innovations_algo(acovf, nobs)`.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};

/// Result of the [`innovations_algo`] recursion.
#[derive(Debug, Clone)]
pub struct InnovationsResult {
    /// The `nobs × (nobs - 1)` coefficient matrix `theta`.
    ///
    /// Row `n` holds the predictor coefficients `theta[n, 0..n]`, left-aligned,
    /// so that `theta[[n, j]]` is the classical `theta_{n, j+1}`. The one-step
    /// predictor is
    /// `Xhat_{n+1} = sum_{j=0}^{n-1} theta[[n, j]] (X_{n-j} - Xhat_{n-j})`.
    /// Columns beyond `n` in a given row are zero.
    pub theta: Array2<f64>,
    /// The length-`nobs` sequence of one-step prediction error variances
    /// `v = (v_0, v_1, ..., v_{nobs-1})`, where `v_0 = gamma(0)`.
    pub sigma2: Array1<f64>,
}

/// Run the innovations algorithm for `nobs` steps from an autocovariance
/// sequence.
///
/// `acovf` must provide at least the first `nobs` autocovariances
/// `gamma(0), gamma(1), ..., gamma(nobs-1)`. Returns the predictor coefficient
/// matrix `theta` and the prediction-error variance sequence `sigma2` (named
/// `v` in the reference).
pub fn innovations_algo(acovf: &Array1<f64>, nobs: usize) -> Result<InnovationsResult> {
    if nobs == 0 {
        return Err(Error::Value("nobs must be positive".into()));
    }
    if acovf.len() < nobs {
        return Err(Error::Value(
            "acovf must contain at least nobs autocovariances".into(),
        ));
    }
    let cols = nobs.saturating_sub(1);
    let mut theta = Array2::<f64>::zeros((nobs, cols));
    let mut v = Array1::<f64>::zeros(nobs);
    v[0] = acovf[0];

    for n in 1..nobs {
        for k in 0..n {
            // theta_{n, n-k} = (gamma(n-k)
            //   - sum_{j=0}^{k-1} theta_{k, k-j} theta_{n, n-j} v_j) / v_k
            let mut sub = 0.0;
            for j in 0..k {
                sub += theta[[k, k - j - 1]] * theta[[n, n - j - 1]] * v[j];
            }
            if v[k] == 0.0 {
                return Err(Error::Value(
                    "innovations algorithm encountered a zero prediction variance".into(),
                ));
            }
            theta[[n, n - k - 1]] = (acovf[n - k] - sub) / v[k];
        }
        // v_n = gamma(0) - sum_{j=0}^{n-1} theta_{n, n-j}^2 v_j.
        let mut s = 0.0;
        for j in 0..n {
            let t = theta[[n, n - j - 1]];
            s += t * t * v[j];
        }
        v[n] = acovf[0] - s;
    }

    Ok(InnovationsResult { theta, sigma2: v })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn ma1_recovers_coefficient_and_variance() {
        // MA(1) with theta=0.5, sigma2=1: gamma(0)=1.25, gamma(1)=0.5.
        let acov = Array1::from_vec(vec![1.25, 0.5, 0.0, 0.0, 0.0, 0.0]);
        let r = innovations_algo(&acov, 6).unwrap();
        // theta_{n,1} = theta[[n, 0]] converges to 0.5.
        assert!((r.theta[[5, 0]] - 0.5).abs() < 1e-3);
        assert!(r.theta[[5, 0]] < 0.5);
        // v converges to the innovation variance 1.0.
        assert!((r.sigma2[5] - 1.0).abs() < 1e-3);
        assert_eq!(r.sigma2[0], 1.25);
        // Upper triangle of theta is zero.
        for n in 1..6 {
            for j in n..5 {
                assert_eq!(r.theta[[n, j]], 0.0);
            }
        }
    }

    #[test]
    fn white_noise_has_zero_coefficients() {
        let acov = Array1::from_vec(vec![2.0, 0.0, 0.0, 0.0]);
        let r = innovations_algo(&acov, 4).unwrap();
        for v in r.theta.iter() {
            assert_eq!(*v, 0.0);
        }
        for v in r.sigma2.iter() {
            assert_eq!(*v, 2.0);
        }
    }

    #[test]
    fn rejects_short_acovf() {
        let acov = Array1::from_vec(vec![1.0, 0.5]);
        assert!(innovations_algo(&acov, 5).is_err());
    }
}

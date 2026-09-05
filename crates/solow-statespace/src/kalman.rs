//! Linear-Gaussian Kalman filter and smoother.
//!
//! The time-invariant state-space model handled here is
//!
//! ```text
//!   alpha_t = T alpha_{t-1} + R eta_t,     eta_t ~ N(0, Q)
//!   y_t     = Z alpha_t + eps_t,           eps_t ~ N(0, H)
//! ```
//!
//! with `alpha_t` the (`m`-dimensional) latent state and `y_t` a scalar
//! observation (univariate measurement). This is exactly the representation
//! used by the SARIMAX state space: `T` is the companion transition matrix,
//! `R` the selection vector, `Q = sigma^2` the state disturbance variance,
//! `Z` the design row and `H = 0` the (absent) measurement noise.
//!
//! The filter returns the exact Gaussian log-likelihood together with the
//! predicted and filtered state means, and a fixed-interval smoother recovers
//! the smoothed states.

use ndarray::{Array1, Array2};
use solow_core::error::Result;
use solow_linalg::solve;

/// A time-invariant linear-Gaussian state-space model with scalar observations.
#[derive(Clone, Debug)]
pub struct StateSpace {
    /// Transition matrix `T` (`m x m`).
    pub transition: Array2<f64>,
    /// Selection matrix `R` (`m x r`).
    pub selection: Array2<f64>,
    /// State disturbance covariance `Q` (`r x r`).
    pub state_cov: Array2<f64>,
    /// Design row `Z` (`1 x m`).
    pub design: Array1<f64>,
    /// Scalar measurement-noise variance `H`.
    pub obs_cov: f64,
    /// Initial state mean `a_1|0` (length `m`).
    pub init_state: Array1<f64>,
    /// Initial state covariance `P_1|0` (`m x m`).
    pub init_cov: Array2<f64>,
}

/// Result of running the Kalman filter forward over a series.
#[derive(Clone, Debug)]
pub struct FilterOutput {
    /// Total Gaussian log-likelihood `sum_t log p(y_t | y_{1:t-1})`.
    pub loglike: f64,
    /// Per-observation log-likelihood contributions.
    pub loglike_obs: Array1<f64>,
    /// Predicted states `a_{t|t-1}` (rows are time, `n x m`).
    pub predicted_state: Array2<f64>,
    /// Predicted state covariances stacked as `n` blocks of `m x m`.
    pub predicted_cov: Vec<Array2<f64>>,
    /// Filtered states `a_{t|t}` (`n x m`).
    pub filtered_state: Array2<f64>,
    /// Filtered state covariances (`n` blocks of `m x m`).
    pub filtered_cov: Vec<Array2<f64>>,
    /// One-step-ahead prediction errors `v_t = y_t - Z a_{t|t-1}`.
    pub forecast_error: Array1<f64>,
    /// Prediction-error variances `F_t = Z P_{t|t-1} Z' + H`.
    pub forecast_error_cov: Array1<f64>,
}

const LOG_2PI: f64 = 1.837_877_066_409_345_6;

impl StateSpace {
    /// Run the Kalman filter forward over `y`, returning the exact
    /// log-likelihood and the filtered/predicted state sequences.
    ///
    /// `loglike_burn` observations at the start are excluded from the returned
    /// total `loglike` (the per-observation values are still recorded); this
    /// matches the SARIMAX treatment of differenced models.
    pub fn filter(&self, y: &Array1<f64>, loglike_burn: usize) -> FilterOutput {
        let n = y.len();
        let m = self.transition.nrows();

        let mut predicted_state = Array2::<f64>::zeros((n, m));
        let mut filtered_state = Array2::<f64>::zeros((n, m));
        let mut predicted_cov = Vec::with_capacity(n);
        let mut filtered_cov = Vec::with_capacity(n);
        let mut forecast_error = Array1::<f64>::zeros(n);
        let mut forecast_error_cov = Array1::<f64>::zeros(n);
        let mut loglike_obs = Array1::<f64>::zeros(n);

        // R Q R' — the additive state disturbance covariance.
        let rqr = self.selection.dot(&self.state_cov).dot(&self.selection.t());

        let z = &self.design;
        let mut a = self.init_state.clone();
        let mut p = self.init_cov.clone();
        let mut loglike = 0.0;

        for t in 0..n {
            // Store the prediction a_{t|t-1}, P_{t|t-1}.
            predicted_state.row_mut(t).assign(&a);
            predicted_cov.push(p.clone());

            // Forecast error and its variance (scalar observation).
            let za = z.dot(&a);
            let v = y[t] - za;
            let pz = p.dot(z); // P Z'  (length m)
            let f = z.dot(&pz) + self.obs_cov;
            forecast_error[t] = v;
            forecast_error_cov[t] = f;

            // Log-likelihood contribution.
            let ll = -0.5 * (LOG_2PI + f.ln() + v * v / f);
            loglike_obs[t] = ll;
            if t >= loglike_burn {
                loglike += ll;
            }

            // Filtering update: a_{t|t} = a + K v, P_{t|t} = P - K (PZ')'.
            // K = P Z' / F  (length m).
            let k = &pz / f;
            let a_filt = &a + &(&k * v);
            // P_filt = P - k (pz)^T
            let mut p_filt = p.clone();
            for i in 0..m {
                for j in 0..m {
                    p_filt[[i, j]] -= k[i] * pz[j];
                }
            }
            filtered_state.row_mut(t).assign(&a_filt);
            filtered_cov.push(p_filt.clone());

            // Prediction to t+1: a_{t+1|t} = T a_{t|t}; P_{t+1|t} = T P_{t|t} T' + RQR'.
            a = self.transition.dot(&a_filt);
            p = self.transition.dot(&p_filt).dot(&self.transition.t()) + &rqr;
            // Enforce numerical symmetry.
            symmetrize(&mut p);
        }

        FilterOutput {
            loglike,
            loglike_obs,
            predicted_state,
            predicted_cov,
            filtered_state,
            filtered_cov,
            forecast_error,
            forecast_error_cov,
        }
    }

    /// Fixed-interval (Rauch-Tung-Striebel) smoother.
    ///
    /// Runs [`Self::filter`] internally and returns the smoothed state means
    /// `a_{t|n}` as rows of an `n x m` matrix.
    pub fn smooth(&self, y: &Array1<f64>, loglike_burn: usize) -> Result<Array2<f64>> {
        let out = self.filter(y, loglike_burn);
        let n = y.len();
        let m = self.transition.nrows();
        let mut smoothed = Array2::<f64>::zeros((n, m));
        if n == 0 {
            return Ok(smoothed);
        }
        // Last smoothed state equals the last filtered state.
        smoothed
            .row_mut(n - 1)
            .assign(&out.filtered_state.row(n - 1));
        for t in (0..n - 1).rev() {
            let pf = &out.filtered_cov[t]; // P_{t|t}
            let pp = &out.predicted_cov[t + 1]; // P_{t+1|t}
                                                // Smoother gain J = P_{t|t} T' P_{t+1|t}^{-1}.
            let ptt_tt = pf.dot(&self.transition.t()); // m x m
                                                       // Solve J P_{t+1|t} = ptt_tt  =>  J = ptt_tt P_{t+1|t}^{-1}; do column-wise.
            let j = solve_right(&ptt_tt, pp)?;
            let a_filt = out.filtered_state.row(t).to_owned();
            let a_pred_next = out.predicted_state.row(t + 1).to_owned();
            let a_smooth_next = smoothed.row(t + 1).to_owned();
            let diff = &a_smooth_next - &a_pred_next;
            let a_smooth = &a_filt + &j.dot(&diff);
            smoothed.row_mut(t).assign(&a_smooth);
        }
        Ok(smoothed)
    }
}

/// Solve `X B = A` for `X` (i.e. `X = A B^{-1}`) with `B` symmetric PD-ish.
fn solve_right(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    // X = A B^{-1}  <=>  B' X' = A'  => solve B^T x_col = a_col for each row of A.
    let bt = b.t().to_owned();
    let m = a.nrows();
    let k = a.ncols();
    let mut x = Array2::<f64>::zeros((m, k));
    for i in 0..m {
        let rhs = a.row(i).to_owned();
        let sol = solve(&bt, &rhs)?;
        x.row_mut(i).assign(&sol);
    }
    Ok(x)
}

fn symmetrize(p: &mut Array2<f64>) {
    let m = p.nrows();
    for i in 0..m {
        for j in (i + 1)..m {
            let v = 0.5 * (p[[i, j]] + p[[j, i]]);
            p[[i, j]] = v;
            p[[j, i]] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Scalar AR(1) Kalman filter reduces to the textbook recursion.
    #[test]
    fn ar1_loglike_matches_manual_recursion() {
        let phi = 0.5_f64;
        let s2 = 1.3_f64;
        let y = array![0.5, -0.2, 1.1, 0.3, -0.9, 0.7];
        let ss = StateSpace {
            transition: array![[phi]],
            selection: array![[1.0]],
            state_cov: array![[s2]],
            design: array![1.0],
            obs_cov: 0.0,
            init_state: array![0.0],
            init_cov: array![[s2 / (1.0 - phi * phi)]],
        };
        let out = ss.filter(&y, 0);

        // Manual scalar recursion.
        let mut a = 0.0;
        let mut p = s2 / (1.0 - phi * phi);
        let mut ll = 0.0;
        for &yt in y.iter() {
            let v = yt - a;
            let f = p;
            ll += -0.5 * (LOG_2PI + f.ln() + v * v / f);
            let k = p / f;
            a = phi * (a + k * v);
            p = phi * phi * (p - k * p) + s2;
        }
        assert!(
            (out.loglike - ll).abs() < 1e-12,
            "{} vs {}",
            out.loglike,
            ll
        );
    }

    /// Smoother end conditions: last smoothed == last filtered.
    #[test]
    fn smoother_endpoint() {
        let phi = 0.4_f64;
        let s2 = 1.0_f64;
        let y = array![0.5, -0.2, 1.1, 0.3];
        let ss = StateSpace {
            transition: array![[phi]],
            selection: array![[1.0]],
            state_cov: array![[s2]],
            design: array![1.0],
            obs_cov: 0.0,
            init_state: array![0.0],
            init_cov: array![[s2 / (1.0 - phi * phi)]],
        };
        let sm = ss.smooth(&y, 0).unwrap();
        let out = ss.filter(&y, 0);
        let n = y.len();
        assert!((sm[[n - 1, 0]] - out.filtered_state[[n - 1, 0]]).abs() < 1e-12);
    }
}

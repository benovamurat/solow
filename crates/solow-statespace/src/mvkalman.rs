//! Multivariate linear-Gaussian Kalman filter.
//!
//! Where [`crate::kalman`] handles a *scalar* observation, this module filters a
//! state space whose observation vector `y_t` is `p`-dimensional:
//!
//! ```text
//!   alpha_t = T alpha_{t-1} + R eta_t,     eta_t ~ N(0, Q)
//!   y_t     = Z alpha_t + eps_t,           eps_t ~ N(0, H)
//! ```
//!
//! It is used by the [`crate::dynamic_factor`] estimator, where several observed
//! series load on a small number of latent factors. The recursion is the
//! standard one with a matrix forecast-error variance `F_t = Z P Z' + H`, the
//! Gaussian log-likelihood
//!
//! ```text
//!   log p(y_t | y_{1:t-1}) = -0.5 ( p log 2π + log|F_t| + v_t' F_t^{-1} v_t ).
//! ```

use ndarray::{Array1, Array2};
use solow_core::error::Result;
use solow_linalg::{det, inv};

/// A time-invariant linear-Gaussian state space with vector observations.
#[derive(Clone, Debug)]
pub struct MvStateSpace {
    /// Transition matrix `T` (`m x m`).
    pub transition: Array2<f64>,
    /// Selection matrix `R` (`m x r`).
    pub selection: Array2<f64>,
    /// State disturbance covariance `Q` (`r x r`).
    pub state_cov: Array2<f64>,
    /// Design matrix `Z` (`p x m`).
    pub design: Array2<f64>,
    /// Measurement-noise covariance `H` (`p x p`).
    pub obs_cov: Array2<f64>,
    /// Initial state mean `a_1|0` (length `m`).
    pub init_state: Array1<f64>,
    /// Initial state covariance `P_1|0` (`m x m`).
    pub init_cov: Array2<f64>,
}

const LOG_2PI: f64 = 1.837_877_066_409_345_6;

impl MvStateSpace {
    /// Run the Kalman filter forward over the `n x p` matrix of observations
    /// `y` (one observation vector per row), returning the total Gaussian
    /// log-likelihood `sum_{t >= loglike_burn} log p(y_t | y_{1:t-1})`.
    pub fn loglike(&self, y: &Array2<f64>, loglike_burn: usize) -> Result<f64> {
        let n = y.nrows();
        let p = self.design.nrows();

        let rqr = self.selection.dot(&self.state_cov).dot(&self.selection.t());
        let z = &self.design;
        let zt = z.t().to_owned();
        let mut a = self.init_state.clone();
        let mut pcov = self.init_cov.clone();
        let mut loglike = 0.0;

        for t in 0..n {
            let yt = y.row(t).to_owned();
            // Forecast error v = y - Z a.
            let v = &yt - &z.dot(&a);
            // F = Z P Z' + H.
            let pz = pcov.dot(&zt); // m x p
            let f = z.dot(&pz) + &self.obs_cov; // p x p
            let finv = inv(&f)?;
            let logdet = det(&f)?.ln();
            let quad = v.dot(&finv.dot(&v));
            let ll = -0.5 * (p as f64 * LOG_2PI + logdet + quad);
            if t >= loglike_burn {
                loglike += ll;
            }

            // Kalman gain K = P Z' F^{-1} (m x p); filtered update.
            let k = pz.dot(&finv); // m x p
            let a_filt = &a + &k.dot(&v);
            // P_filt = P - K Z P = P - K (pz)'.
            let p_filt = &pcov - &k.dot(&pz.t());

            // Predict to t+1.
            a = self.transition.dot(&a_filt);
            pcov = self.transition.dot(&p_filt).dot(&self.transition.t()) + &rqr;
            symmetrize(&mut pcov);
        }
        Ok(loglike)
    }

    /// One-step-ahead forecasts `Z a_{t|t-1}` for every observation (the
    /// in-sample fitted values), returned as an `n x p` matrix.
    pub fn forecasts(&self, y: &Array2<f64>) -> Result<Array2<f64>> {
        let n = y.nrows();
        let p = self.design.nrows();
        let rqr = self.selection.dot(&self.state_cov).dot(&self.selection.t());
        let z = &self.design;
        let zt = z.t().to_owned();
        let mut a = self.init_state.clone();
        let mut pcov = self.init_cov.clone();
        let mut out = Array2::<f64>::zeros((n, p));

        for t in 0..n {
            let yt = y.row(t).to_owned();
            out.row_mut(t).assign(&z.dot(&a));
            let v = &yt - &z.dot(&a);
            let pz = pcov.dot(&zt);
            let f = z.dot(&pz) + &self.obs_cov;
            let finv = inv(&f)?;
            let k = pz.dot(&finv);
            let a_filt = &a + &k.dot(&v);
            let p_filt = &pcov - &k.dot(&pz.t());
            a = self.transition.dot(&a_filt);
            pcov = self.transition.dot(&p_filt).dot(&self.transition.t()) + &rqr;
            symmetrize(&mut pcov);
        }
        Ok(out)
    }
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

    /// A 1-dimensional observation reduces to the scalar AR(1) recursion, so
    /// the multivariate filter must agree with the textbook scalar formula.
    #[test]
    fn univariate_reduces_to_scalar() {
        let phi = 0.5_f64;
        let s2 = 1.3_f64;
        let y = array![[0.5], [-0.2], [1.1], [0.3], [-0.9], [0.7]];
        let ss = MvStateSpace {
            transition: array![[phi]],
            selection: array![[1.0]],
            state_cov: array![[s2]],
            design: array![[1.0]],
            obs_cov: array![[0.0]],
            init_state: array![0.0],
            init_cov: array![[s2 / (1.0 - phi * phi)]],
        };
        let got = ss.loglike(&y, 0).unwrap();

        let mut a = 0.0;
        let mut p = s2 / (1.0 - phi * phi);
        let mut ll = 0.0;
        for r in y.rows() {
            let yt = r[0];
            let v = yt - a;
            let f = p;
            ll += -0.5 * (LOG_2PI + f.ln() + v * v / f);
            let k = p / f;
            a = phi * (a + k * v);
            p = phi * phi * (p - k * p) + s2;
        }
        assert!((got - ll).abs() < 1e-12, "{got} vs {ll}");
    }
}

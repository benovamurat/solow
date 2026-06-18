//! Rolling fixed-window ordinary least squares.
//!
//! For a window of `w` consecutive observations, an OLS fit is computed for every
//! window ending at index `t = w−1, w, …, n−1`. The result exposes the
//! coefficient *path* — one row of parameters per window end — matching the
//! non-missing rows of the canonical reference's `RollingOLS.params` matrix.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::pinv;

/// A rolling-window OLS model awaiting `fit`.
#[derive(Clone, Debug)]
pub struct RollingOLS {
    endog: Array1<f64>,
    exog: Array2<f64>,
    window: usize,
}

impl RollingOLS {
    /// Build a rolling-window OLS model over a fixed `window` of observations.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>, window: usize) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        let n = exog.nrows();
        if window == 0 || window > n {
            return Err(Error::Value("window must be in 1..=nobs".into()));
        }
        if window < exog.ncols() {
            return Err(Error::Value(
                "window must be >= number of regressors".into(),
            ));
        }
        Ok(RollingOLS {
            endog,
            exog,
            window,
        })
    }

    /// Fit OLS over each fixed window.
    pub fn fit(&self) -> Result<RollingOLSResults> {
        let (n, p) = self.exog.dim();
        let w = self.window;
        let n_windows = n - w + 1;

        let mut params = Array2::<f64>::zeros((n_windows, p));
        let mut window_ends = Vec::with_capacity(n_windows);

        for (row, start) in (0..n_windows).map(|s| (s, s)) {
            let end = start + w; // exclusive
            let xw = self.exog.slice(ndarray::s![start..end, ..]).to_owned();
            let yw = self.endog.slice(ndarray::s![start..end]).to_owned();
            // OLS via the pseudoinverse of the window design.
            let (xw_pinv, _) = pinv(&xw)?;
            let beta = xw_pinv.dot(&yw);
            for j in 0..p {
                params[[row, j]] = beta[j];
            }
            window_ends.push(end - 1);
        }

        Ok(RollingOLSResults {
            params,
            window_ends,
            window: w,
            nobs: n,
        })
    }
}

/// The fitted result of a [`RollingOLS`].
#[derive(Clone, Debug)]
pub struct RollingOLSResults {
    /// One row of coefficients per window, ordered by window end.
    pub params: Array2<f64>,
    /// The (0-based) ending observation index of each window.
    pub window_ends: Vec<usize>,
    /// The window length.
    pub window: usize,
    /// Total number of observations.
    pub nobs: usize,
}

impl RollingOLSResults {
    /// Number of fitted windows.
    pub fn n_windows(&self) -> usize {
        self.params.nrows()
    }

    /// Coefficients for the window ending at observation `end` (0-based), if any.
    pub fn params_at(&self, end: usize) -> Option<Array1<f64>> {
        self.window_ends
            .iter()
            .position(|&e| e == end)
            .map(|row| self.params.row(row).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn full_window_equals_ols() {
        // window == nobs => a single window equal to the full OLS fit.
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let y = array![1.0, 3.0, 5.0, 7.0]; // exact y = 1 + 2x
        let res = RollingOLS::new(y, x, 4).unwrap().fit().unwrap();
        assert_eq!(res.n_windows(), 1);
        assert_eq!(res.window_ends, vec![3]);
        assert!((res.params[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((res.params[[0, 1]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn rolling_recovers_local_slopes() {
        // Exactly linear data => every window recovers slope 2, intercept 1.
        let n = 8;
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = i as f64;
            y[i] = 1.0 + 2.0 * i as f64;
        }
        let res = RollingOLS::new(y, x, 3).unwrap().fit().unwrap();
        assert_eq!(res.n_windows(), n - 2);
        for row in 0..res.n_windows() {
            assert!((res.params[[row, 0]] - 1.0).abs() < 1e-9);
            assert!((res.params[[row, 1]] - 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn rejects_bad_window() {
        let x = array![[1.0, 0.0], [1.0, 1.0]];
        let y = array![0.0, 1.0];
        assert!(RollingOLS::new(y.clone(), x.clone(), 0).is_err());
        assert!(RollingOLS::new(y.clone(), x.clone(), 3).is_err());
        // window < regressors
        assert!(RollingOLS::new(y, x, 1).is_err());
    }
}

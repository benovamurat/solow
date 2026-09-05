//! Deterministic trend/seasonal term construction.
//!
//! [`DeterministicProcess`] builds the in-sample design matrix of deterministic
//! regressors (a constant, polynomial time trends, and seasonal dummies) used
//! by time-series models, mirroring the reference `tsa.deterministic`
//! `DeterministicProcess(...).in_sample()` term matrix.

use ndarray::Array2;
use solow_core::error::{Error, Result};

/// Builder for a matrix of deterministic time-series terms.
///
/// Configure with a constant, a polynomial time-trend `order`, and optional
/// seasonal dummies of a given `period`, then call [`Self::in_sample`] to
/// materialise the `steps × k` term matrix.
///
/// Column order is `[const?, trend, trend_squared, ..., seasonal dummies?]`.
/// When the constant is present the first seasonal period is used as the
/// reference category and dropped, yielding `period - 1` dummy columns named
/// `s(2,period)..s(period,period)`; otherwise all `period` dummies are kept.
/// The time index for the trend terms runs `1, 2, ..., steps`.
#[derive(Debug, Clone)]
pub struct DeterministicProcess {
    steps: usize,
    constant: bool,
    order: usize,
    seasonal: bool,
    period: usize,
}

impl DeterministicProcess {
    /// Create a process spanning `steps` in-sample observations.
    ///
    /// `constant` toggles the intercept column, `order` is the degree of the
    /// polynomial time trend (0 disables it), and `seasonal` toggles seasonal
    /// dummies with the given `period` (which must be at least 2 when seasonal
    /// terms are requested).
    pub fn new(
        steps: usize,
        constant: bool,
        order: usize,
        seasonal: bool,
        period: usize,
    ) -> Result<Self> {
        if seasonal && period < 2 {
            return Err(Error::Value(
                "period must be >= 2 when seasonal terms are requested".into(),
            ));
        }
        Ok(Self {
            steps,
            constant,
            order,
            seasonal,
            period,
        })
    }

    /// Number of deterministic seasonal dummy columns.
    fn n_seasonal(&self) -> usize {
        if !self.seasonal {
            0
        } else if self.constant {
            self.period - 1
        } else {
            self.period
        }
    }

    /// Total number of deterministic columns produced by [`Self::in_sample`].
    pub fn ncols(&self) -> usize {
        (self.constant as usize) + self.order + self.n_seasonal()
    }

    /// Materialise the in-sample deterministic term matrix.
    ///
    /// The result has shape `(steps, ncols)` with columns ordered
    /// `[const?, trend^1..trend^order, seasonal dummies?]`.
    pub fn in_sample(&self) -> Array2<f64> {
        let n = self.steps;
        let k = self.ncols();
        let mut out = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let mut col = 0;
            if self.constant {
                out[[i, col]] = 1.0;
                col += 1;
            }
            // Polynomial trend with time index starting at 1.
            let time = (i + 1) as f64;
            for power in 1..=self.order {
                out[[i, col]] = time.powi(power as i32);
                col += 1;
            }
            // Seasonal dummies.
            if self.seasonal {
                let phase = i % self.period; // 0-based period within cycle
                if self.constant {
                    // Reference category is phase 0 (period 1); dummies cover
                    // phases 1..period-1 mapping to columns s(2,p)..s(p,p).
                    if phase >= 1 {
                        out[[i, col + (phase - 1)]] = 1.0;
                    }
                } else {
                    out[[i, col + phase]] = 1.0;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_and_quadratic_trend() {
        let dp = DeterministicProcess::new(3, true, 2, false, 0).unwrap();
        let m = dp.in_sample();
        assert_eq!(m.dim(), (3, 3));
        // [const, trend, trend_squared].
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[0, 1]], 1.0);
        assert_eq!(m[[0, 2]], 1.0);
        assert_eq!(m[[2, 1]], 3.0);
        assert_eq!(m[[2, 2]], 9.0);
    }

    #[test]
    fn seasonal_with_constant_drops_first_period() {
        let dp = DeterministicProcess::new(8, true, 0, true, 4).unwrap();
        let m = dp.in_sample();
        // const + (period-1) dummies.
        assert_eq!(m.dim(), (8, 4));
        // Row 0 (phase 0) -> all dummies zero, const 1.
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[0, 1]], 0.0);
        assert_eq!(m[[0, 2]], 0.0);
        assert_eq!(m[[0, 3]], 0.0);
        // Row 1 (phase 1) -> s(2,4) = 1.
        assert_eq!(m[[1, 1]], 1.0);
        // Row 3 (phase 3) -> s(4,4) = 1.
        assert_eq!(m[[3, 3]], 1.0);
    }

    #[test]
    fn seasonal_without_constant_keeps_all_dummies() {
        let dp = DeterministicProcess::new(4, false, 0, true, 4).unwrap();
        let m = dp.in_sample();
        assert_eq!(m.dim(), (4, 4));
        // Identity-like block: row i has a 1 in column i.
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(m[[i, j]], if i == j { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn rejects_tiny_period() {
        assert!(DeterministicProcess::new(4, true, 0, true, 1).is_err());
    }
}

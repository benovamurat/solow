//! Time-series helper transforms: [`lagmat`] and [`add_trend`].
//!
//! These mirror the reference `tsatools` utilities and are the building blocks
//! used by the augmented Dickey-Fuller regression and the partial
//! autocorrelation OLS estimator.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};

/// How to trim the partially observed rows produced by lag construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trim {
    /// Keep all rows including the leading zero-padded ones (`startobs = 0`,
    /// `stopobs = nobs + maxlag`).
    None,
    /// Drop the leading `maxlag` rows that contain padding from the front
    /// (`startobs = 0`, `stopobs = nobs`).
    Forward,
    /// Drop the trailing padded rows (`startobs = maxlag`,
    /// `stopobs = nobs + maxlag`).
    Backward,
    /// Drop both leading and trailing padded rows (`startobs = maxlag`,
    /// `stopobs = nobs`).
    Both,
}

/// Which original (lag-0) columns to return alongside the lagged columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Original {
    /// Exclude the original series; return only the lagged columns.
    Ex,
    /// Return the lagged columns and the original columns separately.
    Sep,
    /// Include the original columns interleaved with the lags.
    In,
}

/// Construct a matrix of lags of a (possibly multivariate) series.
///
/// Equivalent to the reference `lagmat(x, maxlag, trim, original)`. The input
/// `x` has shape `(nobs, nvar)`; the result groups columns by lag with the
/// lag-0 block first, then lag-1, etc.
///
/// Returns `(lags, leads)` where `leads` is empty unless `original` is
/// [`Original::Sep`].
pub fn lagmat(
    x: &Array2<f64>,
    maxlag: usize,
    trim: Trim,
    original: Original,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let (nobs, nvar) = x.dim();
    if maxlag >= nobs {
        return Err(Error::Value("maxlag should be < nobs".into()));
    }
    let ncols = nvar * (maxlag + 1);
    let mut lm = Array2::<f64>::zeros((nobs + maxlag, ncols));
    // Place the original block into the (maxlag - k)-th column group, shifted
    // down by (maxlag - k) rows. This reproduces the reference layout exactly.
    for k in 0..=maxlag {
        let row0 = maxlag - k;
        let col0 = nvar * (maxlag - k);
        for i in 0..nobs {
            for j in 0..nvar {
                lm[[row0 + i, col0 + j]] = x[[i, j]];
            }
        }
    }

    let (startobs, stopobs) = match trim {
        Trim::None => (0usize, nobs + maxlag),
        Trim::Forward => (0usize, nobs),
        Trim::Backward => (maxlag, nobs + maxlag),
        Trim::Both => (maxlag, nobs),
    };

    let dropidx = match original {
        Original::Ex | Original::Sep => nvar,
        Original::In => 0,
    };

    let nrows = stopobs - startobs;
    let lag_ncols = ncols - dropidx;
    let mut lags = Array2::<f64>::zeros((nrows, lag_ncols));
    for i in 0..nrows {
        for j in 0..lag_ncols {
            lags[[i, j]] = lm[[startobs + i, dropidx + j]];
        }
    }

    let leads = if original == Original::Sep {
        let mut leads = Array2::<f64>::zeros((nrows, nvar));
        for i in 0..nrows {
            for j in 0..nvar {
                leads[[i, j]] = lm[[startobs + i, j]];
            }
        }
        leads
    } else {
        Array2::<f64>::zeros((nrows, 0))
    };

    Ok((lags, leads))
}

/// Convenience wrapper over [`lagmat`] for a 1-D series.
pub fn lagmat1d(
    x: &Array1<f64>,
    maxlag: usize,
    trim: Trim,
    original: Original,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let col = x.view().insert_axis(ndarray::Axis(1)).to_owned();
    lagmat(&col, maxlag, trim, original)
}

/// Deterministic trend specification for [`add_trend`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    /// No trend (returns the input unchanged).
    N,
    /// Constant only.
    C,
    /// Linear time trend only.
    T,
    /// Constant plus linear trend.
    Ct,
    /// Constant, linear and quadratic trend.
    Ctt,
}

impl Trend {
    /// Parse a trend code (`"n"`, `"c"`, `"t"`, `"ct"`, `"ctt"`).
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "n" => Ok(Trend::N),
            "c" => Ok(Trend::C),
            "t" => Ok(Trend::T),
            "ct" => Ok(Trend::Ct),
            "ctt" => Ok(Trend::Ctt),
            other => Err(Error::Value(format!("unknown trend '{other}'"))),
        }
    }
}

/// Build the deterministic trend columns for `nobs` observations.
///
/// The time index runs `1, 2, ..., nobs`. Column order is always
/// `[const, trend, trend_squared]` restricted to the requested components.
fn trend_columns(trend: Trend, nobs: usize) -> Array2<f64> {
    let specs: &[u8] = match trend {
        Trend::N => &[],
        Trend::C => &[0],
        Trend::T => &[1],
        Trend::Ct => &[0, 1],
        Trend::Ctt => &[0, 1, 2],
    };
    let mut out = Array2::<f64>::zeros((nobs, specs.len()));
    for i in 0..nobs {
        let t = (i + 1) as f64;
        for (c, &power) in specs.iter().enumerate() {
            out[[i, c]] = t.powi(power as i32);
        }
    }
    out
}

/// Add deterministic trend columns to a design matrix.
///
/// Mirrors the reference `add_trend(x, trend, prepend)`. When `prepend` is
/// true the trend columns are placed before `x`, otherwise after. This always
/// adds the requested columns (equivalent to the reference
/// `has_constant="add"`).
pub fn add_trend(x: &Array2<f64>, trend: Trend, prepend: bool) -> Array2<f64> {
    if trend == Trend::N {
        return x.clone();
    }
    let (nobs, nvar) = x.dim();
    let tr = trend_columns(trend, nobs);
    let ntr = tr.ncols();
    let mut out = Array2::<f64>::zeros((nobs, nvar + ntr));
    if prepend {
        for i in 0..nobs {
            for j in 0..ntr {
                out[[i, j]] = tr[[i, j]];
            }
            for j in 0..nvar {
                out[[i, ntr + j]] = x[[i, j]];
            }
        }
    } else {
        for i in 0..nobs {
            for j in 0..nvar {
                out[[i, j]] = x[[i, j]];
            }
            for j in 0..ntr {
                out[[i, nvar + j]] = tr[[i, j]];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn lagmat_forward_shifts_columns() {
        // x = [1,2,3,4,5], maxlag = 2, original "ex", trim "forward".
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let (lags, leads) = lagmat(&x, 2, Trim::Forward, Original::Ex).unwrap();
        assert_eq!(lags.dim(), (5, 2));
        assert_eq!(leads.ncols(), 0);
        // First row has no history -> zeros; row 2 (index 2) -> [2, 1].
        assert_eq!(lags[[0, 0]], 0.0);
        assert_eq!(lags[[0, 1]], 0.0);
        assert_eq!(lags[[2, 0]], 2.0);
        assert_eq!(lags[[2, 1]], 1.0);
    }

    #[test]
    fn lagmat_both_drops_padding() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let (lags, _) = lagmat(&x, 2, Trim::Both, Original::Ex).unwrap();
        assert_eq!(lags.dim(), (3, 2));
        // Row 0 corresponds to t=2 (0-based) -> lag1=2, lag2=1.
        assert_eq!(lags[[0, 0]], 2.0);
        assert_eq!(lags[[0, 1]], 1.0);
    }

    #[test]
    fn add_trend_prepend_orders_columns() {
        let x = array![[7.0], [8.0], [9.0]];
        let out = add_trend(&x, Trend::Ct, true);
        assert_eq!(out.dim(), (3, 3));
        // [const, trend, original].
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 1.0);
        assert_eq!(out[[2, 1]], 3.0);
        assert_eq!(out[[2, 2]], 9.0);
    }

    #[test]
    fn add_trend_none_is_identity() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let out = add_trend(&x, Trend::N, false);
        assert_eq!(out, x);
    }

    #[test]
    fn lagmat_rejects_too_large_maxlag() {
        let x = array![[1.0], [2.0]];
        assert!(lagmat(&x, 2, Trim::Both, Original::Ex).is_err());
    }
}

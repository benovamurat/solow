//! LOWESS — locally-weighted scatterplot smoothing (Cleveland, 1979).
//!
//! [`lowess`] computes a robust locally-weighted linear regression. For each
//! anchor point it fits a tricube-weighted line to the `frac · n` nearest
//! neighbours (by `x`), then optionally refines the fit with `it` rounds of
//! bisquare residual reweighting to down-weight outliers.

use ndarray::Array1;
use solow_core::error::{Error, Result};

/// Options controlling [`lowess`].
#[derive(Debug, Clone, Copy)]
pub struct LowessOptions {
    /// Fraction of the data used to estimate each fitted value, in `(0, 1]`.
    /// The neighbourhood size is `r = floor(frac · n)`, clamped to `[2, n]`.
    pub frac: f64,
    /// Number of robustifying (residual-reweighting) iterations.
    pub it: usize,
    /// Distance within which linear interpolation replaces weighted regression.
    /// `0.0` (the default) disables interpolation and fits at every point. A
    /// good speed-up choice for large data is `0.01 · (max(x) − min(x))`.
    pub delta: f64,
}

impl Default for LowessOptions {
    /// The reference defaults: `frac = 2/3`, `it = 3`, `delta = 0`.
    fn default() -> Self {
        LowessOptions {
            frac: 2.0 / 3.0,
            it: 3,
            delta: 0.0,
        }
    }
}

/// Result of a [`lowess`] fit: the input abscissae sorted ascending, together
/// with the smoothed ordinate at each sorted abscissa.
#[derive(Debug, Clone)]
pub struct LowessFit {
    /// Sorted `x` values.
    pub x: Array1<f64>,
    /// Smoothed `y` values aligned with [`LowessFit::x`].
    pub fitted: Array1<f64>,
}

/// Tricube weight `(1 − |t|³)³` for `|t| < 1`, else `0`.
#[inline]
fn tricube(t: f64) -> f64 {
    let a = t.abs();
    if a < 1.0 {
        let c = 1.0 - a * a * a;
        c * c * c
    } else {
        0.0
    }
}

/// Bisquare weight `(1 − t²)²` for `|t| < 1`, else `0`.
#[inline]
fn bisquare(t: f64) -> f64 {
    let a = t.abs();
    if a < 1.0 {
        let c = 1.0 - a * a;
        c * c
    } else {
        0.0
    }
}

/// Median of a slice (copies and sorts internally). Returns `0.0` for empty.
fn median(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Weighted local linear regression at `xs[i]`.
///
/// The window is `[left, left + r)`, slid rightward so that it holds the `r`
/// nearest neighbours of `xs[i]`. Returns the fitted value and the (possibly
/// advanced) `left` index for reuse at the next anchor.
fn local_fit(
    xs: &[f64],
    ys: &[f64],
    rweights: &[f64],
    r: usize,
    i: usize,
    mut left: usize,
) -> (f64, usize) {
    let n = xs.len();
    let xi = xs[i];

    // Slide the window right while the next point on the right is closer than
    // the current left edge (monotone in `i` because `xs` is sorted).
    while left + r < n {
        if (xi - xs[left]) > (xs[left + r] - xi) {
            left += 1;
        } else {
            break;
        }
    }
    let lo = left;
    let hi = left + r;

    // Bandwidth = distance to the farthest point in the window.
    let h = (xi - xs[lo]).max(xs[hi - 1] - xi);

    // Accumulate weighted sums for the local line.
    let mut sw = 0.0;
    let mut swx = 0.0;
    let mut swy = 0.0;
    for j in lo..hi {
        let w = if h > 0.0 {
            tricube((xs[j] - xi) / h)
        } else {
            1.0
        } * rweights[j];
        sw += w;
        swx += w * xs[j];
        swy += w * ys[j];
    }

    if sw <= 0.0 {
        return (ys[i], left);
    }

    let xbar = swx / sw;
    let ybar = swy / sw;

    // Slope via weighted covariance / variance.
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for j in lo..hi {
        let w = if h > 0.0 {
            tricube((xs[j] - xi) / h)
        } else {
            1.0
        } * rweights[j];
        let dx = xs[j] - xbar;
        sxx += w * dx * dx;
        sxy += w * dx * ys[j];
    }

    let fitted = if sxx > 1e-12 {
        let slope = sxy / sxx;
        ybar + slope * (xi - xbar)
    } else {
        ybar
    };
    (fitted, left)
}

/// Compute the LOWESS smooth of `endog` against `exog`.
///
/// The data is sorted by `exog`; the returned [`LowessFit`] is aligned to that
/// sorted order. The algorithm fits a tricube-weighted local line at each
/// anchor point and performs `options.it` rounds of bisquare residual
/// reweighting (Cleveland, 1979).
///
/// # Errors
/// Returns an error if the inputs differ in length, are empty, contain
/// non-finite values, or if `frac` is not in `(0, 1]`.
///
/// # Notes
/// With the default `delta = 0` every point is fitted directly and the result
/// matches the reference to machine precision. A positive `delta` enables the
/// reference's linear-interpolation speed-up between anchors that are within
/// `delta` of one another.
pub fn lowess(
    endog: &Array1<f64>,
    exog: &Array1<f64>,
    options: LowessOptions,
) -> Result<LowessFit> {
    let n = exog.len();
    if endog.len() != n {
        return Err(Error::Shape(format!(
            "endog ({}) and exog ({}) must have equal length",
            endog.len(),
            n
        )));
    }
    if n == 0 {
        return Err(Error::Value(
            "lowess requires at least one observation".into(),
        ));
    }
    if !(options.frac > 0.0 && options.frac <= 1.0) {
        return Err(Error::Value("frac must lie in (0, 1]".into()));
    }
    if !options.delta.is_finite() || options.delta < 0.0 {
        return Err(Error::Value(
            "delta must be a non-negative finite number".into(),
        ));
    }
    for (&xv, &yv) in exog.iter().zip(endog.iter()) {
        if !xv.is_finite() || !yv.is_finite() {
            return Err(Error::Value("exog and endog must be finite".into()));
        }
    }

    // Sort by exog (stable, to mirror the reference's mergesort tie handling).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| exog[a].total_cmp(&exog[b]));
    let xs: Vec<f64> = order.iter().map(|&k| exog[k]).collect();
    let ys: Vec<f64> = order.iter().map(|&k| endog[k]).collect();

    // Neighbourhood size r = floor(frac * n), clamped to [2, n].
    let r = ((options.frac * n as f64) as usize).clamp(2.min(n), n);

    let mut fitted = vec![0.0; n];
    let mut rweights = vec![1.0; n];

    for iteration in 0..=options.it {
        let mut left = 0usize;
        // Always fit the first point.
        let (f0, l0) = local_fit(&xs, &ys, &rweights, r, 0, left);
        fitted[0] = f0;
        left = l0;

        let mut i = 0usize;
        while i < n - 1 {
            // Find the last point within `delta` of xs[i] (the next anchor).
            let cut = xs[i] + options.delta;
            let mut last_new = i;
            let mut j = i + 1;
            while j < n && xs[j] <= cut {
                last_new = j;
                j += 1;
            }
            if last_new <= i {
                last_new = i + 1;
            }

            let (f, l) = local_fit(&xs, &ys, &rweights, r, last_new, left);
            fitted[last_new] = f;
            left = l;

            // Linearly interpolate strictly-interior points between the two
            // anchors at indices `i` and `last_new`.
            let denom = xs[last_new] - xs[i];
            let fi = fitted[i];
            let fj = fitted[last_new];
            for k in (i + 1)..last_new {
                fitted[k] = if denom > 0.0 {
                    let alpha = (xs[k] - xs[i]) / denom;
                    (1.0 - alpha) * fi + alpha * fj
                } else {
                    fi
                };
            }
            i = last_new;
        }

        if iteration < options.it {
            // Bisquare reweighting from the absolute residuals.
            let resid: Vec<f64> = (0..n).map(|k| ys[k] - fitted[k]).collect();
            let abs_resid: Vec<f64> = resid.iter().map(|v| v.abs()).collect();
            let s = median(&abs_resid);
            if s == 0.0 {
                for w in rweights.iter_mut() {
                    *w = 1.0;
                }
            } else {
                for (w, &rk) in rweights.iter_mut().zip(resid.iter()) {
                    *w = bisquare(rk / (6.0 * s));
                }
            }
        }
    }

    Ok(LowessFit {
        x: Array1::from_vec(xs),
        fitted: Array1::from_vec(fitted),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn straight_line_is_recovered() {
        // For data exactly on a line, lowess reproduces it.
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y: Array1<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect();
        let fit = lowess(&y, &x, LowessOptions::default()).unwrap();
        for (xi, fi) in fit.x.iter().zip(fit.fitted.iter()) {
            assert!((fi - (2.0 * xi + 1.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn output_is_sorted_by_x() {
        let x = array![3.0, 1.0, 2.0, 5.0, 4.0];
        let y = array![3.0, 1.0, 2.0, 5.0, 4.0];
        let fit = lowess(&y, &x, LowessOptions::default()).unwrap();
        for w in fit.x.windows(2).into_iter() {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn mismatched_lengths_error() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0];
        assert!(lowess(&y, &x, LowessOptions::default()).is_err());
    }
}

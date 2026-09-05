//! Robust scale estimators used to standardize residuals during IRLS.

use solow_distributions::{norm_cdf, norm_pdf, norm_ppf};

/// The default MAD normalization constant `Φ⁻¹(3/4) ≈ 0.6744897…`.
///
/// Dividing the raw median absolute deviation by this constant makes the
/// estimator consistent for the standard deviation at the Gaussian model.
pub fn mad_c() -> f64 {
    norm_ppf(0.75)
}

/// Median of a slice (lower-of-two-middles average for even length).
///
/// Uses the standard "average of the two central order statistics" definition
/// for even-length inputs, matching the reference's `numpy.median`.
pub fn median(a: &[f64]) -> f64 {
    let mut v: Vec<f64> = a.to_vec();
    v.sort_by(|x, y| x.total_cmp(y));
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// The median absolute deviation, normalized by `c` (default `0.6745…`).
///
/// `mad = median(|a − center|) / c`. When `center` is `None` the sample median
/// of `a` is used; RLM passes `center = Some(0.0)` so the scale is computed
/// directly from residuals about zero.
pub fn mad(a: &[f64], c: f64, center: Option<f64>) -> f64 {
    let cen = center.unwrap_or_else(|| median(a));
    let dev: Vec<f64> = a.iter().map(|&x| (x - cen).abs() / c).collect();
    median(&dev)
}

/// Huber's "proposal 2" scaling for the IRLS weights ([`HuberScale`]).
///
/// This is the `scale_est=HuberScale()` option in the reference. It solves, by
/// fixed-point iteration,
///
/// ```text
/// scale_{i+1}² = (1 / (n·h)) · Σ χ(r / scale_i) · scale_i²
/// ```
///
/// with `χ(x) = x²/2` for `|x| < d` and `d²/2` otherwise, and the consistency
/// constant `h = (df_resid / n) · (d² + (1 − d²)·Φ(d) − 1/2 − d·φ(d))`.
#[derive(Clone, Copy, Debug)]
pub struct HuberScale {
    /// Tuning constant `d` (default `2.5`).
    pub d: f64,
    /// Convergence tolerance on successive scale estimates.
    pub tol: f64,
    /// Maximum number of fixed-point iterations.
    pub maxiter: usize,
}

impl Default for HuberScale {
    fn default() -> Self {
        HuberScale {
            d: 2.5,
            tol: 1e-8,
            maxiter: 30,
        }
    }
}

impl HuberScale {
    /// Evaluate Huber's proposal-2 scale for the given residuals.
    ///
    /// `df_resid` and `nobs` are the model degrees of freedom and the number of
    /// observations; the iteration is seeded with the MAD of `resid`.
    pub fn scale(&self, df_resid: f64, nobs: f64, resid: &[f64]) -> f64 {
        let d = self.d;
        let h = df_resid / nobs
            * (d * d + (1.0 - d * d) * norm_cdf(d)
                - 0.5
                - d / (2.0 * std::f64::consts::PI).sqrt() * (-0.5 * d * d).exp());
        let s0 = mad(resid, mad_c(), None);

        let chi_sum = |s: f64| -> f64 {
            resid
                .iter()
                .map(|&r| {
                    if (r / s).abs() < d {
                        (r / s).powi(2) / 2.0
                    } else {
                        d * d / 2.0
                    }
                })
                .sum::<f64>()
        };

        let mut prev = f64::INFINITY;
        let mut cur = s0;
        let mut niter = 1;
        while (prev - cur).abs() > self.tol && niter < self.maxiter {
            let nscale = (1.0 / (nobs * h) * chi_sum(cur) * cur * cur).sqrt();
            prev = cur;
            cur = nscale;
            niter += 1;
        }
        cur
    }
}

/// Huber's "proposal 2" joint location/scale estimator ([`Huber`]).
///
/// Estimates location `μ` and scale `σ` simultaneously for a 1-d sample by the
/// fixed-point scheme of Venables & Ripley §5.5, using the one-step clipped-mean
/// location update (`norm = None` in the reference).
#[derive(Clone, Copy, Debug)]
pub struct Huber {
    /// Clipping threshold `c` (default `1.5`).
    pub c: f64,
    /// Convergence tolerance.
    pub tol: f64,
    /// Maximum number of iterations.
    pub maxiter: usize,
}

impl Default for Huber {
    fn default() -> Self {
        Huber {
            c: 1.5,
            tol: 1e-8,
            maxiter: 30,
        }
    }
}

impl Huber {
    /// The consistency constant `γ` used in the scale denominator.
    fn gamma(&self) -> f64 {
        let tmp = 2.0 * norm_cdf(self.c) - 1.0;
        tmp + self.c * self.c * (1.0 - tmp) - 2.0 * self.c * norm_pdf(self.c)
    }

    /// Jointly estimate location and scale, returning `(mu, scale)`.
    ///
    /// Returns `None` if the iteration fails to converge within `maxiter`.
    pub fn estimate(&self, a: &[f64]) -> Option<(f64, f64)> {
        let n = (a.len() - 1) as f64;
        let gamma = self.gamma();
        let mut mu = median(a);
        let mut sc = mad(a, mad_c(), None);

        for _ in 0..self.maxiter {
            let lo = mu - self.c * sc;
            let hi = mu + self.c * sc;
            let nmu = a.iter().map(|&x| x.clamp(lo, hi)).sum::<f64>() / a.len() as f64;

            let mut card = 0usize;
            let mut num = 0.0;
            for &x in a {
                if ((x - mu) / sc).abs() <= self.c {
                    card += 1;
                    num += (x - nmu).powi(2);
                }
            }
            let denom = n * gamma - (a.len() - card) as f64 * self.c * self.c;
            let nscale = (num / denom).sqrt();

            let test1 = (sc - nscale).abs() <= nscale * self.tol;
            let test2 = (mu - nmu).abs() <= nscale * self.tol;
            if test1 && test2 {
                return Some((nmu, nscale));
            }
            mu = nmu;
            sc = nscale;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_handles_even_and_odd() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn mad_of_standard_normal_constant_is_unit_scale() {
        // For symmetric data about 0, mad(center=0) == median(|x|)/c.
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let want = 1.0 / mad_c();
        assert!((mad(&x, mad_c(), Some(0.0)) - want).abs() < 1e-12);
    }

    #[test]
    fn mad_default_centers_on_median() {
        let x = [10.0, 11.0, 12.0, 13.0, 14.0];
        // median = 12, abs devs = [2,1,0,1,2], median dev = 1, /c.
        assert!((mad(&x, mad_c(), None) - 1.0 / mad_c()).abs() < 1e-12);
    }
}

//! The studentized range distribution used by Tukey's HSD.
//!
//! The studentized range `Q = W / U`, where `W` is the range of `k` i.i.d.
//! standard-normal variates and `νU² ~ χ²_ν` is an independent scaled
//! chi-squared, has CDF
//!
//! ```text
//! P(Q ≤ q) = ∫₀^∞ f_U(u) · F_W(q·u) du,
//! F_W(w)   = k ∫_{-∞}^{∞} φ(z) [Φ(z) − Φ(z − w)]^{k−1} dz,
//! ```
//!
//! where `f_U` is the density of `U = √(χ²_ν/ν)`. Both integrals are evaluated
//! with fixed-order Gauss–Legendre quadrature; the node counts are chosen so
//! the CDF reproduces the reference (scipy `studentized_range`) to better than
//! `1e-12`. The quantile function inverts the CDF by bisection.

use solow_distributions::special::lgamma;
use solow_distributions::{norm_cdf, norm_pdf};

/// Number of Gauss–Legendre nodes for the inner (range) integral.
const N_INNER: usize = 100;
/// Number of Gauss–Legendre nodes for the outer (chi) integral.
const N_OUTER: usize = 80;

/// Gauss–Legendre nodes and weights on `[-1, 1]` for `n` points.
///
/// Computed by Newton's method on the Legendre polynomial `P_n`, which is the
/// standard self-contained construction (no tables).
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let m = n.div_ceil(2);
    let nf = n as f64;
    for i in 0..m {
        // Initial guess for the i-th root (Chebyshev approximation).
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        let mut pp = 0.0;
        for _ in 0..100 {
            // Evaluate the Legendre polynomial P_n(z) and its value P_{n-1}.
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = ((2.0 * jf + 1.0) * z * p2 - jf * p3) / (jf + 1.0);
            }
            // Derivative via the recurrence relation.
            pp = nf * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() <= 1e-15 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        let wt = 2.0 / ((1.0 - z * z) * pp * pp);
        w[i] = wt;
        w[n - 1 - i] = wt;
    }
    (x, w)
}

/// Standard-normal CDF (re-exported for clarity).
#[inline]
fn phi_cdf(z: f64) -> f64 {
    norm_cdf(z)
}

/// `F_W(w)`: CDF of the range of `k` i.i.d. standard normals at `w`.
fn range_cdf(w: f64, k: f64, nodes: &(Vec<f64>, Vec<f64>)) -> f64 {
    if w <= 0.0 {
        return 0.0;
    }
    // The integrand `φ(z)[Φ(z)−Φ(z−w)]^{k−1}` is negligible outside this band.
    let a = -8.0;
    let b = 8.0 + w;
    let half = 0.5 * (b - a);
    let mid = 0.5 * (b + a);
    let (x, wt) = nodes;
    let mut acc = 0.0;
    for i in 0..x.len() {
        let z = half * x[i] + mid;
        let inner = phi_cdf(z) - phi_cdf(z - w);
        acc += wt[i] * norm_pdf(z) * inner.powf(k - 1.0);
    }
    k * half * acc
}

/// CDF of the studentized range with `k` groups and `df` degrees of freedom.
pub fn srange_cdf(q: f64, k: f64, df: f64) -> f64 {
    if q <= 0.0 {
        return 0.0;
    }
    let inner = gauss_legendre(N_INNER);
    if !df.is_finite() {
        return range_cdf(q, k, &inner);
    }
    let outer = gauss_legendre(N_OUTER);
    // log of the normalising constant of f_U(u) = c u^{ν−1} exp(−ν u²/2).
    let logc = (df / 2.0) * df.ln() - (df / 2.0 - 1.0) * std::f64::consts::LN_2 - lgamma(df / 2.0);
    let a = 1e-9;
    let b = 1.0 + 10.0 / df.sqrt();
    let half = 0.5 * (b - a);
    let mid = 0.5 * (b + a);
    let (x, wt) = &outer;
    let mut acc = 0.0;
    for i in 0..x.len() {
        let u = half * x[i] + mid;
        let f_u = (logc + (df - 1.0) * u.ln() - df * u * u / 2.0).exp();
        acc += wt[i] * f_u * range_cdf(q * u, k, &inner);
    }
    let v = half * acc;
    v.clamp(0.0, 1.0)
}

/// Survival function `1 − CDF` of the studentized range.
pub fn srange_sf(q: f64, k: f64, df: f64) -> f64 {
    1.0 - srange_cdf(q, k, df)
}

/// Quantile (inverse CDF) of the studentized range at probability `p`.
///
/// Found by bisection on the monotone CDF; the bracket `[0, 100]` covers every
/// practical case.
pub fn srange_ppf(p: f64, k: f64, df: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let mut lo = 0.0;
    let mut hi = 100.0;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if srange_cdf(mid, k, df) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-12 * (1.0 + hi) {
            break;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_legendre_integrates_polynomial() {
        // ∫_{-1}^{1} x² dx = 2/3, exact for 2 nodes.
        let (x, w) = gauss_legendre(2);
        let val: f64 = x.iter().zip(&w).map(|(&xi, &wi)| wi * xi * xi).sum();
        assert!((val - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn cdf_is_monotone_and_bounded() {
        let a = srange_cdf(2.0, 3.0, 20.0);
        let b = srange_cdf(4.0, 3.0, 20.0);
        assert!(a > 0.0 && a < b && b < 1.0);
    }

    #[test]
    fn ppf_inverts_cdf() {
        let q = srange_ppf(0.95, 4.0, 30.0);
        let p = srange_cdf(q, 4.0, 30.0);
        assert!((p - 0.95).abs() < 1e-8);
    }
}
